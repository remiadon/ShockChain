"""
Fine-tune a sentence transformer on news headlines → multi-instrument market bins.

Design notes
------------
* Many-to-one labelling: every headline on day T inherits that day's market bins
  for all instruments (^GSPC, ^VIX, DX-Y.NYB, WTI, ^TNX).
  This is the standard weak-supervision approach for NLP-based market prediction.

* Multi-task target: one classification head per instrument. Targets are Polars
  categoricals; .to_physical() gives the integer class index directly — no
  one-hot encoding needed.

* Embeddings: the encoder (mean-pooled transformer) is saved alongside the heads.
  populate_lancedb.py reloads the full model and uses only the encoder output
  to populate the hybrid-search table.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import dvc.api
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# CLI  (paths only — hyperparams come from params.yaml via dvc.api)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--headlines", default="data/news_headlines.parquet")
parser.add_argument("--targets",   default="data/sp500_targets.parquet")
parser.add_argument("--output",    default="models")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _key(name: str) -> str:
    """Sanitize an instrument name for use as an nn.ModuleDict key."""
    return re.sub(r"\W", "_", name)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HeadlineDataset(Dataset):
    def __init__(self, frame: pl.DataFrame, instruments: list[str]):
        self.texts  = frame["title"]
        # shape (N, num_instruments) — integer class indices
        self.labels = frame.select(instruments).to_numpy()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.texts[idx], self.labels[idx]  # str, ndarray(num_instruments,)


class BatchTokenizer:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[tuple[str, np.ndarray]]):
        texts, labels = zip(*batch)
        encodings = self.tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encodings, torch.tensor(np.stack(labels), dtype=torch.long)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HeadlineClassifier(nn.Module):
    """
    Mean-pooled transformer encoder + one classification head per instrument.

    heads_meta maps instrument name → list of category label strings (in
    physical/ordinal order matching the integer indices from .to_physical()).
    It is stored on the model so a loaded checkpoint is self-describing.
    The mean-pooled encoder output is the reusable sentence embedding.
    """

    def __init__(self, base_model_name: str, heads_meta: dict[str, list[str]]):
        super().__init__()
        self.heads_meta = heads_meta
        self.encoder    = AutoModel.from_pretrained(base_model_name)
        hidden          = self.encoder.config.hidden_size
        self.heads      = nn.ModuleDict({
            _key(name): nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden, len(labels)),
            )
            for name, labels in heads_meta.items()
        })

    @staticmethod
    def _mean_pool(
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        return torch.sum(last_hidden * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emb = self._mean_pool(out.last_hidden_state, attention_mask)  # (B, H)
        logits = {name: head(emb) for name, head in self.heads.items()}
        return logits, emb


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_pairs(
    headlines: pl.DataFrame,
    targets: pl.DataFrame,
    cfg: dict,
) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """
    Join headlines with market targets on date.

    Returns (pairs, heads_meta) where:
    - pairs has columns: title (str) + one int64 column per instrument
    - heads_meta maps instrument name → list of category label strings in
      physical (ordinal) order, matching the integer indices in pairs
    """
    instruments = [c for c in targets.columns if c != "date"]

    # Collect category labels BEFORE converting to integer indices.
    # cat.get_categories() returns labels in physical (ordinal) order,
    # so index i corresponds to heads_meta[inst][i].
    heads_meta: dict[str, list[str]] = {
        inst: targets[inst].cat.get_categories().to_list()
        for inst in instruments
    }

    # Replace categorical columns with their integer physical codes.
    targets = targets.with_columns([
        pl.col(inst).to_physical().cast(pl.Int64)
        for inst in instruments
    ])

    lag = cfg["target_lag_days"]
    if lag > 0:
        targets = targets.with_columns(
            pl.col("date").shift(-lag).alias("date")
        ).drop_nulls("date")

    pairs = (
        headlines
        .join(targets.select(["date"] + instruments), on="date", how="inner")
        .drop_nulls("title")
    )

    if cfg["max_samples"] is not None:
        pairs = pairs.sample(int(cfg["max_samples"])).sort("date")

    pairs = pairs.select(["title"] + instruments)  # reorder columns

    print(
        f"Joined {pairs.height:,} pairs | "
        f"{len(instruments)} instruments | "
        f"lag={lag}d"
    )
    return pairs, heads_meta


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    headlines: pl.DataFrame,
    targets: pl.DataFrame,
    cfg: dict,
) -> HeadlineClassifier:
    pairs, heads_meta = build_pairs(headlines, targets, cfg)
    instruments = list(heads_meta.keys())

    # Chronological train/val split
    split    = int((1 - cfg["val_ratio"]) * pairs.height)
    train_df = pairs.slice(0, split)
    val_df   = pairs.slice(split, pairs.height - split)

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    collate   = BatchTokenizer(tokenizer, cfg["max_length"])
    train_ds  = HeadlineDataset(train_df, instruments)
    val_ds    = HeadlineDataset(val_df, instruments)

    num_workers = int(cfg.get("num_workers", 0))
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        persistent_workers=num_workers > 0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        persistent_workers=num_workers > 0,
    )

    device     = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    amp_dtype  = torch.float16 if device.type in ("cuda", "mps") else torch.bfloat16
    grad_accum = int(cfg.get("grad_accum", 1))
    print(f"Device: {device}  |  amp_dtype: {amp_dtype}  |  grad_accum: {grad_accum}")

    model     = HeadlineClassifier(cfg["base_model"], heads_meta).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg["epochs"]):
        # ---- train ----
        model.train()
        running_loss = 0.0
        train_iter = tqdm(
            train_dl,
            total=len(train_dl),
            desc=f"Epoch {epoch + 1}/{cfg['epochs']} [train]",
            dynamic_ncols=True,
        )

        optimizer.zero_grad()
        for step, (batch, batch_labels) in enumerate(train_iter, start=1):
            batch        = {k: v.to(device) for k, v in batch.items()}
            batch_labels = batch_labels.to(device)        # (B, num_instruments)
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                logits_dict, _ = model(**batch)
                loss = sum(
                    criterion(logits_dict[_key(inst)], batch_labels[:, i])
                    for i, inst in enumerate(instruments)
                ) / len(instruments) / grad_accum
            loss.backward()
            if step % grad_accum == 0 or step == len(train_dl):
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item() * grad_accum * len(instruments)
            train_iter.set_postfix(loss=f"{running_loss / step:.4f}")

        # ---- validate ----
        model.eval()
        correct = {inst: 0 for inst in instruments}
        total   = 0
        val_iter = tqdm(
            val_dl,
            total=len(val_dl),
            desc=f"Epoch {epoch + 1}/{cfg['epochs']} [val]",
            dynamic_ncols=True,
        )
        with torch.no_grad():
            for batch, batch_labels in val_iter:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    logits_dict, _ = model(**batch)
                for i, inst in enumerate(instruments):
                    preds = logits_dict[_key(inst)].argmax(dim=-1).cpu()
                    correct[inst] += (preds == batch_labels[:, i]).sum().item()
                total += len(batch_labels)

        acc_str = "  ".join(
            f"{inst}: {correct[inst] / max(1, total):.3f}"
            for inst in instruments
        )
        print(
            f"Epoch {epoch + 1}/{cfg['epochs']} | "
            f"Loss: {running_loss / len(train_dl):.4f} | "
            f"Val Acc — {acc_str}"
        )

        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parser.parse_args()
    cfg  = dvc.api.params_show()["train"]

    os.nice(cfg.get("nice", 0))
    if "num_threads" in cfg:
        torch.set_num_threads(cfg["num_threads"])
        torch.set_num_interop_threads(max(1, cfg["num_threads"] // 2))

    headlines = pl.read_parquet(args.headlines)
    targets   = pl.read_parquet(args.targets)

    model = train(headlines, targets, cfg)

    ckpt = Path(args.output)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    # Save the full model object so torch.load() restores it directly without
    # calling __init__ — meaning AutoModel.from_pretrained() is never triggered
    # again when the checkpoint is consumed by populate_lancedb.py.
    torch.save(model, ckpt)
    print(f"Saved → {ckpt}")
