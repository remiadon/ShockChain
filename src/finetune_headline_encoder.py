"""
Fine-tune a sentence transformer on news headlines → multi-instrument return prediction.

Design notes
------------
* Many-to-one labelling: every headline on day T inherits that day's pct_change
  for all instruments (^GSPC, ^VIX, DX-Y.NYB, WTI, ^TNX).
  This is the standard weak-supervision approach for NLP-based market prediction.

* Multi-task regression: one scalar output head per instrument, trained with MSELoss.
  Validation reports MAE per instrument.

* Embeddings: the encoder (mean-pooled transformer) is saved alongside the heads.
  populate_lancedb.py reloads the full model and uses only the encoder output
  to populate the hybrid-search table.
"""

from __future__ import annotations

import argparse
import os
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
        return encodings, torch.tensor(np.stack(labels), dtype=torch.float)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HeadlineClassifier(nn.Module):
    """
    Mean-pooled transformer encoder + single regression head over all instruments.

    One Linear(hidden, N) is equivalent to N×Linear(hidden, 1) but cleaner.
    instruments is stored on the model so a loaded checkpoint is self-describing.
    The mean-pooled encoder output is the reusable sentence embedding.
    """

    def __init__(self, base_model_name: str, instruments: list[str]):
        super().__init__()
        self.instruments = instruments
        self.encoder     = AutoModel.from_pretrained(base_model_name)
        hidden           = self.encoder.config.hidden_size
        self.head        = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, len(instruments)),
        )

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
        emb  = self._mean_pool(out.last_hidden_state, attention_mask)  # (B, H)
        preds = self.head(emb)                                          # (B, N)
        return preds, emb


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_pairs(
    headlines: pl.DataFrame,
    targets: pl.DataFrame,
    cfg: dict,
) -> tuple[pl.DataFrame, list[str]]:
    """
    Join headlines with market targets on date.

    Returns (pairs, instruments) where:
    - pairs has columns: title (str) + one float column per instrument (pct_change)
    - instruments is the ordered list of target column names
    """
    instruments = [c for c in targets.columns if c != "date"]

    lag = cfg["target_lag_days"]
    if lag > 0:
        targets = targets.with_columns(
            pl.col("date").shift(-lag).alias("date")
        ).drop_nulls("date")

    pairs = (
        headlines
        .join(targets.select(["date"] + instruments), on="date", how="inner")
        .drop_nulls("title")
        .drop_nulls(instruments)
    )

    if cfg["max_samples"] is not None:
        pairs = pairs.sample(int(cfg["max_samples"])).sort("date")
    pairs = pairs.select(["title"] + instruments)

    print(
        f"Joined {pairs.height:,} pairs | "
        f"{len(instruments)} instruments | "
        f"lag={lag}d"
    )
    return pairs, instruments


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    headlines: pl.DataFrame,
    targets: pl.DataFrame,
    cfg: dict,
) -> HeadlineClassifier:
    pairs, instruments = build_pairs(headlines, targets, cfg)

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

    model     = HeadlineClassifier(cfg["base_model"], instruments).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    criterion = nn.MSELoss()

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
                preds, _ = model(**batch)                   # (B, N)
                loss = criterion(preds, batch_labels) / grad_accum
            loss.backward()
            if step % grad_accum == 0 or step == len(train_dl):
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item() * grad_accum
            train_iter.set_postfix(loss=f"{running_loss / step:.6f}")

        # ---- validate ----
        model.eval()
        abs_err = {inst: 0.0 for inst in instruments}
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
                    preds, _ = model(**batch)               # (B, N)
                preds = preds.float().cpu()
                for i, inst in enumerate(instruments):
                    abs_err[inst] += (preds[:, i] - batch_labels[:, i]).abs().sum().item()
                total += len(batch_labels)

        mae_str = "  ".join(
            f"{inst}: {abs_err[inst] / max(1, total):.5f}"
            for inst in instruments
        )
        print(
            f"Epoch {epoch + 1}/{cfg['epochs']} | "
            f"Loss: {running_loss / len(train_dl):.6f} | "
            f"Val MAE — {mae_str}"
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
