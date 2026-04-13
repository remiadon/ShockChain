"""
Fine-tune a sentence transformer on news headlines → SP500 daily-movement bin.

Design notes
------------
* Many-to-one labelling: every headline on day T inherits that day's SP500 bin.
  This is the standard weak-supervision approach for NLP-based market prediction.

* Multi-class target: all `price_`-prefixed columns from sp500_targets.parquet are
  used as classes (one-hot → argmax → class index).  The symmetric cut breaks in
  extract.py produce one bin per magnitude tier in both directions, plus a flat
  band around zero.  `price_null` rows (first observation, no pct_change) are
  excluded automatically.

* Embeddings: the encoder (mean-pooled transformer) is saved alongside the head.
  populate_lancedb.py reloads the full model and uses only the encoder output
  to populate the hybrid-search table.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import dvc.api
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
    def __init__(
        self,
        frame: pl.DataFrame,
    ):
        self.texts  = frame["title"]
        self.labels = frame["label"].to_numpy()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.texts[idx], int(self.labels[idx])


class BatchTokenizer:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[tuple[str, int]]):
        texts, labels = zip(*batch)
        encodings = self.tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encodings, torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HeadlineClassifier(nn.Module):
    """
    Mean-pooled transformer encoder + multi-class classification head.

    class_names is stored on the model so that a loaded checkpoint is
    self-describing — no need to re-derive the class list from the data.
    The mean-pooled encoder output is the reusable sentence embedding.
    """

    def __init__(self, base_model_name: str, class_names: list[str]):
        super().__init__()
        self.class_names = class_names
        self.encoder     = AutoModel.from_pretrained(base_model_name)
        hidden           = self.encoder.config.hidden_size
        self.head        = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, len(class_names)),
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emb    = self._mean_pool(out.last_hidden_state, attention_mask)  # (B, H)
        logits = self.head(emb)                                          # (B, C)
        return logits, emb


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_pairs(
    headlines: pl.DataFrame,
    targets: pl.DataFrame,
    cfg: dict,
) -> tuple[pl.DataFrame, list[str]]:
    """
    Join headlines with SP500 targets on date.
    Returns (pairs, class_names) where pairs has columns:
    - title: str
    - label: int class index
    """
    # All real bins in sorted order; exclude the null sentinel column
    class_names = sorted(
        c for c in targets.columns
        if c.startswith("price_") and c != "price_null"
    )

    # Drop rows where pct_change was null (first observation)
    if "price_null" in targets.columns:
        targets = targets.filter(pl.col("price_null") == 0)

    lag = cfg["target_lag_days"]
    if lag > 0:
        targets = targets.with_columns(
            pl.col("date").shift(-lag).alias("date")
        ).drop_nulls("date")

    df = (
        headlines
        .join(targets.select(["date"] + class_names), on="date", how="inner")
        .drop_nulls("title")
    )

    if cfg["max_samples"] is not None: # TODO : temporal continuity is not enough -> test on cases with high cosine-distance ?
        df = df.sample(int(cfg["max_samples"])).sort('date')

    # one-hot → class index (kept inside Polars to avoid huge Python objects)
    pairs = (
        df
        .with_columns(
            pl.concat_list(class_names).list.arg_max().cast(pl.Int64).alias("label")
        )
        .select(["title", "label"])
    )

    print(
        f"Joined {pairs.height:,} pairs | "
        f"{len(class_names)} classes | "
        f"lag={lag}d"
    )
    return pairs, class_names


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    headlines: pl.DataFrame,
    targets: pl.DataFrame,
    cfg: dict,
) -> HeadlineClassifier:
    pairs, class_names = build_pairs(headlines, targets, cfg)

    # Chronological train/val split
    split    = int((1 - cfg["val_ratio"]) * pairs.height)
    train_df = pairs.slice(0, split)
    val_df   = pairs.slice(split, pairs.height - split)

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    collate  = BatchTokenizer(tokenizer, cfg["max_length"])
    train_ds = HeadlineDataset(train_df)
    val_ds   = HeadlineDataset(val_df)

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
    # float16 for CUDA/MPS (bfloat16 not universally supported on Apple Silicon)
    amp_dtype  = torch.float16 if device.type in ("cuda", "mps") else torch.bfloat16
    grad_accum = int(cfg.get("grad_accum", 1))
    print(f"Device: {device}  |  amp_dtype: {amp_dtype}  |  grad_accum: {grad_accum}")

    model     = HeadlineClassifier(cfg["base_model"], class_names).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()
    log_every = max(1, int(cfg.get("log_every_batches", 200)))

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
            batch_labels = batch_labels.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                logits, _ = model(**batch)
                loss      = criterion(logits, batch_labels) / grad_accum
            loss.backward()
            if step % grad_accum == 0 or step == len(train_dl):
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item() * grad_accum
            if step % log_every == 0 or step == len(train_dl):
                train_iter.set_postfix(loss=f"{running_loss / step:.4f}")

        # ---- validate ----
        model.eval()
        correct = total = 0
        val_iter = tqdm(
            val_dl,
            total=len(val_dl),
            desc=f"Epoch {epoch + 1}/{cfg['epochs']} [val]",
            dynamic_ncols=True,
        )
        with torch.no_grad():
            for step, (batch, batch_labels) in enumerate(val_iter, start=1):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    logits, _ = model(**batch)
                preds    = logits.argmax(dim=-1).cpu()
                correct += (preds == batch_labels).sum().item()
                total   += len(batch_labels)
                if step % log_every == 0 or step == len(val_dl):
                    val_iter.set_postfix(acc=f"{correct / max(1, total):.4f}")

        print(
            f"Epoch {epoch + 1}/{cfg['epochs']} | "
            f"Loss: {running_loss / len(train_dl):.4f} | "
            f"Val Acc: {correct / max(1, total):.4f}"
        )

        # release cached-but-unused memory back to the OS after each epoch
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

    # ---- resource limits (reduce heat on laptops) ----
    # Lower scheduling priority: macOS will yield CPU time under thermal pressure.
    os.nice(cfg.get("nice", 0))
    # Cap CPU thread pools used by PyTorch (affects CPU-side ops and DataLoader).
    if "num_threads" in cfg:
        torch.set_num_threads(cfg["num_threads"])
        torch.set_num_interop_threads(max(1, cfg["num_threads"] // 2))

    headlines = pl.read_parquet(args.headlines)
    targets   = pl.read_parquet(args.targets)

    model = train(headlines, targets, cfg)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    ckpt = output / "headline_classifier.pt"
    # Save the full model object so torch.load() restores it directly without
    # calling __init__ — meaning AutoModel.from_pretrained() is never triggered
    # again when the checkpoint is consumed by populate_lancedb.py.
    torch.save(model, ckpt)
    print(f"Saved → {ckpt}")
