"""
Data loading + temporal split + PyTorch Datasets/DataLoaders.

Loads:
  - output/embeddings.npy            shape (n_raw, 768)
  - output/X_num.npy                 shape (n_train, n_feat)
  - output/training_dataset.parquet  rows aligned with X_num, has event_id

The parquet's event_id maps each surviving row back to its position in
events_verified.csv, which is also the row order in embeddings.npy.

Targets (y, vol-normalized): 15 columns in spec order
  SPX_1d, VIX_1d, DXY_1d, WTI_1d, US10Y_1d,
  SPX_5d, ..., US10Y_5d,
  SPX_15d, ..., US10Y_15d.
"""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
EMBEDDINGS_NPY = OUTPUT_DIR / "embeddings.npy"
XNUM_NPY = OUTPUT_DIR / "X_num.npy"
YTARGETS_NPY = OUTPUT_DIR / "y_targets.npy"
TRAIN_PARQUET = OUTPUT_DIR / "training_dataset.parquet"

INDICATORS = ["SPX", "VIX", "DXY", "WTI", "US10Y"]
HORIZONS = [1, 5, 15]
TARGET_COLS = [f"{ind}_{h}d_z" for h in HORIZONS for ind in INDICATORS]

VAL_START = date(2022, 1, 1)
TEST_START = date(2024, 1, 1)


class ShockDataset(Dataset):
    def __init__(self, x_embed: np.ndarray, x_num: np.ndarray, y: np.ndarray):
        self.x_embed = torch.from_numpy(x_embed).float()
        self.x_num = torch.from_numpy(x_num).float()
        # y stored as (N, 3, 5): horizon, indicator
        y3 = y.reshape(-1, len(HORIZONS), len(INDICATORS))
        self.y = torch.from_numpy(y3).float()

    def __len__(self) -> int:
        return self.x_embed.shape[0]

    def __getitem__(self, idx):
        return {
            "x_embed": self.x_embed[idx],
            "x_num": self.x_num[idx],
            "y_1d": self.y[idx, 0],
            "y_5d": self.y[idx, 1],
            "y_15d": self.y[idx, 2],
        }


def load_data(batch_size: int = 64, num_workers: int = 0) -> dict:
    embeddings = np.load(EMBEDDINGS_NPY).astype(np.float32)
    x_num_all = np.load(XNUM_NPY).astype(np.float32)
    df = pl.read_parquet(TRAIN_PARQUET).sort("event_id")

    if df.height != x_num_all.shape[0]:
        raise RuntimeError(
            f"Parquet rows ({df.height}) != X_num rows ({x_num_all.shape[0]}). "
            "Re-run build_features.py."
        )

    event_ids = df["event_id"].to_numpy()
    if event_ids.max() >= embeddings.shape[0]:
        raise RuntimeError(
            f"event_id max ({event_ids.max()}) >= embeddings rows ({embeddings.shape[0]}). "
            "Re-run embed_headlines.py against the current events_verified.csv."
        )
    x_embed_all = embeddings[event_ids]

    y_all = df.select(TARGET_COLS).to_numpy().astype(np.float32)
    np.save(YTARGETS_NPY, y_all)

    n_before = df.height
    valid = np.isfinite(x_num_all).all(axis=1) & np.isfinite(y_all).all(axis=1)
    n_dropped = int((~valid).sum())
    if n_dropped:
        print(f"Dropped {n_dropped} rows with non-finite values in x_num or y_targets")
    df = df.filter(pl.Series(valid))
    x_embed_all = x_embed_all[valid]
    x_num_all = x_num_all[valid]
    y_all = y_all[valid]

    dates = df["date"].to_numpy()
    train_mask = dates < np.datetime64(VAL_START)
    val_mask = (dates >= np.datetime64(VAL_START)) & (dates < np.datetime64(TEST_START))
    test_mask = dates >= np.datetime64(TEST_START)

    splits = {}
    for name, m in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        idx = np.where(m)[0]
        splits[name] = ShockDataset(x_embed_all[idx], x_num_all[idx], y_all[idx])

    print(
        f"Total events: {n_before - n_dropped}  |  "
        f"train: {len(splits['train'])}  val: {len(splits['val'])}  test: {len(splits['test'])}"
    )

    loaders = {
        "train": DataLoader(splits["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False),
        "val": DataLoader(splits["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(splits["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    return {
        "datasets": splits,
        "loaders": loaders,
        "embed_dim": embeddings.shape[1],
        "num_dim": x_num_all.shape[1],
        "n_indicators": len(INDICATORS),
        "indicators": INDICATORS,
        "horizons": HORIZONS,
    }


if __name__ == "__main__":
    load_data()
