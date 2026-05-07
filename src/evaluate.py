"""
Evaluation metrics for the regression task.

evaluate_model returns a nested dict:
  per_indicator_horizon: {indicator: {horizon: {rmse, mae, dir_acc, r2, tail_rmse}}}
  per_horizon:           {horizon: {avg_rmse, avg_dir_acc}}
  overall:               {avg_rmse, avg_dir_acc}
"""

from __future__ import annotations

import numpy as np
import torch

DIR_ZERO_BAND = 0.1   # |y| < this is excluded from directional accuracy
TAIL_THRESHOLD = 2.0  # |y| > this defines a tail event


@torch.no_grad()
def _collect_predictions(model, dataloader, device):
    model.eval()
    preds = {h: [] for h in (0, 1, 2)}
    truths = {h: [] for h in (0, 1, 2)}
    for batch in dataloader:
        x_embed = batch["x_embed"].to(device)
        x_num = batch["x_num"].to(device)
        y1, y2, y3 = model(x_embed, x_num)
        for h_idx, (yh, key) in enumerate([(y1, "y_1d"), (y2, "y_5d"), (y3, "y_15d")]):
            preds[h_idx].append(yh.cpu().numpy())
            truths[h_idx].append(batch[key].numpy())
    p = {h: np.concatenate(preds[h], axis=0) for h in preds}
    t = {h: np.concatenate(truths[h], axis=0) for h in truths}
    return p, t


def _metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    err = pred - true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    mask = np.abs(true) >= DIR_ZERO_BAND
    if mask.any():
        dir_acc = float(np.mean(np.sign(pred[mask]) == np.sign(true[mask])))
    else:
        dir_acc = float("nan")

    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    tail_mask = np.abs(true) > TAIL_THRESHOLD
    if tail_mask.any():
        tail_rmse = float(np.sqrt(np.mean(err[tail_mask] ** 2)))
    else:
        tail_rmse = float("nan")

    return {"rmse": rmse, "mae": mae, "dir_acc": dir_acc, "r2": r2, "tail_rmse": tail_rmse}


def evaluate_model(model, dataloader, indicator_names, horizons, device: str = "cpu") -> dict:
    p, t = _collect_predictions(model, dataloader, device)

    per_ind_h: dict = {ind: {} for ind in indicator_names}
    per_horizon: dict = {}
    all_rmse, all_dir = [], []

    for h_idx, h in enumerate(horizons):
        h_rmse, h_dir = [], []
        for i_idx, ind in enumerate(indicator_names):
            m = _metrics(p[h_idx][:, i_idx], t[h_idx][:, i_idx])
            per_ind_h[ind][h] = m
            h_rmse.append(m["rmse"])
            if not np.isnan(m["dir_acc"]):
                h_dir.append(m["dir_acc"])
        per_horizon[h] = {
            "avg_rmse": float(np.mean(h_rmse)),
            "avg_dir_acc": float(np.mean(h_dir)) if h_dir else float("nan"),
        }
        all_rmse.extend(h_rmse)
        all_dir.extend(h_dir)

    overall = {
        "avg_rmse": float(np.mean(all_rmse)),
        "avg_dir_acc": float(np.mean(all_dir)) if all_dir else float("nan"),
    }

    return {
        "per_indicator_horizon": per_ind_h,
        "per_horizon": per_horizon,
        "overall": overall,
    }
