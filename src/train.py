"""
Train ShockChain + 3 baselines, evaluate AlwaysZero, save checkpoints.

Run: python train.py    (uses defaults; see argparse for overrides)
"""

from __future__ import annotations

import argparse
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from baselines import AlwaysZero, IndependentHeads, MarketStateOnly, SentimentOnly
from dataset import HORIZONS, INDICATORS, load_data
from evaluate import evaluate_model
from model import ShockChain

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "output" / "models"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--noise_std", type=float, default=0.5)
    p.add_argument("--huber_delta", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _val_rmse(model, loader, device) -> float:
    model.eval()
    sq, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x_embed = batch["x_embed"].to(device)
            x_num = batch["x_num"].to(device)
            y1 = batch["y_1d"].to(device)
            y5 = batch["y_5d"].to(device)
            y15 = batch["y_15d"].to(device)
            p1, p2, p3 = model(x_embed, x_num)
            for pred, true in ((p1, y1), (p2, y5), (p3, y15)):
                sq += float(((pred - true) ** 2).sum().item())
                n += pred.numel()
    return math.sqrt(sq / max(n, 1))


def train_one(
    model: nn.Module,
    name: str,
    loaders: dict,
    args: argparse.Namespace,
    device: str = "cpu",
) -> nn.Module:
    model = model.to(device)
    has_params = any(p.requires_grad for p in model.parameters())
    if not has_params:
        print(f"\n[{name}] no parameters — skipping training.")
        return model

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.max_epochs, eta_min=1e-5,
    )
    huber = nn.HuberLoss(delta=args.huber_delta, reduction="mean")

    print(f"\n[{name}] training (params={sum(p.numel() for p in model.parameters()):,})")

    best_val = float("inf")
    best_state = deepcopy(model.state_dict())
    bad = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_losses = []
        for batch in loaders["train"]:
            x_embed = batch["x_embed"].to(device)
            x_num = batch["x_num"].to(device)
            y1 = batch["y_1d"].to(device)
            y5 = batch["y_5d"].to(device)
            y15 = batch["y_15d"].to(device)

            optim.zero_grad()
            if isinstance(model, ShockChain):
                p1, p2, p3 = model(x_embed, x_num, y_1d_true=y1, y_5d_true=y5, noise_std=args.noise_std)
            else:
                p1, p2, p3 = model(x_embed, x_num)

            loss = huber(p1, y1) + huber(p2, y5) + huber(p3, y15)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            train_losses.append(float(loss.item()))

        sched.step()
        val = _val_rmse(model, loaders["val"], device)

        if val < best_val - 1e-6:
            best_val = val
            best_state = deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1

        if epoch == 1 or epoch % 10 == 0 or bad >= args.patience:
            lr = optim.param_groups[0]["lr"]
            print(
                f"  epoch {epoch:3d}  train_loss={np.mean(train_losses):.4f}  "
                f"val_rmse={val:.4f}  best={best_val:.4f}  lr={lr:.2e}"
            )

        if bad >= args.patience:
            print(f"  early stop at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    model.load_state_dict(best_state)
    print(f"[{name}] restored best val RMSE: {best_val:.4f}")
    return model


def train_all_models(data: dict, args: argparse.Namespace | None = None, device: str = "cpu") -> dict:
    if args is None:
        args = parse_args()
    set_seed(args.seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    embed_dim = data["embed_dim"]
    num_dim = data["num_dim"]
    n_ind = data["n_indicators"]
    loaders = data["loaders"]

    models = {
        "AlwaysZero": AlwaysZero(n_indicators=n_ind),
        "SentimentOnly": SentimentOnly(embed_dim=embed_dim, n_indicators=n_ind),
        "MarketStateOnly": MarketStateOnly(num_dim=num_dim, n_indicators=n_ind),
        "IndependentHeads": IndependentHeads(embed_dim=embed_dim, num_dim=num_dim, n_indicators=n_ind),
        "ShockChain": ShockChain(embed_dim=embed_dim, num_dim=num_dim, n_indicators=n_ind),
    }

    results = {}
    for name, model in models.items():
        set_seed(args.seed)  # same init across models for fair comparison
        if name == "AlwaysZero":
            print(f"\n[{name}] no training, evaluating directly.")
        else:
            model = train_one(model, name, loaders, args, device=device)
            torch.save(model.state_dict(), MODELS_DIR / f"{_to_filename(name)}.pt")

        metrics = evaluate_model(model, loaders["test"], INDICATORS, HORIZONS, device=device)
        results[name] = metrics

    return results


def _to_filename(name: str) -> str:
    return {
        "AlwaysZero": "always_zero",
        "SentimentOnly": "sentiment_only",
        "MarketStateOnly": "market_state_only",
        "IndependentHeads": "independent_heads",
        "ShockChain": "shockchain",
    }[name]


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    data = load_data(batch_size=args.batch_size)
    train_all_models(data, args=args, device="cpu")
