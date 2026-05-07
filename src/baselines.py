"""
Baseline models. All return (ŷ_1d, ŷ_5d, ŷ_15d), each shape (B, 5).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import _head, _trunk


class AlwaysZero(nn.Module):
    """Null hypothesis: events don't move markets."""

    def __init__(self, n_indicators: int = 5):
        super().__init__()
        self.n_indicators = n_indicators
        # Register a parameter-less buffer so .to(device) is a no-op friendly.
        self.register_buffer("_zero", torch.zeros(1, n_indicators))

    def forward(self, x_embed, x_num, y_1d_true=None, y_5d_true=None, noise_std=0.0):
        b = x_embed.shape[0]
        z = self._zero.expand(b, self.n_indicators)
        return z, z, z


def _small_mlp(in_dim: int, hidden: int, n_indicators: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_indicators),
    )


class SentimentOnly(nn.Module):
    """Embedding only, three independent small MLPs (no chain, no x_num)."""

    def __init__(self, embed_dim: int = 768, hidden: int = 64, n_indicators: int = 5):
        super().__init__()
        self.h1 = _small_mlp(embed_dim, hidden, n_indicators)
        self.h2 = _small_mlp(embed_dim, hidden, n_indicators)
        self.h3 = _small_mlp(embed_dim, hidden, n_indicators)

    def forward(self, x_embed, x_num, y_1d_true=None, y_5d_true=None, noise_std=0.0):
        return self.h1(x_embed), self.h2(x_embed), self.h3(x_embed)


class MarketStateOnly(nn.Module):
    """Market state only, three independent small MLPs (no chain, no x_embed)."""

    def __init__(self, num_dim: int = 16, hidden: int = 128, n_indicators: int = 5):
        super().__init__()
        self.h1 = _small_mlp(num_dim, hidden, n_indicators)
        self.h2 = _small_mlp(num_dim, hidden, n_indicators)
        self.h3 = _small_mlp(num_dim, hidden, n_indicators)

    def forward(self, x_embed, x_num, y_1d_true=None, y_5d_true=None, noise_std=0.0):
        return self.h1(x_num), self.h2(x_num), self.h3(x_num)


class IndependentHeads(nn.Module):
    """Same scale as ShockChain (per-stage trunks + shared indicator heads) but
    no chain — each stage predicts from [Q_proj, X_num] only."""

    def __init__(self, embed_dim: int = 768, num_dim: int = 16, bottleneck: int = 64, n_indicators: int = 5):
        super().__init__()
        self.projection = nn.Linear(embed_dim, bottleneck)
        in_dim = bottleneck + num_dim
        self.trunk1 = _trunk(in_dim)
        self.trunk2 = _trunk(in_dim)
        self.trunk3 = _trunk(in_dim)
        self.heads = nn.ModuleList([_head() for _ in range(n_indicators)])

    def _heads_forward(self, h):
        return torch.cat([head(h) for head in self.heads], dim=1)

    def forward(self, x_embed, x_num, y_1d_true=None, y_5d_true=None, noise_std=0.0):
        q = self.projection(x_embed)
        base = torch.cat([q, x_num], dim=1)
        return (
            self._heads_forward(self.trunk1(base)),
            self._heads_forward(self.trunk2(base)),
            self._heads_forward(self.trunk3(base)),
        )
