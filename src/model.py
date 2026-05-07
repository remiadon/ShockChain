"""
ShockChain three-stage chained regression model.

Stage 1 input:  [Q_proj | x_num]                   -> ŷ_1d  (5 indicators)
Stage 2 input:  [Q_proj | x_num | y_1d]            -> ŷ_5d
Stage 3 input:  [Q_proj | x_num | y_5d]            -> ŷ_15d

During training, stages 2 and 3 are fed the *true* previous-stage targets plus
Gaussian noise (teacher forcing with noise injection). At eval, the model's
own outputs are chained through.
"""

from __future__ import annotations

import torch
import torch.nn as nn


TRUNK_HIDDEN1 = 64
TRUNK_HIDDEN2 = 32
TRUNK_DROPOUT = 0.5
HEAD_HIDDEN = 32
HEAD_DROPOUT = 0.4


def _trunk(in_dim: int, hidden1: int = TRUNK_HIDDEN1, hidden2: int = TRUNK_HIDDEN2, dropout: float = TRUNK_DROPOUT) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden1),
        nn.BatchNorm1d(hidden1),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden1, hidden2),
        nn.BatchNorm1d(hidden2),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


def _head(trunk_out: int = TRUNK_HIDDEN2, head_hidden: int = HEAD_HIDDEN, dropout: float = HEAD_DROPOUT) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(trunk_out, head_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(head_hidden, 1),
    )


class ShockChain(nn.Module):
    """Three-stage chain with per-stage trunks and SHARED indicator heads.

    Heads are reused across all 3 stages — keeps capacity tight (the dataset
    is tiny) and forces the heads to learn an indicator-level projection that
    works at every horizon.
    """

    def __init__(self, embed_dim: int = 768, num_dim: int = 16, bottleneck: int = 64, n_indicators: int = 5):
        super().__init__()
        self.n_indicators = n_indicators
        self.projection = nn.Linear(embed_dim, bottleneck)
        in1 = bottleneck + num_dim
        in2 = bottleneck + num_dim + n_indicators
        self.trunk1 = _trunk(in1)
        self.trunk2 = _trunk(in2)
        self.trunk3 = _trunk(in2)
        # Shared heads — one per indicator, reused across all 3 stages.
        self.heads = nn.ModuleList([_head() for _ in range(n_indicators)])

    def _heads_forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(h) for head in self.heads], dim=1)

    def forward(
        self,
        x_embed: torch.Tensor,
        x_num: torch.Tensor,
        y_1d_true: torch.Tensor | None = None,
        y_5d_true: torch.Tensor | None = None,
        noise_std: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.projection(x_embed)
        base = torch.cat([q, x_num], dim=1)

        y1 = self._heads_forward(self.trunk1(base))

        if self.training and y_1d_true is not None:
            in2 = y_1d_true + torch.randn_like(y_1d_true) * noise_std
        else:
            in2 = y1
        y2 = self._heads_forward(self.trunk2(torch.cat([base, in2], dim=1)))

        if self.training and y_5d_true is not None:
            in3 = y_5d_true + torch.randn_like(y_5d_true) * noise_std
        else:
            in3 = y2
        y3 = self._heads_forward(self.trunk3(torch.cat([base, in3], dim=1)))

        return y1, y2, y3

    def predict_with_override(
        self,
        x_embed: torch.Tensor,
        x_num: torch.Tensor,
        override_1d: torch.Tensor | None = None,
        override_5d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            q = self.projection(x_embed)
            base = torch.cat([q, x_num], dim=1)
            y1 = self._heads_forward(self.trunk1(base))
            in2 = override_1d if override_1d is not None else y1
            y2 = self._heads_forward(self.trunk2(torch.cat([base, in2], dim=1)))
            in3 = override_5d if override_5d is not None else y2
            y3 = self._heads_forward(self.trunk3(torch.cat([base, in3], dim=1)))
        if was_training:
            self.train()
        return y1, y2, y3
