from __future__ import annotations

import torch
from torch import nn


class Pooler(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int | None = None,
        bias: bool = True,
    ):
        super().__init__()
        if out_features is None:
            out_features = in_features
        self.pool = nn.Sequential(
            nn.Linear(in_features, out_features, bias=bias),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled_output = self.pool(x)
        return pooled_output
