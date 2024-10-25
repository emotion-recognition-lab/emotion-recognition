from __future__ import annotations

import torch
from torch import nn


class Projector(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int | None = None,
        bias: bool = False,
        *,
        depth: int = 1,
    ):
        super().__init__()
        if out_features is None:
            out_features = in_features
        self.proj = nn.Sequential(
            nn.Linear(in_features, out_features, bias=bias),
            *[nn.Tanh(), nn.Linear(out_features, out_features, bias=bias)] * (depth - 1),
        )

    @torch.compile(dynamic=True, fullgraph=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
