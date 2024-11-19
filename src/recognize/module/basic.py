from __future__ import annotations

import itertools
from collections.abc import Mapping

import torch
from torch import nn


class Projector(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int | None = None,
        bias: bool = True,
        *,
        depth: int = 1,
    ):
        super().__init__()
        if out_features is None:
            out_features = in_features
        self.proj = nn.Sequential(
            nn.Linear(in_features, out_features, bias=bias),
            *itertools.chain(
                *[[nn.GELU(), nn.Linear(out_features, out_features, bias=bias)] for _ in range(depth - 1)]
            ),
        )

    @torch.compile(dynamic=True, fullgraph=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SelfAttentionProjector(Projector):
    def __init__(
        self,
        in_features: int,
        out_features: int | None = None,
        bias: bool = True,
        *,
        depth: int = 1,
        head_num: int = 8,
    ):
        super().__init__(in_features, out_features, bias, depth=depth)
        self.multihead_attn = nn.MultiheadAttention(in_features, head_num)
        self.layer_norm = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.1),
        )

    @torch.compile(dynamic=True, fullgraph=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention, _ = self.multihead_attn(x, x, x)
        return self.proj(self.layer_norm(attention + x))


class Pooler(nn.Module):
    def __init__(self, dims: Mapping[str, int], out_features: int | None = None):
        super().__init__()
        if out_features is None:
            out_features = max(dims.values())
        self.proj = nn.ModuleDict({modal: nn.Linear(dim, out_features) for modal, dim in dims.items()})

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {modal: self.proj[modal](inputs[modal]) for modal in self.proj.keys()}
