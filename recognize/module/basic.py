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


class MoE(nn.Module):
    def __init__(self, input_dim: int, experts: list[nn.Module]):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.router = nn.Sequential(
            nn.Linear(input_dim, len(experts)),
            nn.Softmax(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_outputs = self.router(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.sum(gate_outputs.unsqueeze(2) * expert_outputs, dim=1)
        return output
