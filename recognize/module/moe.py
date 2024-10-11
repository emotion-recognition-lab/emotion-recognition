from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial

import torch
from torch import nn

from .basic import Pooler
from .fusion import LowRankFusionLayer


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


class MultiHeadMoE(nn.Module):
    def __init__(
        self, router: Callable[[Mapping[str, torch.Tensor | None]], torch.Tensor], experts: Mapping[str, nn.Module]
    ):
        super().__init__()
        self.router = router
        self.expert_names = sorted(experts.keys())
        self.experts = nn.ModuleDict(experts)

    def forward(self, inputs: Mapping[str, torch.Tensor | None]) -> torch.Tensor:
        gate_outputs = self.router(inputs)
        sum_weights = torch.zeros(gate_outputs.shape[0], device=gate_outputs.device)
        outputs = []
        for i, name in enumerate(self.expert_names):
            if name not in inputs or inputs[name] is None or name not in self.experts:
                continue
            expert_outputs = self.experts[name](inputs[name])
            outputs.append(gate_outputs[:, i : i + 1] * expert_outputs)
            sum_weights += gate_outputs[:, i]
        return sum(outputs) / sum_weights.unsqueeze(1)


class MoELowRankFusionLayer(MultiHeadMoE, LowRankFusionLayer):
    def __init__(self, dims: dict[str, int], rank: int, output_size: int, *, trainable_placeholder: bool = False):
        LowRankFusionLayer.__init__(self, dims, rank, len(dims), trainable_placeholder=trainable_placeholder)
        self.router = partial(LowRankFusionLayer.forward, self)
        self.expert_names = sorted(dims.keys())
        self.experts = nn.ModuleDict({name: Pooler(dim, output_size) for name, dim in dims.items()})
        self.output_size = output_size
