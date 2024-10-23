from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from .basic import Pooler
from .fusion import LowRankFusionLayer


class MoE(nn.Module):
    def __init__(self, feature_size: int, experts: Sequence[nn.Module]):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.router = nn.Sequential(
            nn.Linear(feature_size, len(experts)),
            nn.Softmax(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_outputs = self.router(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.sum(gate_outputs.unsqueeze(2) * expert_outputs, dim=1)
        return output


class SparseMoE(nn.Module):
    def __init__(self, feature_size: int, experts: Sequence[nn.Module], *, act_expert_num: int = 1):
        super().__init__()
        self.feature_size = feature_size
        self.experts = nn.ModuleList(experts)
        self.router = nn.Sequential(
            nn.Linear(feature_size, len(experts)),
            nn.Softmax(1),
        )
        self.act_expert_num = act_expert_num

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_experts = len(self.experts)
        batch_size = x.size(0)

        routing_weights = self.router(x)
        routing_weights, selected_experts = torch.topk(routing_weights, self.act_expert_num, dim=1)
        routing_weights /= routing_weights.sum(dim=1, keepdim=True)

        # shape of expert_mask: (batch_size, num_experts, act_expert_num)
        expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(1, 2, 0)

        # TODO: to find a way to support different output size
        output = torch.zeros((batch_size, self.feature_size), dtype=x.dtype, device=x.device)
        for expert_idx in range(num_experts):
            expert_layer = self.experts[expert_idx]
            batch_idx, act_expert_idx = torch.nonzero(expert_mask[expert_idx], as_tuple=True)
            if batch_idx.shape[0] == 0:
                continue

            expert_output = expert_layer(x[batch_idx]) * routing_weights[batch_idx, act_expert_idx, None]
            output.index_add_(0, batch_idx, expert_output)
        return output


class MultiHeadMoE(nn.Module):
    def __init__(self, router: Callable[[Mapping[str, torch.Tensor]], torch.Tensor], experts: Mapping[str, nn.Module]):
        super().__init__()
        self.router = router
        self.expert_names = sorted(experts.keys())
        self.experts = nn.ModuleDict(experts)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        routing_weights = torch.softmax(self.router(inputs), dim=-1)
        sum_weights = torch.zeros(routing_weights.shape[0], device=routing_weights.device)
        outputs = []
        for i, name in enumerate(self.expert_names):
            if name not in inputs or inputs[name] is None or name not in self.experts:
                continue
            expert_outputs = self.experts[name](inputs[name])
            outputs.append(routing_weights[:, i : i + 1] * expert_outputs)
            sum_weights += routing_weights[:, i]
        return sum(outputs) / sum_weights.unsqueeze(1)


class MultimodalMoE(MultiHeadMoE, LowRankFusionLayer):
    def __init__(self, dims: dict[str, int], rank: int, output_size: int, *, trainable_placeholder: bool = False):
        # NOTE: __init__ of MultiHeadMoE or LowRankFusionLayer all has Module.__init__, so only one can be used.
        LowRankFusionLayer.__init__(self, dims, rank, len(dims), trainable_placeholder=trainable_placeholder)
        self.router = partial(LowRankFusionLayer.forward, self)
        self.expert_names = sorted(dims.keys())
        self.experts = nn.ModuleDict({name: Pooler(dim, output_size) for name, dim in dims.items()})
        self.output_size = output_size
