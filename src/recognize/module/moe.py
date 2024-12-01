from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .router import NoiseRouter, Router


class CopyExpert(nn.Module):
    def forward(self, inputs):
        return inputs


class ZeroExpert(nn.Module):
    def forward(self, inputs):
        return torch.zeros_like(inputs).to(inputs.dtype).to(inputs.device)


class ConstantExpert(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.constant = nn.Parameter(torch.randn(hidden_size))
        self.wg = nn.Sequential(nn.Linear(hidden_size, 2, bias=False), nn.Softmax(dim=-1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        alphas = self.wg(inputs)
        return torch.einsum("b,bd->bd", [alphas[:, 0], inputs]) + torch.einsum("b,d->bd", [alphas[:, 1], self.constant])


class MoE(nn.Module):
    def __init__(
        self,
        feature_size: int,
        experts: Sequence[nn.Module],
        *,
        noisy_router: bool = True,
    ):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.router = (
            Router(feature_size, len(experts)) if not noisy_router else NoiseRouter(feature_size, len(experts))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_outputs = self.router(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.sum(gate_outputs.unsqueeze(2) * expert_outputs, dim=1)
        return output


class SparseMoE(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        experts: Sequence[nn.Module],
        *,
        act_expert_num: int = 1,
        noisy_router: bool = True,
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.act_expert_num = act_expert_num
        self.experts = nn.ModuleList(experts)
        self.router = Router(in_size, len(experts)) if not noisy_router else NoiseRouter(in_size, len(experts))

    def cv_squared(self, x: torch.Tensor) -> torch.Tensor:
        """The squared coefficient of variation of a sample."""
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_experts = len(self.experts)
        batch_size = x.size(0)

        routing_weights = self.router(x)
        routing_weights, selected_experts = torch.topk(routing_weights, self.act_expert_num, dim=1)
        routing_weights /= routing_weights.sum(dim=1, keepdim=True)

        # shape of expert_mask: (batch_size, num_experts, act_expert_num)
        expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(1, 2, 0)

        # TODO: to find a way to support different output size
        output = torch.zeros((batch_size, self.out_size), dtype=x.dtype, device=x.device)
        for expert_idx in range(num_experts):
            expert_layer = self.experts[expert_idx]
            # NOTE: act_expert_idx 是 expert_idx 在 routing_weights 里对应的 idx
            batch_idx, act_expert_idx = torch.nonzero(expert_mask[expert_idx], as_tuple=True)
            if batch_idx.shape[0] == 0:
                continue

            expert_output = expert_layer(x[batch_idx]) * routing_weights[batch_idx, act_expert_idx, None]
            output.index_add_(0, batch_idx, expert_output)
        return output


class MultiHeadMoE(nn.Module):
    def __init__(
        self,
        router_input_size: int,
        experts: Mapping[str, nn.Module],
        *,
        hold_experts: bool = False,
        noisy_router: bool = True,
    ):
        super().__init__()
        self.router = (
            Router(router_input_size, len(experts))
            if not noisy_router
            else NoiseRouter(router_input_size, len(experts))
        )
        self.expert_names = sorted(experts.keys())
        self.hold_experts = hold_experts
        if hold_experts:
            self.experts = nn.ModuleDict(experts)
        else:
            self.experts = experts

    def forward(self, router_input: torch.Tensor, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        routing_weights = self.router(router_input)
        sum_weights = torch.zeros(routing_weights.shape[0], device=routing_weights.device)
        outputs = []
        for i, name in enumerate(self.expert_names):
            if name not in inputs or inputs[name] is None or name not in self.experts:
                continue
            expert_outputs = self.experts[name](inputs[name])
            outputs.append(routing_weights[:, i : i + 1] * expert_outputs)
            sum_weights += routing_weights[:, i]
        return sum(outputs) / sum_weights.unsqueeze(1)
