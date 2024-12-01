from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Router(nn.Module):
    def __init__(self, dim: int, num_experts: int):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(dim, num_experts, bias=False),
            nn.Softmax(1),
        )

    def forward(self, x: torch.Tensor):
        routing_weights = self.router(x)
        return routing_weights


class NoiseRouter(nn.Module):
    def __init__(self, dim: int, num_experts: int):
        super().__init__()
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.noise = nn.Sequential(
            nn.Linear(dim, num_experts, bias=False),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor):
        routing_logit = self.router(x)
        if self.training:
            noise_logit = self.noise(x)
            routing_logit += torch.randn_like(routing_logit) * noise_logit
        return F.softmax(routing_logit, dim=1)
