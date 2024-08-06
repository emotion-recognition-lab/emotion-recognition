from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class Trainer:
    @staticmethod
    def init_torch():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LambdaLR | None = None,
        *,
        max_grad_norm: float | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.losses = []

    def clear_losses(self):
        self.losses.clear()

    def step(self, loss: torch.Tensor | None) -> float:
        if loss is None:
            return -1
        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return sum(self.losses) / len(self.losses)
