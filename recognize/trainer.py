from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from loguru import logger
from pydantic import BaseModel, Field
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from recognize.evaluate import TrainingResult, calculate_accuracy_and_f1_score

if TYPE_CHECKING:
    from .model import ClassifierModel


class EarlyStopper(BaseModel):
    patience: int = 20
    best_scores: dict[str, Any] = Field(default_factory=dict)
    best_epoch: int = -1

    @classmethod
    def from_file(cls, path: str):
        with open(path) as f:
            state_dict = f.read()
        return cls.model_validate_json(state_dict)

    def load(self, path: str | Path):
        with open(path) as f:
            state_dict = self.model_validate_json(f.read()).model_dump()
        self.patience = state_dict["patience"]
        self.best_scores = state_dict["best_scores"]
        self.best_epoch = state_dict["best_epoch"]

    def save(self, path: str | Path):
        with open(path, "w") as f:
            f.write(self.model_dump_json())

    def update(self, epoch: int, **kwargs: float):
        for key, value in kwargs.items():
            if key not in self.best_scores:
                self.best_scores[key] = (float("-inf"), 0)
            if value > self.best_scores[key][0]:
                self.best_scores[key] = (value, 0)
                self.best_epoch = epoch
            else:
                repeat_times = self.best_scores[key][1]
                self.best_scores[key] = (self.best_scores[key][0], repeat_times + 1)
                if self.best_scores[key][1] >= self.patience:
                    logger.info(f"Early stopping by {key}!")
                    return True
        return False


class Trainer:
    @staticmethod
    def init_torch():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __init__(
        self,
        model: ClassifierModel,
        data_loaders: dict[Literal["train", "valid", "test"], DataLoader],
        optimizer: Optimizer,
        scheduler: LambdaLR | None = None,
        *,
        max_grad_norm: float | None = None,
    ):
        self.init_torch()
        self.model = model
        self.data_loaders = data_loaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.losses = []
        self.loss_mean_cache: float = -1

    def clear_losses(self):
        self.losses.clear()

    def loss_mean(self) -> float:
        if not self.losses:
            return -1
        self.loss_mean_cache = torch.stack(self.losses).mean().item()
        return self.loss_mean_cache

    def training_step(self, loss: torch.Tensor | None) -> None:
        if loss is None:
            return

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self.losses.append(loss.detach())

    def fit(self) -> TrainingResult:
        logger.warning("Trainer.fit is not implemented!")
        return self.eval("train")

    def eval(self, key: Literal["train", "valid", "test"]) -> TrainingResult:
        data_loader = self.data_loaders[key]
        accuracy, f1_score = calculate_accuracy_and_f1_score(self.model, data_loader)
        return TrainingResult(
            accuracy=accuracy,
            f1_score=f1_score,
            # confusion_matrix=confusion_matrix(model, data_loader),
        )
