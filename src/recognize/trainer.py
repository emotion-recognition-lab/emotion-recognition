from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from loguru import logger
from pydantic import BaseModel, Field
from torch import amp, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from recognize.config import load_dict_from_path, save_dict_to_file
from recognize.evaluate import TrainingResult

if TYPE_CHECKING:
    from .model import ClassifierModel


class EarlyStopper(BaseModel):
    patience: int = 20
    history: list[tuple[int, dict[str, float]]] = Field(default_factory=list)
    best_scores: dict[str, float] = Field(default_factory=dict)
    best_epoch: int = -1
    finished: bool = False

    @classmethod
    def from_file(cls, path: Path, create: bool = False):
        if not Path(path).exists():
            if not create:
                raise FileNotFoundError(f"Stopper file [blue]{path}[/] not found, set create=True to create it")
            logger.info(f"Stopper file [blue]{path}[/] not found, use default config")
            return cls()
        state_dict = load_dict_from_path(path)
        return cls.model_validate(state_dict)

    def load(self, path: Path):
        state_dict = load_dict_from_path(path)
        self.patience = state_dict["patience"]
        self.best_scores = state_dict["best_scores"]
        self.best_epoch = state_dict["best_epoch"]

    def save(self, path: Path):
        save_dict_to_file(self.model_dump(mode="json"), path)

    def update(self, epoch: int, **kwargs: float) -> list[str]:
        self.history.append((epoch, kwargs))
        better_metrics = []
        for key, value in kwargs.items():
            if key not in self.best_scores:
                self.best_scores[key] = float("-inf")
            if value > self.best_scores[key]:
                self.best_scores[key] = value
                self.best_epoch = epoch
                better_metrics.append(key)
            else:
                if epoch - self.best_epoch >= self.patience:
                    logger.info(f"Early stopping! Best epoch: {self.best_epoch}")
                    self.finished = True
                    return []
        return better_metrics


class Trainer:
    def __init__(
        self,
        model: ClassifierModel,
        data_loaders: dict[Literal["train", "valid", "test"], DataLoader],
        optimizer: Optimizer,
        scheduler: LambdaLR | None = None,
        scaler: amp.GradScaler | None = None,
        *,
        max_grad_norm: float | None = None,
    ):
        self.model = model
        self.data_loaders = data_loaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
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
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self.losses.append(loss.detach())

    def fit(self) -> TrainingResult:
        logger.warning("Trainer.fit is not implemented!")
        return self.eval("train")

    def eval(self, key: Literal["train", "valid", "test"]) -> TrainingResult:
        return TrainingResult.auto_compute(self.model, self.data_loaders[key])
