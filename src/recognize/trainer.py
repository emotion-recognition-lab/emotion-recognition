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

if TYPE_CHECKING:
    from .evaluate import TrainingResult
    from .model import ClassifierModel, ModelInput


class EarlyStopper(BaseModel):
    patience: int = 20
    history: list[tuple[int, dict[str, float]]] = Field(default_factory=list)
    conf_matrix_history: dict[int, list[list[int]] | None] = Field(default_factory=dict)
    best_scores: dict[str, float] = Field(default_factory=dict)
    best_epoch: dict[str, int] = Field(default_factory=dict)
    finished: bool = False

    @property
    def last_better_epoch(self) -> int | None:
        if not self.best_epoch:
            return None
        return max(self.best_epoch.values())

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

    def update(self, epoch: int, conf_matrix: list[list[int]] | None = None, **kwargs: float) -> list[str]:
        self.history.append((epoch, kwargs))
        self.conf_matrix_history[epoch] = conf_matrix

        better_metrics = []
        for key, value in kwargs.items():
            if key not in self.best_scores:
                self.best_scores[key] = float("-inf")
            if value > self.best_scores[key]:
                self.best_scores[key] = value
                self.best_epoch[key] = epoch
                better_metrics.append(key)
            elif self.last_better_epoch is not None:
                if epoch - self.last_better_epoch >= self.patience:
                    logger.info(f"Early stopping! Last better epoch: {self.last_better_epoch}")
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

    def train_batch(
        self,
        batch: ModelInput,
    ) -> None:
        if self.scaler is not None:
            with amp.autocast("cuda"):
                output = self.model(batch)
            loss = output.loss
            if loss is None:
                return
            self.scaler.scale(loss).backward()
        else:
            output = self.model(batch)
            loss = output.loss
            if loss is None:
                return
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
        self.optimizer.zero_grad()

    def _save_checkpoint(self, checkpoint_dir: Path, epoch: int):
        checkpoint_dir = Path(checkpoint_dir)
        epoch_dir = checkpoint_dir / str(epoch)
        epoch_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.model.save_checkpoint(epoch_dir)
            if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "save"):
                backbone_dir = epoch_dir / "backbone"
                backbone_dir.mkdir(parents=True, exist_ok=True)
                for name, path in self.model.backbone.save(backbone_dir).items():  # type: ignore[attr-defined]
                    target = backbone_dir / f"{name}.safetensors"
                    if not target.exists():
                        target.symlink_to(path)
            torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed to save checkpoint to {checkpoint_dir}: {e}")

    def fit(
        self,
        *,
        num_epochs: int,
        eval_interval: int = 1,
        checkpoint_dir: Path | None = None,
        stopper: EarlyStopper | None = None,
        use_valid: bool = True,
    ) -> TrainingResult:
        if "train" not in self.data_loaders:
            raise ValueError("train dataloader is required")

        stopper = stopper or EarlyStopper()
        best_result: TrainingResult | None = None
        best_epoch: int | None = None

        train_loader = self.data_loaders["train"]
        for epoch in range(num_epochs):
            self.model.train()
            self.clear_losses()
            for batch in train_loader:
                self.train_batch(batch)

            do_eval = (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1
            if not do_eval:
                continue

            eval_split = "valid" if use_valid and "valid" in self.data_loaders else "test"
            result = self.eval(eval_split)
            metrics = {
                f"{eval_split}_accuracy": result.overall_accuracy,
                f"{eval_split}_f1": result.weighted_f1_score,
            }
            better = stopper.update(epoch=epoch, **metrics)
            if better:
                best_result = result
                best_epoch = epoch
                if checkpoint_dir is not None:
                    self._save_checkpoint(checkpoint_dir, epoch)
            if stopper.finished:
                break

        if best_result is None:
            best_result = self.eval("train")
        if checkpoint_dir is not None and best_epoch is not None:
            stopper.save(Path(checkpoint_dir) / "stopper.yaml")
        return best_result

    def eval(self, key: Literal["train", "valid", "test"], *, output_path: str | None = None) -> TrainingResult:
        from .evaluate import TrainingResult

        return TrainingResult.auto_compute(
            self.model,
            self.data_loaders[key],
            output_path=output_path,
        )
