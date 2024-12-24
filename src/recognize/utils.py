from __future__ import annotations

import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
from loguru import logger
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from safetensors.torch import load_file
from torch import amp
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from recognize.model import LazyMultimodalInput
from recognize.trainer import Trainer

from .evaluate import TrainingResult
from .model import ClassifierModel
from .trainer import EarlyStopper


def save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    model: ClassifierModel,
    optimizer: Optimizer,
):
    epoch_checkpoint_dir = checkpoint_dir / str(epoch)
    epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(epoch_checkpoint_dir)

    epoch_encoder_dir = epoch_checkpoint_dir / "backbone"
    epoch_encoder_dir.mkdir(parents=True, exist_ok=True)
    for name, path in model.backbone.save(epoch_encoder_dir).items():
        if (epoch_encoder_dir / f"{name}.safetensors").exists():
            continue
        (epoch_encoder_dir / f"{name}.safetensors").symlink_to(path)

    (epoch_checkpoint_dir / "preprocessor").symlink_to(
        "../preprocessor",
    )
    (epoch_checkpoint_dir / "inference.toml").symlink_to(
        "../inference.toml",
    )

    # TODO: improve optimizer size by using safetensors
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")


def load_model(checkpoint_dir: Path, model: ClassifierModel):
    checkpoint_dir = Path(checkpoint_dir)
    model_state_dict = load_file(checkpoint_dir / "model.safetensors")
    model.load_state_dict(model_state_dict, strict=False)
    if (checkpoint_dir / "backbone").exists():
        model.backbone.load_checkpoint(checkpoint_dir / "backbone")


def find_best_model(checkpoint_dir: Path, key: str = "test_f1") -> int:
    if (checkpoint_dir / "stopper.yaml").exists():
        stopper = EarlyStopper.from_file(checkpoint_dir / "stopper.yaml")
        best_epoch = stopper.best_epoch
    else:
        logger.warning(f"No stopper or result file found in [blue]{checkpoint_dir}")
        best_epoch = {}
    return best_epoch.get(key, -1)


def load_best_model(checkpoint_dir: Path, model: ClassifierModel) -> int:
    best_epoch = find_best_model(checkpoint_dir)
    logger.info(f"Load best model from [blue]{checkpoint_dir}/{best_epoch}")
    load_model(checkpoint_dir / str(best_epoch), model)
    return best_epoch


def load_last_checkpoint(
    checkpoint_dir: Path | str,
    model: ClassifierModel,
    optimizer: Optimizer | None = None,
) -> int:
    checkpoint_dir = Path(checkpoint_dir)
    model_list = os.listdir(checkpoint_dir)
    model_list = [int(model_name) for model_name in model_list if model_name.isdigit()]
    epoch_start = max(model_list, default=-1)
    checkpoint_dir = Path(checkpoint_dir)
    if optimizer is not None and os.path.exists(checkpoint_dir / "optimizer.pt"):
        optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt", weights_only=True))
    if epoch_start != -1:
        load_model(checkpoint_dir / str(epoch_start), model)

    return epoch_start


def get_trainer(
    model: ClassifierModel,
    data_loaders: dict[Literal["train", "valid", "test"], DataLoader],
    num_warmup_steps: int,
    num_training_steps: int,
    *,
    use_amp: bool = False,
    use_8bit_optimizer: bool = True,
):
    import bitsandbytes as bnb
    from transformers import get_linear_schedule_with_warmup

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if use_8bit_optimizer:
        optimizer = bnb.optim.AdamW8bit(parameters, lr=1e-6, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(parameters, lr=1e-6, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    if use_amp:
        # NOTE: GradScaler will introduce more randomness in the training process
        scaler = amp.GradScaler("cuda")
    else:
        scaler = None
    return Trainer(model, data_loaders, optimizer, scheduler, max_grad_norm=10, scaler=scaler)


def train_epoch(
    model: ClassifierModel,
    trainer: Trainer,
    train_data_loader: DataLoader,
    *,
    dropout_prob: float | None = None,
    update_hook: Callable[[int, float], None] | None = None,
):
    model.train()
    trainer.clear_losses()
    for batch_index, batch in enumerate(train_data_loader):
        # NOTE: randomly remove every modality independently
        if isinstance(batch, LazyMultimodalInput) and dropout_prob is not None:
            if random.random() < dropout_prob:
                batch.audio_paths = None
            elif random.random() < dropout_prob:
                batch.video_paths = None
        trainer.train_batch(batch)
        if update_hook is not None:
            if batch_index % (len(train_data_loader) // 10):
                loss_value = trainer.loss_mean()
                trainer.clear_losses()
            else:
                loss_value = trainer.loss_mean_cache
            update_hook(batch_index, loss_value)


def train_and_eval(
    model: ClassifierModel,
    train_data_loader: DataLoader,
    valid_data_loader: DataLoader,
    test_data_loader: DataLoader | None = None,
    *,
    checkpoint_dir: Path,
    stopper: EarlyStopper | None = None,
    num_epochs: int = 100,
    model_label: str | None = None,
    use_valid: bool = True,
    eval_interval: int = 1,
    dropout_prob: float | None = None,
):
    trainer = get_trainer(
        model,
        {"train": train_data_loader, "valid": valid_data_loader, "test": test_data_loader or valid_data_loader},
        len(train_data_loader),
        len(train_data_loader) * num_epochs,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if test_data_loader is None:
        test_data_loader = valid_data_loader
    if model_label is None:
        # TODO: improve default model label
        model_label = f"{model.__class__.__name__}-{id(model)}"
    if stopper is None:
        stopper = EarlyStopper.from_file(checkpoint_dir / "stopper.yaml", create=True)
        if stopper.finished:
            result = trainer.eval("test")
            return result

    epoch_start = load_last_checkpoint(
        checkpoint_dir,
        model,
        trainer.optimizer,
    )
    stopper.history = [(epoch, history) for epoch, history in stopper.history if epoch <= epoch_start]

    last_better_epoch = stopper.last_better_epoch
    batch_num = len(train_data_loader)
    result: TrainingResult | None = None

    logger.info(f"Train model [blue]{model_label}[/]. Save to [blue]{checkpoint_dir}[/]")
    with Progress(
        "[red](Loss: {task.fields[loss]:.6f}, "
        "Accuracy: {task.fields[accuracy]:.2f}%, F1 Score: {task.fields[f1_score]:.2f}%)",
        TextColumn("[progress.description]{task.description}"),
        TaskProgressColumn("[progress.percentage](Epoch {task.completed}/{task.total})"),
        BarColumn(bar_width=40),
        TaskProgressColumn(f"[progress.percentage]({{task.fields[batch_index]}}/{batch_num})"),
        TimeRemainingColumn(),
    ) as progress:
        loss_value = float("inf")
        task = progress.add_task(
            "[green]Training model",
            total=num_epochs,
            loss=loss_value,
            accuracy=0,
            f1_score=0,
            completed=epoch_start + 1,
            batch_index=0,
        )
        for epoch in range(epoch_start + 1, num_epochs):
            train_epoch(
                model,
                trainer,
                train_data_loader,
                update_hook=lambda batch_index, loss_value: progress.update(
                    task, loss=loss_value, batch_index=batch_index + 1
                ),
                dropout_prob=dropout_prob,
            )

            if (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1:
                if use_valid:
                    result = trainer.eval("valid")
                    valid_f1_score = result.f1_score
                    valid_accuracy = result.accuracy
                    better_metrics = stopper.update(epoch=epoch, valid_accuracy=valid_accuracy, valid_f1=valid_f1_score)
                    if stopper.finished:
                        save_checkpoint(checkpoint_dir, epoch, model, trainer.optimizer)
                        break
                result = trainer.eval("test")
                test_f1_score = result.f1_score
                test_accuracy = result.accuracy
                progress.update(task, f1_score=test_f1_score, accuracy=test_accuracy)
                better_metrics = stopper.update(epoch=epoch, test_accuracy=test_accuracy, test_f1=test_f1_score)
                if stopper.finished:
                    save_checkpoint(checkpoint_dir, epoch, model, trainer.optimizer)
                    break
                if stopper.last_better_epoch != last_better_epoch:
                    last_better_epoch = stopper.last_better_epoch
                    save_checkpoint(checkpoint_dir, epoch, model, trainer.optimizer)
                    better_metrics_str = ", ".join(better_metrics)
                    logger.info(f"Epoch {epoch}: Better model found(improved {better_metrics_str})")
                    logger.info(
                        f"Test - [red]Accuracy: {test_accuracy:.2f}%[/], [red]F1 Score: {test_f1_score:.2f}%[/]"
                    )

            if epoch == num_epochs - 1:
                save_checkpoint(checkpoint_dir, epoch, model, trainer.optimizer)

            stopper.save(checkpoint_dir / "stopper.yaml")
            progress.update(task, advance=1)
        stopper.save(checkpoint_dir / "stopper.yaml")

    assert stopper.last_better_epoch is not None
    load_model(checkpoint_dir / str(stopper.last_better_epoch), model)
    result = trainer.eval("test")
    return result
