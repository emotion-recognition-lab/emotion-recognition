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
from .model import ClassifierModel, ModelInput
from .module import FeatureLoss, LogitLoss
from .trainer import EarlyStopper


def save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    model: ClassifierModel,
    optimizer: Optimizer,
    stopper: EarlyStopper,
):
    epoch_checkpoint_dir = checkpoint_dir / str(epoch)
    epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(epoch_checkpoint_dir)

    epoch_encoder_dir = epoch_checkpoint_dir / "backbone"
    original_encoder_dir = checkpoint_dir / "backbone"
    epoch_encoder_dir.mkdir(parents=True, exist_ok=True)
    for name, path in model.backbone.save(epoch_encoder_dir).items():
        (epoch_encoder_dir / f"{name}.safetensors").symlink_to(path)

    original_encoder_dir.unlink(missing_ok=True)
    original_encoder_dir.symlink_to(
        epoch_encoder_dir.relative_to(original_encoder_dir.parent),
        target_is_directory=True,
    )

    (epoch_checkpoint_dir / "preprocessor").symlink_to(
        "../preprocessor",
    )
    (epoch_checkpoint_dir / "inference.toml").symlink_to(
        "../inference.toml",
    )

    # TODO: improve optimizer size
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    stopper.save(checkpoint_dir / "stopper.yaml")


def load_model(checkpoint_dir: Path, model: ClassifierModel):
    checkpoint_dir = Path(checkpoint_dir)
    model_state_dict = load_file(checkpoint_dir / "model.safetensors")
    model.load_state_dict(model_state_dict, strict=False)
    if (checkpoint_dir / "backbone").exists():
        model.backbone = model.backbone.from_checkpoint(checkpoint_dir / "backbone").cuda()


def find_best_model(checkpoint_dir: Path) -> int:
    if (checkpoint_dir / "stopper.yaml").exists():
        stopper = EarlyStopper.from_file(checkpoint_dir / "stopper.yaml")
        best_epoch = stopper.best_epoch
    else:
        logger.warning(f"No stopper or result file found in [blue]{checkpoint_dir}")
        best_epoch = -1
    return best_epoch


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
    use_amp: bool = True,
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
        scaler = amp.GradScaler("cuda")
    else:
        scaler = None
    return Trainer(model, data_loaders, optimizer, scheduler, max_grad_norm=10, scaler=scaler)


def distill_batch(
    student_model: ClassifierModel,
    teacher_model: ClassifierModel,
    batch: ModelInput,
):
    with amp.autocast("cuda"):
        with torch.no_grad():
            teacher_embs = teacher_model.backbone(batch)["T"]
            teacher_logits = teacher_model.classifier(teacher_embs)

        student_embs_dict = student_model.backbone(batch)
        if len(student_embs_dict) < 2:
            # NOTE: only T modality or no input
            return
        student_output = student_model.classify(student_model.fusion_layer(student_embs_dict), batch.labels)
        loss = LogitLoss()(student_output.logits, teacher_logits) + student_output.loss
        # TODO: to remove feature loss
        for name, student_embs in student_embs_dict.items():
            if name == "T":
                continue
            loss += FeatureLoss()(student_embs, teacher_embs)
    return loss


def train_batch(
    model: ClassifierModel,
    batch: ModelInput,
):
    with amp.autocast("cuda"):
        output = model(batch)
        loss = output.loss
    return loss


def train_epoch(
    model: ClassifierModel,
    trainer: Trainer,
    train_data_loader: DataLoader,
    *,
    teacher_model: ClassifierModel | None = None,
    dropout_prob: float | None = None,
    update_hook: Callable[[int, float], None] | None = None,
):
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    trainer.clear_losses()
    for batch_index, batch in enumerate(train_data_loader):
        # NOTE: randomly remove every modality independently
        if isinstance(batch, LazyMultimodalInput) and dropout_prob is not None:
            if random.random() < dropout_prob:
                batch.audio_paths = None
            elif random.random() < dropout_prob:
                batch.video_paths = None
        if teacher_model is not None:
            loss = distill_batch(model, teacher_model, batch)
        else:
            loss = train_batch(model, batch)
        if loss is None:
            continue
        trainer.training_step(loss)
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
    teacher_model: ClassifierModel | None = None,
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
    if stopper is None:
        stopper = EarlyStopper.from_file(checkpoint_dir / "stopper.yaml", create=True)
        if stopper.finished:
            result = trainer.eval("test")
            return result
    if test_data_loader is None:
        test_data_loader = valid_data_loader
    if model_label is None:
        # TODO: improve default model label
        model_label = f"{model.__class__.__name__}-{id(model)}"
    logger.info(f"Train model [blue]{model_label}[/] . Save to [blue]{checkpoint_dir}[/]")

    epoch_start = load_last_checkpoint(
        checkpoint_dir,
        model,
        trainer.optimizer,
    )

    best_epoch = stopper.best_epoch
    batch_num = len(train_data_loader)
    result: TrainingResult | None = None
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
                teacher_model=teacher_model,
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
                        save_checkpoint(checkpoint_dir, epoch, model, trainer.optimizer, stopper)
                        break
                result = trainer.eval("test")
                test_f1_score = result.f1_score
                test_accuracy = result.accuracy
                progress.update(task, f1_score=test_f1_score, accuracy=test_accuracy)
                better_metrics = stopper.update(epoch=epoch, test_accuracy=test_accuracy, test_f1=test_f1_score)
                if stopper.finished:
                    save_checkpoint(checkpoint_dir, epoch, model, trainer.optimizer, stopper)
                    break
                if stopper.best_epoch != best_epoch:
                    best_epoch = stopper.best_epoch
                    save_checkpoint(checkpoint_dir, epoch, model, trainer.optimizer, stopper)
                    better_metrics_str = ", ".join(better_metrics)
                    logger.info(f"Epoch {epoch}: Better model found(improved {better_metrics_str})")
                    logger.info(
                        f"Test - [red]Accuracy: {test_accuracy:.2f}%[/], [red]F1 Score: {test_f1_score:.2f}%[/]"
                    )

            if epoch == num_epochs - 1:
                save_checkpoint(checkpoint_dir, epoch, model, trainer.optimizer, stopper)

            progress.update(task, advance=1)

    load_model(checkpoint_dir / str(best_epoch), model)
    result = trainer.eval("test")
    return result
