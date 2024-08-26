from __future__ import annotations

import os
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
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from recognize.trainer import Trainer

from .evaluate import TrainingResult
from .model.base import Backbone, ClassifierModel
from .module.loss import FeatureLoss, LogitLoss
from .trainer import EarlyStopper


def save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    model: ClassifierModel[Backbone],
    optimizer: Optimizer,
    stopper: EarlyStopper,
):
    epoch_checkpoint_dir = checkpoint_dir / str(epoch)
    epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(epoch_checkpoint_dir)

    epoch_encoder_dir = epoch_checkpoint_dir / "backbones"
    original_encoder_dir = checkpoint_dir / "backbones"
    if model.backbone.is_frozen:
        # save and link original backbone state_dict
        epoch_encoder_dir.symlink_to(
            "../backbones",
            target_is_directory=True,
        )
        if not original_encoder_dir.exists():
            original_encoder_dir.mkdir(parents=True)
            for name, path in model.backbone.save(original_encoder_dir).items():
                (original_encoder_dir / f"{name}.safetensors").symlink_to(path)
    else:
        # save and link backbone state_dict
        epoch_encoder_dir.mkdir(parents=True, exist_ok=True)
        for name, path in model.backbone.save(epoch_encoder_dir).items():
            (epoch_encoder_dir / f"{name}.safetensors").symlink_to(path)

        original_encoder_dir.unlink(missing_ok=True)
        original_encoder_dir.symlink_to(
            epoch_encoder_dir.relative_to(original_encoder_dir.parent),
            target_is_directory=True,
        )
    (epoch_checkpoint_dir / "inference.toml").symlink_to(
        "../inference.toml",
    )

    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")  # TODO: improve size
    stopper.save(checkpoint_dir / "stopper.json")


def load_checkpoint(
    checkpoint_dir: Path | str,
    epoch: int,
    model: ClassifierModel[Backbone],
    optimizer: Optimizer | None = None,
    stopper: EarlyStopper | None = None,
):
    checkpoint_dir = Path(checkpoint_dir)
    if optimizer is not None and os.path.exists(checkpoint_dir / "optimizer.pt"):
        optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt", weights_only=True))
    if stopper is not None and os.path.exists(checkpoint_dir / "stopper.json"):
        stopper.load(checkpoint_dir / "stopper.json")
    if epoch == -1:
        return

    load_model(f"{checkpoint_dir}/{epoch}", model)


def load_model(checkpoint_dir: Path | str, model: ClassifierModel):
    checkpoint_dir = Path(checkpoint_dir)
    model_state_dict = load_file(checkpoint_dir / "model.safetensors")
    model.load_state_dict(model_state_dict, strict=False)
    if (checkpoint_dir / "backbones").exists():
        model.backbone = model.backbone.from_checkpoint(checkpoint_dir / "backbones").cuda()


def find_best_model(checkpoint_dir: Path | str) -> int:
    if os.path.exists(f"{checkpoint_dir}/stopper.json"):
        stopper = EarlyStopper.from_file(f"{checkpoint_dir}/stopper.json")
        best_epoch = stopper.best_epoch
    else:
        logger.warning(f"No stopper or result file found in [blue]{checkpoint_dir}")
        best_epoch = -1
    return best_epoch


def load_best_model(checkpoint_dir: Path | str, model: ClassifierModel) -> int:
    best_epoch = find_best_model(checkpoint_dir)
    logger.info(f"Load best model from [blue]{checkpoint_dir}/{best_epoch}")
    load_model(f"{checkpoint_dir}/{best_epoch}", model)
    return best_epoch


def load_last_checkpoint(
    checkpoint_dir: Path | str,
    model: ClassifierModel,
    optimizer: Optimizer | None = None,
    stopper: EarlyStopper | None = None,
) -> int:
    checkpoint_dir = Path(checkpoint_dir)
    model_list = os.listdir(checkpoint_dir)
    model_list = [int(model_name) for model_name in model_list if model_name.isdigit()]
    epoch_start = max(model_list, default=-1)
    load_checkpoint(checkpoint_dir, epoch_start, model, optimizer, stopper)
    return epoch_start


def get_trainer(
    model: ClassifierModel,
    data_loaders: dict[Literal["train", "valid", "test"], DataLoader],
    num_warmup_steps: int,
    num_training_steps: int,
):
    from transformers import get_linear_schedule_with_warmup

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-6, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return Trainer(model, data_loaders, optimizer, scheduler)


def train_epoch(
    model: ClassifierModel,
    trainer: Trainer,
    train_data_loader: DataLoader,
    update_hook: Callable[[int, float], None] | None = None,
):
    model.train()
    trainer.clear_losses()
    for batch_index, batch in enumerate(train_data_loader):
        output = model(batch)
        loss = output.loss
        loss_value = trainer.training_step(loss)
        if update_hook is not None:
            update_hook(batch_index, loss_value)


def train_and_eval(
    model: ClassifierModel,
    train_data_loader: DataLoader,
    valid_data_loader: DataLoader,
    test_data_loader: DataLoader | None = None,
    *,
    stopper: EarlyStopper | None = None,
    num_epochs: int = 100,
    model_label: str | None = None,
    eval_interval: int = 1,
):
    trainer = get_trainer(
        model,
        {"train": train_data_loader, "valid": valid_data_loader, "test": test_data_loader or valid_data_loader},
        len(train_data_loader),
        len(train_data_loader) * num_epochs,
    )

    if stopper is None:
        stopper = EarlyStopper()
    if test_data_loader is None:
        test_data_loader = valid_data_loader
    if model_label is None:
        model_label = f"{model.__class__.__name__}-{id(model)}"
    # TODO: use checkpoint_dir
    checkpoint_dir = Path(f"./checkpoints/training/{model_label}")
    logger.add(f"./logs/{model_label}.txt")
    logger.info(f"Train model [blue]{model_label}[/]. Save to [blue]{checkpoint_dir}[/]")

    epoch_start = load_last_checkpoint(
        checkpoint_dir,
        model,
        trainer.optimizer,
        stopper,
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
                lambda batch_index, loss_value: progress.update(task, loss=loss_value, batch_index=batch_index + 1),
            )

            if (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1:
                result = trainer.eval("valid")
                valid_f1_score = result.f1_score
                valid_accuracy = result.accuracy
                if stopper.update(epoch=epoch, f1=valid_f1_score):
                    save_checkpoint(checkpoint_dir, epoch, model, trainer.optimizer, stopper)
                    break

                progress.update(task, f1_score=valid_f1_score, accuracy=valid_accuracy)
                if stopper.best_epoch != best_epoch:
                    # TODO: maybe model is not better in test set?
                    best_epoch = stopper.best_epoch
                    save_checkpoint(checkpoint_dir, epoch, model, trainer.optimizer, stopper)
                    logger.info(
                        f"Epoch {best_epoch}: Better model found "
                        f"[red](Accuracy: {valid_accuracy:.2f}%, F1 Score: {valid_f1_score:.2f}%)",
                    )

                if epoch == num_epochs - 1:
                    save_checkpoint(checkpoint_dir, epoch, model, trainer.optimizer, stopper)

            progress.update(task, advance=1)

    load_model(checkpoint_dir / str(best_epoch), model)
    result = trainer.eval("test")
    return result


def distill_epoch(
    teacher_model: ClassifierModel,
    student_model: ClassifierModel,
    trainer: Trainer,
    train_data_loader: DataLoader,
    update_hook: Callable[[int, float], None] | None = None,
):
    teacher_model.eval()
    student_model.train()
    trainer.clear_losses()
    for batch_index, batch in enumerate(train_data_loader):
        with torch.no_grad():
            teacher_embs = teacher_model.backbone(batch)["T"]
            teacher_logits = teacher_model.classifier(teacher_embs)

        student_embs_dict = student_model.backbone(batch)
        if student_embs_dict.get("A") is None:
            continue
        student_output = student_model.classify(student_model.fusion_layer(*student_embs_dict.values()), batch.labels)

        assert student_output.loss is not None
        loss = (
            LogitLoss()(student_output.logits, teacher_logits)
            + FeatureLoss()(student_embs_dict["A"], teacher_embs)
            # + FeatureLoss()(student_pooled_embs_tuple[2], teacher_pooled_embs)
            + 5 * student_output.loss
        )

        loss_value = trainer.step(loss)
        if update_hook is not None:
            update_hook(batch_index, loss_value)


def train_and_eval_distill(
    teacher_model: ClassifierModel,
    model: ClassifierModel,
    train_data_loader: DataLoader,
    valid_data_loader: DataLoader,
    test_data_loader: DataLoader | None = None,
    *,
    stopper: EarlyStopper | None = None,
    num_epochs: int = 100,
    model_label: str | None = None,
    eval_interval: int = 1,
):
    trainer = get_trainer(
        model,
        {"train": train_data_loader, "valid": valid_data_loader, "test": test_data_loader or valid_data_loader},
        len(train_data_loader),
        len(train_data_loader) * num_epochs,
    )
    if stopper is None:
        stopper = EarlyStopper()
    if test_data_loader is None:
        test_data_loader = valid_data_loader
    if model_label is None:
        model_label = f"{model.__class__.__name__}-{id(model)}"
    checkpoint_dir = Path(f"./checkpoints/training/{model_label}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.add(f"./logs/{model_label}.txt")
    logger.info(f"Train model [blue]{model_label}[/]. Save to [blue]{checkpoint_dir}[/]")

    epoch_start = load_last_checkpoint(
        checkpoint_dir,
        model,
        trainer.optimizer,
        stopper,
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
            distill_epoch(
                teacher_model,
                model,
                trainer,
                train_data_loader,
                lambda batch_index, loss_value: progress.update(task, loss=loss_value, batch_index=batch_index + 1),
            )

            if (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1:
                result = trainer.eval("valid")
                test_f1_score = result.f1_score
                test_accuracy = result.accuracy
                if stopper.update(epoch=epoch, f1=test_f1_score):
                    save_checkpoint(checkpoint_dir, epoch, model, trainer.optimizer, stopper)
                    break

                progress.update(task, f1_score=test_f1_score, accuracy=test_accuracy)
                if stopper.best_epoch != best_epoch:
                    best_epoch = stopper.best_epoch
                    save_checkpoint(checkpoint_dir, epoch, model, trainer.optimizer, stopper)
                    logger.info(
                        f"Epoch {best_epoch}: Better model found "
                        f"[red](Accuracy: {test_accuracy:.2f}%, F1 Score: {test_f1_score:.2f}%)",
                    )
                if epoch == num_epochs - 1:
                    save_checkpoint(checkpoint_dir, epoch, model, trainer.optimizer, stopper)

            progress.update(task, advance=1)

    load_model(checkpoint_dir / str(best_epoch), model)
    result = trainer.eval("test")
    return result
