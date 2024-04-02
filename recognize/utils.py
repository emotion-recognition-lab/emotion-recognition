from __future__ import annotations

import os
from pathlib import Path

import torch
from loguru import logger
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from .evaluate import EarlyStopper, TrainingResult
from .model.base import ClassifierModel


def save_checkpoint(
    checkpoint_dir: Path, epoch: int, model: ClassifierModel, optimizer: torch.optim.Optimizer, stopper: EarlyStopper
):
    epoch_checkpoint_dir = checkpoint_dir / str(epoch)
    epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(epoch_checkpoint_dir)

    backbones_dir = epoch_checkpoint_dir / "backbones"
    if not model.backbone.is_frozen:
        # save and link backbone state_dict
        backbones_dir.mkdir(parents=True, exist_ok=True)
        for name, path in model.backbone.save().items():
            (backbones_dir / f"{name}.safetensors").symlink_to(path)
    else:
        # save and link original backbone state_dict
        original_backbones_dir = checkpoint_dir / "backbones"
        if not original_backbones_dir.exists():
            original_backbones_dir.mkdir(parents=True)
            for name, path in model.backbone.save().items():
                (original_backbones_dir / f"{name}.safetensors").symlink_to(path)

        backbones_dir.symlink_to("../backbones")

    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")  # TODO: improve size
    stopper.save(checkpoint_dir / "stopper.json")


def load_model(checkpoint_dir: Path | str, model: ClassifierModel):
    checkpoint_dir = Path(checkpoint_dir)
    model_state_dict = load_file(checkpoint_dir / "model.safetensors")
    model.load_state_dict(model_state_dict, strict=False)
    if (checkpoint_dir / "backbones").exists():
        model.backbone.set_state_dicts(
            {
                name.split(".")[0]: load_file(checkpoint_dir / f"backbones/{name}")
                for name in os.listdir(checkpoint_dir / "backbones")
            }
        )


def find_best_model(checkpoint_dir: Path | str) -> int:
    if os.path.exists(f"{checkpoint_dir}/stopper.json"):
        stopper = EarlyStopper.from_file(f"{checkpoint_dir}/stopper.json")
        best_epoch = stopper.best_epoch
    elif os.path.exists(f"{checkpoint_dir}/result.json"):
        with open(f"{checkpoint_dir}/result.json", "r") as f:
            best_epoch = TrainingResult.model_validate_json(f.read()).best_epoch
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
    optimizer: torch.optim.Optimizer | None = None,
    stopper: EarlyStopper | None = None,
) -> int:
    checkpoint_dir = Path(checkpoint_dir)
    model_list = os.listdir(checkpoint_dir)
    epoch_start = -1
    model_list = [int(model_name) for model_name in model_list if model_name.isdigit()]
    if model_list:
        model_list.sort()
        epoch_start = int(model_list[-1])
        logger.info(f"Load last model from [blue]{checkpoint_dir}/{epoch_start}")
        load_model(f"{checkpoint_dir}/{epoch_start}", model)
    if optimizer is not None and os.path.exists(checkpoint_dir / "optimizer.pt"):
        optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt"))
    if stopper is not None and os.path.exists(checkpoint_dir / "stopper.json"):
        stopper.load(checkpoint_dir / "stopper.json")
    return epoch_start


def train_and_eval(
    model: ClassifierModel,
    train_data_loader: DataLoader,
    valid_data_loader: DataLoader,
    test_data_loader: DataLoader | None = None,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    stopper: EarlyStopper | None = None,
    num_epochs: int = 100,
    model_label: str | None = None,
    eval_interval: int = 1,
):
    if optimizer is None:
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=0.1, amsgrad=True)
    if stopper is None:
        stopper = EarlyStopper()
    if test_data_loader is None:
        test_data_loader = valid_data_loader
    if model_label is None:
        model_label = f"{model.__class__.__name__}-{id(model)}"
    checkpoint_dir = Path(f"./checkpoints/{model_label}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.add(f"./logs/{model_label}.txt")
    logger.info(f"Train model [blue]{model_label}[/]. Save to [blue]{checkpoint_dir}[/]")

    epoch_start = load_last_checkpoint(
        checkpoint_dir,
        model,
        optimizer,
        stopper,
    )

    best_epoch = stopper.best_epoch
    batch_num = len(train_data_loader)
    result: TrainingResult | None = None
    with Progress(
        "[red](Loss: {task.fields[loss]:.8f}, Accuracy: {task.fields[accuracy]:.2f}%, F1 Score: {task.fields[f1_score]:.2f}%)",
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
            model.train()
            loss_value_list = []
            for batch_index, batch in enumerate(train_data_loader):
                optimizer.zero_grad()
                output = model(batch)
                loss = output.loss
                assert loss is not None
                loss.backward()
                optimizer.step()
                loss_value_list.append(loss.item())
                loss_value = sum(loss_value_list) / len(loss_value_list)
                progress.update(task, loss=loss_value, batch_index=batch_index + 1)

            if (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1:
                result = TrainingResult.auto_compute(model, valid_data_loader)
                test_f1_score = result.f1_score
                test_accuracy = result.accuracy
                if stopper.update(epoch=epoch, f1=test_f1_score):
                    save_checkpoint(checkpoint_dir, epoch, model, optimizer, stopper)
                    break

                progress.update(task, f1_score=test_f1_score, accuracy=test_accuracy)
                if stopper.best_epoch != best_epoch:
                    best_epoch = stopper.best_epoch
                    result.best_epoch = best_epoch
                    save_checkpoint(checkpoint_dir, epoch, model, optimizer, stopper)
                    logger.info(
                        f"Epoch {best_epoch}: Better model found [red](Accuracy: {test_accuracy:.2f}%, F1 Score: {test_f1_score:.2f}%)",
                    )
                if epoch == num_epochs - 1:
                    save_checkpoint(checkpoint_dir, epoch, model, optimizer, stopper)

            progress.update(task, advance=1)

    load_model(checkpoint_dir / str(best_epoch), model)
    result = TrainingResult.auto_compute(model, test_data_loader)
    result.best_epoch = best_epoch
    result.save(f"{checkpoint_dir}/result.json")
    return result


def init_logger(log_level: str):
    handler = RichHandler(highlighter=NullHighlighter(), markup=True)
    logger.remove()
    logger.add(handler, format="{message}", level=log_level)
