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
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from .evaluate import EarlyStopper, TrainingResult
from .model.base import ClassifierModel


def save_checkpoint(
    checkpoint_dir: Path, epoch: int, model: ClassifierModel, optimizer: torch.optim.Optimizer, stopper: EarlyStopper
):
    epoch_checkpoint_dir = checkpoint_dir / str(epoch)
    epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_state_dict = {key: value for key, value in model.state_dict().items() if not key.startswith("backbone.")}
    save_file(model_state_dict, epoch_checkpoint_dir / "model.safetensors")
    if not model.backbone.is_frozen:
        # save backbone state_dict
        backbones_dir = epoch_checkpoint_dir / "backbones"
        backbones_dir.mkdir(parents=True, exist_ok=True)
        for name, state_dict in model.backbone.get_state_dicts().items():
            save_file(state_dict, backbones_dir / f"{name}.safetensors")
    else:
        # link original backbone state_dict
        original_backbones_dir = checkpoint_dir / "backbones"
        if not original_backbones_dir.exists():
            original_backbones_dir.mkdir(parents=True)
            # TODO: maybe use symlink
            for name, state_dict in model.backbone.get_state_dicts().items():
                save_file(state_dict, original_backbones_dir / f"{name}.safetensors")

        backbones_dir = epoch_checkpoint_dir / "backbones"
        backbones_dir.symlink_to("../backbones")

    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")  # TODO: improve size
    torch.save(stopper.state_dict(), checkpoint_dir / "stopper.pt")


def load_model(
    checkpoint_dir: Path | str,
    model: ClassifierModel,
    # optimizer: torch.optim.Optimizer | None = None,
    # stopper: EarlyStopper | None = None,
):
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


def load_best_model(checkpoint_dir: Path | str, model: ClassifierModel) -> int:
    with open(f"{checkpoint_dir}/result.json", "r") as f:
        best_epoch = TrainingResult.model_validate_json(f.read()).best_epoch

    logger.info(f"Load best model: {best_epoch}")
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
    epoch_start = 0
    if model_list:
        model_list = [int(model_name) for model_name in model_list if model_name.isdigit()]
        model_list.sort()
        epoch_start = int(model_list[-1])
        print(f"Load last model: {epoch_start}")
        load_model(f"{checkpoint_dir}/{model_list[-1]}", model)
    if optimizer is not None and os.path.exists(checkpoint_dir / "optimizer.pt"):
        optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt"))
    if stopper is not None and os.path.exists(checkpoint_dir / "stopper.pt"):
        stopper.load_state_dict(torch.load(checkpoint_dir / "stopper.pt"))
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
            completed=epoch_start,
            batch_index=0,
        )
        for epoch in range(epoch_start, num_epochs):
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
                    break

                progress.update(task, f1_score=test_f1_score, accuracy=test_accuracy)
                if stopper.best_epoch != best_epoch:
                    best_epoch = stopper.best_epoch
                    result.save(f"{checkpoint_dir}/result.json")
                    save_checkpoint(checkpoint_dir, epoch, model, optimizer, stopper)
                    logger.info(
                        f"Epoch {best_epoch}: Better model found [red](Accuracy: {test_accuracy:.2f}%, F1 Score: {test_f1_score:.2f}%)",
                    )
                if epoch == num_epochs - 1:
                    save_checkpoint(checkpoint_dir, epoch, model, optimizer, stopper)

            progress.update(task, advance=1)

    load_best_model(checkpoint_dir, model)
    result = TrainingResult.auto_compute(model, test_data_loader)
    return result


def init_logger(log_level: str):
    handler = RichHandler(highlighter=NullHighlighter(), markup=True)
    logger.remove()
    logger.add(handler, format="{message}", level=log_level)
