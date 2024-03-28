from __future__ import annotations

import os
from pathlib import Path

import torch
from loguru import logger
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from .evaluate import TrainingResult, calculate_accuracy_and_f1_score, confusion_matrix
from .model.base import ClassifierModel


class EarlyStopper:
    def __init__(self, patience: int = 10):
        self.patience = patience
        self.best_scores = {}
        self.best_epoch = -1

    def state_dict(self):
        return {
            "best_scores": self.best_scores,
            "best_epoch": self.best_epoch,
        }

    def load_state_dict(self, state_dict):
        self.best_scores = state_dict["best_scores"]
        self.best_epoch = state_dict["best_epoch"]

    def update(self, epoch: int, **kwargs: float):
        for key, value in kwargs.items():
            if key not in self.best_scores:
                self.best_scores[key] = (float("-inf"), 0)
            if value > self.best_scores[key][0]:
                repeat_times = self.best_scores[key][1]
                self.best_scores[key] = (value, repeat_times)
                self.best_epoch = epoch
            else:
                repeat_times = self.best_scores[key][1]
                self.best_scores[key] = (self.best_scores[key][0], repeat_times + 1)
                if self.best_scores[key][1] >= self.patience:
                    print(f"Early stopping by {key}!")
                    return True
        return False


def save_checkpoint(
    checkpoint_dir: Path, model: ClassifierModel, optimizer: torch.optim.Optimizer, stopper: EarlyStopper
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_state_dict = {key: value for key, value in model.state_dict().items() if not key.startswith("backbone.")}
    save_file(model_state_dict, checkpoint_dir / "model.safetensors")
    for name, state_dict in model.backbone.get_peft_state_dicts().items():
        pefts_dir = checkpoint_dir / "pefts"
        pefts_dir.mkdir(parents=True, exist_ok=True)
        save_file(state_dict, pefts_dir / f"{name}.safetensors")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(stopper.state_dict(), checkpoint_dir / "stopper.pt")


def load_checkpoint(
    checkpoint_dir: Path | str,
    model: ClassifierModel,
    optimizer: torch.optim.Optimizer | None = None,
    stopper: EarlyStopper | None = None,
):
    checkpoint_dir = Path(checkpoint_dir)
    model_state_dict = load_file(checkpoint_dir / "model.safetensors")
    model.load_state_dict(model_state_dict, strict=False)
    if (checkpoint_dir / "pefts").exists():
        model.backbone.set_peft_state_dicts(
            {
                name.split(".")[0]: load_file(checkpoint_dir / f"pefts/{name}")
                for name in os.listdir(checkpoint_dir / "pefts")
            }
        )
    if optimizer is not None and os.path.exists(checkpoint_dir / "optimizer.pt"):
        optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt"))
    if stopper is not None and os.path.exists(checkpoint_dir / "stopper.pt"):
        stopper.load_state_dict(torch.load(checkpoint_dir / "stopper.pt"))


def load_last_checkpoint(
    checkpoint_dir: Path | str,
    model: ClassifierModel,
    optimizer: torch.optim.Optimizer | None = None,
    stopper: EarlyStopper | None = None,
) -> int:
    model_list = os.listdir(checkpoint_dir)
    epoch_start = 0
    if model_list:
        model_list = [int(model_name) for model_name in model_list if model_name.isdigit()]
        model_list.sort()
        epoch_start = int(model_list[-1])
        load_checkpoint(f"{checkpoint_dir}/{model_list[-1]}", model, optimizer, stopper)
    return epoch_start


def load_best_checkpoint(
    checkpoint_dir: Path | str,
    model: ClassifierModel,
    optimizer: torch.optim.Optimizer | None = None,
    stopper: EarlyStopper | None = None,
) -> int:
    with open(f"{checkpoint_dir}/result.json", "r") as f:
        best_epoch = TrainingResult.model_validate_json(f.read()).best_epoch

    load_checkpoint(f"{checkpoint_dir}/{best_epoch}", model, optimizer, stopper)
    return best_epoch


def train_and_eval(
    model: ClassifierModel,
    train_data_loader: DataLoader,
    dev_data_loader: DataLoader,
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
        # optimizer = torch.optim.Adam(parameters, lr=1e-5)
        optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=0.1, amsgrad=True)
    if stopper is None:
        stopper = EarlyStopper(patience=30)
    if test_data_loader is None:
        test_data_loader = dev_data_loader
    if model_label is None:
        model_label = f"{model.__class__.__name__}-{id(model)}"
    checkpoint_dir = Path(f"./checkpoints/{model_label}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.add(f"./logs/{model_label}.txt")
    print(f"Train model `{model_label}`.", flush=False)

    epoch_start = load_last_checkpoint(
        checkpoint_dir,
        model,
        optimizer,
        stopper,
    )

    best_epoch = stopper.best_epoch
    batch_num = len(train_data_loader)
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
                progress.update(task, loss=loss_value, batch_index=batch_index)
                optimizer.zero_grad()
                output = model(batch)
                loss = output.loss
                assert loss is not None
                loss.backward()
                optimizer.step()
                loss_value_list.append(loss.item())
                loss_value = sum(loss_value_list) / len(loss_value_list)

            if (epoch + 1) % eval_interval == 0:
                test_accuracy, test_f1_score = calculate_accuracy_and_f1_score(model, dev_data_loader)
                if stopper.update(epoch=epoch, f1=test_f1_score):
                    break
                progress.update(task, f1_score=test_f1_score, accuracy=test_accuracy)

                if stopper.best_epoch != best_epoch:
                    best_epoch = stopper.best_epoch
                    save_checkpoint(checkpoint_dir / str(epoch), model, optimizer, stopper)
                    print(
                        f"Epoch {epoch}: Better model found (Accuracy: {test_accuracy:.2f}%, F1 Score: {test_f1_score:.2f}%)",
                    )
            progress.update(task, advance=1)

            # if epoch > 100:
            #     model.unfreeze_backbone()

    load_checkpoint(f"{checkpoint_dir}/{best_epoch}", model)
    train_accuracy, train_f1_score = calculate_accuracy_and_f1_score(model, train_data_loader)
    test_accuracy, test_f1_score = calculate_accuracy_and_f1_score(model, test_data_loader)

    result = TrainingResult(
        train_accuracy=train_accuracy,
        train_f1_score=train_f1_score,
        test_accuracy=test_accuracy,
        test_f1_score=test_f1_score,
        best_epoch=best_epoch,
    )
    result.save(f"{checkpoint_dir}/result.json")
    print(confusion_matrix(model, test_data_loader))
    print(confusion_matrix(model, train_data_loader))
    return result
