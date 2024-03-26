from __future__ import annotations

import os
from pathlib import Path

import torch
from cachetools import Cache, cached
from loguru import logger
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader

from .model.base import ClassifierModel


@cached(cache=Cache(maxsize=10))
def calculate_class_weights(data_loader: DataLoader, num_classes: int = 30):
    class_counts = [0] * num_classes

    with torch.no_grad():
        for batch in data_loader:
            for i in range(num_classes):
                labels = batch.labels
                class_counts[i] += (labels == i).sum().item()

    total_samples = sum(class_counts)
    class_weights = []
    for i in range(num_classes):
        class_weights.append(class_counts[i] / total_samples)

    return class_weights


def calculate_accuracy(model: ClassifierModel, data_loader: DataLoader):
    correct: int = 0
    total: int = 0

    with torch.no_grad():
        model.eval()
        for batch in data_loader:
            outputs = model(batch)
            _, predicted = torch.max(outputs.logits.detach(), 1)
            total += batch.labels.size(0)
            correct += (predicted == batch.labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def calculate_f1_score(model: ClassifierModel, data_loader: DataLoader):
    num_classes = model.num_classes
    class_weights = calculate_class_weights(data_loader, num_classes)
    class_f1_scores = [0] * num_classes
    class_counts = [0] * num_classes

    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch)
            _, predicted = torch.max(outputs.logits, 1)
            labels = batch.labels

            for i in range(num_classes):
                true_positives = ((predicted == i) & (labels == i)).sum().item()
                false_positives = ((predicted == i) & (labels != i)).sum().item()
                false_negatives = ((predicted != i) & (labels == i)).sum().item()

                precision = true_positives / (true_positives + false_positives + 1e-10)
                recall = true_positives / (true_positives + false_negatives + 1e-10)

                f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

                class_f1_scores[i] += f1_score
                class_counts[i] += 1

    weighted_f1_score = 0

    for i in range(num_classes):
        class_weight = class_weights[i]
        class_f1 = class_f1_scores[i] / class_counts[i]
        weighted_f1_score += class_weight * class_f1
    return 100 * weighted_f1_score


def calculate_accuracy_and_f1_score(model: ClassifierModel, data_loader: DataLoader):
    model.eval()

    # accuracy
    correct: int = 0
    total: int = 0

    # f1 score
    num_classes = model.num_classes
    class_weights = calculate_class_weights(data_loader, num_classes)
    class_f1_scores = [0] * num_classes
    class_counts = [0] * num_classes

    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch)
            _, predicted = torch.max(outputs.logits, 1)
            labels = batch.labels

            # accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # f1 score
            for i in range(num_classes):
                true_positives = ((predicted == i) & (labels == i)).sum().item()
                false_positives = ((predicted == i) & (labels != i)).sum().item()
                false_negatives = ((predicted != i) & (labels == i)).sum().item()

                precision = true_positives / (true_positives + false_positives + 1e-10)
                recall = true_positives / (true_positives + false_negatives + 1e-10)

                f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

                class_f1_scores[i] += f1_score
                class_counts[i] += 1

    # accuracy
    accuracy = 100 * correct / total

    # f1 score
    weighted_f1_score = 0
    for i in range(num_classes):
        class_weight = class_weights[i]
        class_f1 = class_f1_scores[i] / class_counts[i]
        weighted_f1_score += class_weight * class_f1
    return accuracy, 100 * weighted_f1_score


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
                    logger.info(f"Early stopping by {key}!")
                    return True
        return False


def save_checkpoint(checkpoint_dir: Path, model: nn.Module, optimizer: torch.optim.Optimizer, stopper: EarlyStopper):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), checkpoint_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(stopper.state_dict(), checkpoint_dir / "stopper.pt")


def load_checkpoint(
    checkpoint_dir: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    stopper: EarlyStopper | None = None,
) -> int:
    model_list = os.listdir(checkpoint_dir)
    epoch_start = 0
    if model_list:
        model_list.sort(key=lambda x: int(x))
        epoch_start = int(model_list[-1])
        model.load_state_dict(load_file(f"{checkpoint_dir}/{model_list[-1]}/model.safetensors"))
        if optimizer is not None:
            optimizer.load_state_dict(torch.load(f"{checkpoint_dir}/{model_list[-1]}/optimizer.pt"))
        if stopper is not None:
            stopper.load_state_dict(torch.load(f"{checkpoint_dir}/{model_list[-1]}/stopper.pt"))
    return epoch_start


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
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.1, amsgrad=True)
    if stopper is None:
        stopper = EarlyStopper(patience=10)
    if test_data_loader is None:
        test_data_loader = dev_data_loader
    if model_label is None:
        model_label = f"{model.__class__.__name__}-{id(model)}"
    checkpoint_dir = Path(f"./checkpoints/{model_label}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.add(f"./logs/{model_label}.txt")
    logger.info(f"Train model `{model_label}`.")

    epoch_start = load_checkpoint(
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
                        f"Epoch {epoch}: Better model found (Accuracy: {test_accuracy:.2f}%, F1 Score: {test_f1_score:.2f}%)"
                    )
            progress.update(task, advance=1)

            # if epoch > 100:
            #     model.unfreeze_backbone()

    model.load_state_dict(load_file(checkpoint_dir / f"{best_epoch}/model.safetensors"))

    train_accuracy, train_f1_score = calculate_accuracy_and_f1_score(model, train_data_loader)
    test_accuracy, test_f1_score = calculate_accuracy_and_f1_score(model, test_data_loader)

    return train_accuracy, test_accuracy, train_f1_score, test_f1_score
