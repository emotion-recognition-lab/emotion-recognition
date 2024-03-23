from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import torch
from cachetools import Cache, cached
from loguru import logger
from rich.progress import Progress
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader

from .model.base import ClassifierModel


def calculate_accuracy(model: ClassifierModel, data_loader: DataLoader):
    model.eval()
    correct: int = 0
    total: int = 0

    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch)
            _, predicted = torch.max(outputs.logits, 1)
            total += batch.labels.size(0)
            correct += (predicted == batch.labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


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


class EarlyStopper:
    def __init__(self, model, patience: int = 10):
        self.patience = patience
        self.best_scores = {}
        self.model = model
        self.best_model = deepcopy(model)

    def update(self, **kwargs: float):
        for key, value in kwargs.items():
            if key not in self.best_scores:
                self.best_scores[key] = (float("-inf"), 0)
            if value > self.best_scores[key][0]:
                repeat_times = self.best_scores[key][1]
                self.best_scores[key] = (value, repeat_times)
                self.best_model = deepcopy(self.model)
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


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_dir: Path) -> int:
    model_list = os.listdir(checkpoint_dir)
    epoch_start = 0
    if model_list:
        model_list.sort(key=lambda x: int(x))
        epoch_start = int(model_list[-1])
        model.load_state_dict(load_file(checkpoint_dir / model_list[-1] / "model.safetensors"))
        optimizer.load_state_dict(torch.load(checkpoint_dir / model_list[-1] / "optimizer.pt"))
    return epoch_start


def train_and_eval(
    model: ClassifierModel,
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    num_epochs: int = 100,
    checkpoint_label: str | None = None,
):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.1, amsgrad=True)

    if checkpoint_label is None:
        checkpoint_label = f"{model.__class__.__name__}-{id(model)}"

    checkpoint_dir = Path(f"./checkpoints/{checkpoint_label}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    epoch_start = load_checkpoint(model, optimizer, checkpoint_dir)

    model.train()
    stopper = EarlyStopper(model, patience=20)
    best_model = stopper.best_model
    with Progress(
        "[red](Loss: {task.fields[loss]:.8f}, Accuracy: {task.fields[accuracy]:.2f}%, F1 Score: {task.fields[f1_score]:.2f}%)",
        *Progress.get_default_columns(),
    ) as progress:
        task = progress.add_task(
            "[green]Training model",
            total=num_epochs * len(train_data_loader),
            loss=float("inf"),
            accuracy=0,
            f1_score=0,
            completed=epoch_start,
        )
        for epoch in range(epoch_start, num_epochs):
            loss_value = float("inf")
            loss_value_list = []
            for batch in train_data_loader:
                optimizer.zero_grad()
                output = model(batch)
                loss = output.loss
                assert loss is not None
                loss.backward()
                optimizer.step()
                loss_value_list.append(loss.item())
                loss_value = sum(loss_value_list) / len(loss_value_list)
                progress.update(task, loss=loss_value, advance=1)

            if (epoch + 1) % 5 == 0:
                save_checkpoint(checkpoint_dir / str(epoch), model, optimizer, stopper)

                test_accuracy = calculate_accuracy(model, test_data_loader)
                test_f1_score = calculate_f1_score(model, test_data_loader)
                if stopper.update(f1=test_f1_score):
                    break
                progress.update(task, f1_score=test_f1_score, accuracy=test_accuracy)

                if stopper.best_model != best_model:
                    best_model = stopper.best_model
                    print(f"Best model found at epoch {epoch} (Accuracy: {test_accuracy}, F1 Score: {test_f1_score})!")

            # if epoch > 100:
            #     model.unfreeze_backbone()

    train_accuracy = calculate_accuracy(best_model, train_data_loader)
    test_accuracy = calculate_accuracy(best_model, test_data_loader)

    train_f1_score = calculate_f1_score(best_model, train_data_loader)
    test_f1_score = calculate_f1_score(best_model, test_data_loader)
    return best_model, train_accuracy, test_accuracy, train_f1_score, test_f1_score
