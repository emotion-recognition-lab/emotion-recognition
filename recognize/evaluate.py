from __future__ import annotations

import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel
from rich.table import Table
from torch.utils.data import DataLoader

from .dataset import MultimodalDataset
from .model import ClassifierModel


def confusion_matrix(model: ClassifierModel, data_loader: DataLoader) -> list[list[int]]:
    y_true, y_pred = get_outputs(model, data_loader)

    num_classes = model.num_classes
    if num_classes == 1:
        num_classes = 2
    matrix = [[0] * num_classes for _ in range(num_classes)]

    for true_label, pred_label in zip(y_true, y_pred, strict=True):
        matrix[true_label][pred_label] += 1

    return matrix


def calculate_accuracy(predicted_list: list[int], labels_list: list[int]) -> float:
    correct: int = 0
    total: int = len(predicted_list)
    for predicted, labels in zip(predicted_list, labels_list, strict=True):
        if predicted == labels:
            correct += 1

    accuracy = 100 * correct / total
    return accuracy


def calculate_f1_score(predicted_list: list[int], labels_list: list[int], class_weights: list[float]) -> float:
    num_classes = len(class_weights)
    predicted = np.array(predicted_list)
    labels = np.array(labels_list)

    class_f1_scores = [0] * num_classes
    class_counts = [0] * num_classes

    for i in range(num_classes):
        true_positives = ((predicted == i) & (labels == i)).sum()
        false_positives = ((predicted == i) & (labels != i)).sum()
        false_negatives = ((predicted != i) & (labels == i)).sum()

        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)

        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

        class_f1_scores[i] += f1_score
        class_counts[i] += 1

    weighted_f1_score = 0.0

    for i in range(num_classes):
        class_weight = class_weights[i]
        class_f1 = class_f1_scores[i] / class_counts[i]
        weighted_f1_score += class_weight * class_f1
    return 100 * weighted_f1_score


def calculate_accuracy_and_f1_score(model: ClassifierModel, data_loader: DataLoader) -> tuple[float, float]:
    dataset = data_loader.dataset
    assert isinstance(dataset, MultimodalDataset)
    class_weights = dataset.class_weights
    predicted_list, labels_list = get_outputs(model, data_loader)

    # accuracy
    accuracy = calculate_accuracy(predicted_list, labels_list)

    # f1 score
    weighted_f1_score = calculate_f1_score(predicted_list, labels_list, class_weights)

    return accuracy, weighted_f1_score


def get_outputs(model: ClassifierModel, data_loader: DataLoader) -> tuple[list[int], list[int]]:
    model.eval()
    predicted_list: list[int] = []
    labels_list: list[int] = []
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch)
            if outputs.logits.shape[1] == 1:
                predicted = (outputs.logits > 0).int().view(-1)
                labels = (batch.labels > 0).int()
            else:
                _, predicted = torch.max(outputs.logits.detach(), 1)
                labels = batch.labels
            predicted_list.extend(predicted.detach().cpu().numpy().tolist())
            labels_list.extend(labels.cpu().numpy().tolist())
    return predicted_list, labels_list


class ModelResult(BaseModel):
    accuracy: float = 0
    f1_score: float = 0

    confusion_matrix: list[list[int]] | None = None
    best_epoch: int = 0

    def print(self, *, print_table: bool = False):
        logger.info(f"Accuracy: {self.accuracy:.2f}%, F1 Score: {self.f1_score:.2f}%")
        if not print_table or self.confusion_matrix is None:
            return

        from rich import print

        logger.info("Confusion Matrix:")
        table = Table(show_header=False, show_lines=True)
        for i, row in enumerate(self.confusion_matrix):
            str_row = [f"[red]{v}" if i == j else f"{v}" for j, v in enumerate(row)]
            table.add_row(*str_row)
        print(table)

    @classmethod
    def auto_compute(cls, model: ClassifierModel, data_loader: DataLoader):
        accuracy, f1_score = calculate_accuracy_and_f1_score(model, data_loader)

        return cls(
            accuracy=accuracy,
            f1_score=f1_score,
            confusion_matrix=confusion_matrix(model, data_loader),
        )


class TrainingResult(BaseModel):
    accuracy: float = 0
    f1_score: float = 0

    confusion_matrix: list[list[int]] | None = None

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.model_dump_json())

    def print(self, *, print_table: bool = False):
        logger.info(f"Accuracy: {self.accuracy:.2f}%, F1 Score: {self.f1_score:.2f}%")
        if not print_table or self.confusion_matrix is None:
            return

        from rich import print

        logger.info("Confusion Matrix:")
        table = Table(show_header=False, show_lines=True)
        for i, row in enumerate(self.confusion_matrix):
            str_row = [f"[red]{v}" if i == j else f"{v}" for j, v in enumerate(row)]
            table.add_row(*str_row)
        print(table)

    @classmethod
    def auto_compute(cls, model: ClassifierModel, data_loader: DataLoader):
        accuracy, f1_score = calculate_accuracy_and_f1_score(model, data_loader)
        return cls(
            accuracy=accuracy,
            f1_score=f1_score,
            # confusion_matrix=confusion_matrix(model, data_loader),
        )
