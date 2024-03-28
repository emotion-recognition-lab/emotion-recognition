from __future__ import annotations

import numpy as np
import torch
from cachetools import Cache, cached
from pydantic import BaseModel
from torch.utils.data import DataLoader

from .dataset import MELDDataset
from .model import ClassifierModel


@cached(cache=Cache(maxsize=10))
def calculate_class_weights(data_loader: DataLoader, num_classes: int = 30) -> list[float]:
    if isinstance(data_loader.dataset, MELDDataset):
        return data_loader.dataset.class_weights
    class_counts = [0] * num_classes

    with torch.no_grad():
        for batch in data_loader:
            for i in range(num_classes):
                labels = batch.labels
                class_counts[i] += (labels == i).sum().item()

    total_samples = sum(class_counts)
    class_weights: list[float] = []
    for i in range(num_classes):
        class_weights.append(class_counts[i] / total_samples)

    return class_weights


def confusion_matrix(y_true, y_pred, num_classes: int):
    labels = list(range(num_classes))
    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred, strict=True):
        matrix[true_label, pred_label] += 1

    return matrix


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

    weighted_f1_score = 0.0

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
    weighted_f1_score = 0.0
    for i in range(num_classes):
        class_weight = class_weights[i]
        class_f1 = class_f1_scores[i] / class_counts[i]
        weighted_f1_score += class_weight * class_f1
    return accuracy, 100 * weighted_f1_score


class TrainingResult(BaseModel):
    train_accuracy: float
    train_f1_score: float
    test_accuracy: float
    test_f1_score: float

    confusion_matrix: list[list[int]] | None = None
    best_epoch: int = 0

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.model_dump_json())

    def print(self):
        print(f"Train Accuracy: {self.train_accuracy:.2f}%")
        print(f"Train F1 Score: {self.train_f1_score:.2f}%")
        print(f"Test Accuracy: {self.test_accuracy:.2f}%")
        print(f"Test F1 Score: {self.test_f1_score:.2f}%")

    @classmethod
    def auto_compute(cls, model: ClassifierModel, train_data_loader: DataLoader, test_data_loader: DataLoader):
        train_accuracy, train_f1_score = calculate_accuracy_and_f1_score(model, train_data_loader)
        test_accuracy, test_f1_score = calculate_accuracy_and_f1_score(model, test_data_loader)

        return cls(
            train_accuracy=train_accuracy,
            train_f1_score=train_f1_score,
            test_accuracy=test_accuracy,
            test_f1_score=test_f1_score,
        )
