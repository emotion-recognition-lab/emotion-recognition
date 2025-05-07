from __future__ import annotations

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


def calculate_overall_accuracy(conf_matrix: list[list[int]]) -> float:
    num_classes = len(conf_matrix)
    total = sum(sum(row) for row in conf_matrix)
    correct = sum(conf_matrix[i][i] for i in range(num_classes))
    return 100 * correct / total


def calculate_weighted_f1_score(conf_matrix: list[list[int]], class_weights: list[float]) -> float:
    num_classes = len(conf_matrix)
    class_f1_scores = []

    for i in range(num_classes):
        true_positives = conf_matrix[i][i]
        false_positives = sum(conf_matrix[j][i] for j in range(num_classes) if j != i)
        false_negatives = sum(conf_matrix[i][j] for j in range(num_classes) if j != i)

        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        class_f1_scores.append(f1)

    weighted_f1_score = sum(score * weight for score, weight in zip(class_f1_scores, class_weights, strict=True))
    return 100 * weighted_f1_score


def calculate_class_accuracies(conf_matrix: list[list[int]]) -> list[float]:
    num_classes = len(conf_matrix)
    class_accuracies = []

    for i in range(num_classes):
        true_positives = conf_matrix[i][i]
        total = sum(conf_matrix[i][j] for j in range(num_classes))
        accuracy = 100 * true_positives / (total + 1e-10)
        class_accuracies.append(accuracy)

    return class_accuracies


def calculate_class_f1_scores(conf_matrix: list[list[int]]) -> list[float]:
    num_classes = len(conf_matrix)
    class_f1_scores = []

    for i in range(num_classes):
        true_positives = conf_matrix[i][i]
        false_positives = sum(conf_matrix[j][i] for j in range(num_classes) if j != i)
        false_negatives = sum(conf_matrix[i][j] for j in range(num_classes) if j != i)

        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)

        f1 = 100 * 2 * (precision * recall) / (precision + recall + 1e-10)
        class_f1_scores.append(f1)

    return class_f1_scores


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
                _, predicted = torch.max(outputs.logits, 1)
                labels = batch.labels
            predicted_list.extend(predicted.cpu().numpy().tolist())
            labels_list.extend(labels.cpu().numpy().tolist())
    return predicted_list, labels_list


class TrainingResult(BaseModel):
    overall_accuracy: float = 0
    weighted_f1_score: float = 0
    class_accuracies: list[float] = []
    class_f1_scores: list[float] = []

    confusion_matrix: list[list[int]] | None = None

    def print(self, *, print_table: bool = False):
        logger.info(f"Overall Accuracy: {self.overall_accuracy:.2f}%, Weighted F1 Score: {self.weighted_f1_score:.2f}%")
        if len(self.class_accuracies) > 0:
            for i, (acc, f1) in enumerate(zip(self.class_accuracies, self.class_f1_scores, strict=True)):
                logger.info(f"Class {i} - Accuracy: {acc:.2f}%, F1 Score: {f1:.2f}%")
        if not print_table or self.confusion_matrix is None:
            return

        from rich import print

        logger.info("Confusion Matrix:")
        table = Table(show_header=False, show_lines=True)
        for i, row in enumerate(self.confusion_matrix):
            str_row = [f"[red]{v}" if i == j else f"{v}" for j, v in enumerate(row)]
            table.add_row(*str_row)
        print(table)

    def gen_typst_code(self, gen_accuracy: bool = True, gen_f1: bool = True) -> str:
        items = []
        for acc, f1 in zip(self.class_accuracies, self.class_f1_scores, strict=True):
            if gen_accuracy:
                items.append(f"[{acc:.2f}]")
            if gen_f1:
                items.append(f"[{f1:.2f}]")
        return ", ".join(items)

    @classmethod
    def auto_compute(cls, model: ClassifierModel, data_loader: DataLoader, *, output_path: str | None = None):
        dataset = data_loader.dataset
        assert isinstance(dataset, MultimodalDataset)
        class_weights = dataset.class_weights

        conf_matrix = confusion_matrix(model, data_loader)
        overall_accuracy = calculate_overall_accuracy(conf_matrix)
        weighted_f1_score = calculate_weighted_f1_score(conf_matrix, class_weights)
        class_accuracies = calculate_class_accuracies(conf_matrix)
        class_f1_scores = calculate_class_f1_scores(conf_matrix)

        from utils.visualization import plot_confusion_matrix

        if output_path is not None:
            plot_confusion_matrix(
                confusion_matrix=conf_matrix,
                class_names=list(dataset.emotion_class_names_mapping.keys()),
                output_path=output_path,
                normalize=True,
            )
        return cls(
            overall_accuracy=overall_accuracy,
            weighted_f1_score=weighted_f1_score,
            class_accuracies=class_accuracies,
            class_f1_scores=class_f1_scores,
            confusion_matrix=conf_matrix,
        )
