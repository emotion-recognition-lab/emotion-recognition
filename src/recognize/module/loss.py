from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def pearson_correlation(a: torch.Tensor, b: torch.Tensor, eps=1e-8) -> torch.Tensor:
    return F.cosine_similarity(a - a.mean(1).unsqueeze(1), b - b.mean(1).unsqueeze(1), eps=eps)


def inter_class_relation(y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class LogitLoss(nn.Module):
    def __init__(self, beta: float = 1.0, gamma: float = 1.0, tau: float = 4.0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor):
        y_s = (z_s / self.tau).softmax(dim=1)
        y_t = (z_t / self.tau).softmax(dim=1)
        inter_loss = self.tau**2 * inter_class_relation(y_s, y_t)
        intra_loss = self.tau**2 * intra_class_relation(y_s, y_t)
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        return kd_loss


class FeatureLoss(nn.Module):
    def __init__(self, temp: float = 1.0):
        super().__init__()
        self.temp = temp

    def forward(self, student_embeddings: torch.Tensor, teacher_embeddings: torch.Tensor) -> torch.Tensor:
        teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=1)
        student_embeddings = F.normalize(student_embeddings, p=2, dim=1)
        log_q = torch.log_softmax(
            torch.matmul(teacher_embeddings, student_embeddings.transpose(0, 1)) / self.temp, dim=1
        )
        p = torch.softmax(torch.matmul(teacher_embeddings, teacher_embeddings.transpose(0, 1)) / self.temp, dim=1)
        return F.kl_div(log_q, p, reduction="batchmean")


class InfoNCELoss(nn.Module):
    # TODO: to implement completed InfoNCELoss
    def __init__(self, temp: float = 1.0):
        super().__init__()
        self.temp = temp

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        normalized_inputs = [F.normalize(features, p=2, dim=1) for features in inputs]
        similarity_matrix = [
            torch.matmul(normalized_inputs[i], normalized_inputs[j].transpose(0, 1)) / self.temp
            for i in range(len(normalized_inputs))
            for j in range(i + 1, len(normalized_inputs))
        ]
        return -torch.sum(torch.stack(similarity_matrix))
