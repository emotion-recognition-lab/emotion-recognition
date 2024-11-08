from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from recognize.model.outputs import ClassifierOutput


def pearson_correlation(a: torch.Tensor, b: torch.Tensor, eps=1e-8) -> torch.Tensor:
    return F.cosine_similarity(a - a.mean(1).unsqueeze(1), b - b.mean(1).unsqueeze(1), eps=eps)


def inter_class_relation(y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


def generate_orthogonal_vectors(num_classes: int, hidden_dim: int) -> torch.Tensor:
    q_matrix, _ = torch.linalg.qr(torch.randn(hidden_dim, num_classes, device="cuda"))
    orthogonal_vectors = q_matrix.t()[:num_classes]
    return orthogonal_vectors


def merge_loss_fns[**P](loss_fns: Sequence[Callable[P, torch.Tensor]]) -> Callable[P, torch.Tensor]:
    def merged_loss_fn(*args: P.args, **kwargs: P.kwargs) -> torch.Tensor:
        return torch.sum(torch.stack([loss_fn(*args, **kwargs) for loss_fn in loss_fns]))

    return merged_loss_fn


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


class SupervisedProtoContrastiveLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        *,
        temp: float = 0.08,
        pool_size: int = 512,
        support_set_size: int = 64,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.temp = temp
        self.pool_size = pool_size
        self.support_set_size = support_set_size
        self.eps = eps
        self.default_centers = generate_orthogonal_vectors(num_classes, hidden_dim)
        self.pools = {idx: [] for idx in range(num_classes)}

    def score_fn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (1 + F.cosine_similarity(x, y, dim=-1)) / 2 + self.eps

    def calculate_centers(self) -> torch.Tensor:
        curr_centers = self.default_centers[:]
        for idx in range(self.num_classes):
            if len(self.pools[idx]) < self.support_set_size:
                continue
            tensor_center = torch.stack(self.pools[idx], 0)
            perm = torch.randperm(tensor_center.size(0))
            select_idx = perm[: self.support_set_size]
            # select_idx = random.sample(range(tensor_center.size(0)), self.support_set_size)
            curr_centers[idx] = tensor_center[select_idx].mean(0)
        return curr_centers

    def update_pools(self, embeddings: torch.Tensor, labels: torch.Tensor):
        for idx, label in enumerate(labels):
            label = label.item()
            assert isinstance(label, int), "label must be an integer"
            self.pools[label].append(embeddings[idx].detach())
            random.shuffle(self.pools[label])
            self.pools[label] = self.pools[label][: self.pool_size]

    def compute_scores_and_masks(
        self, concated_embeddings: torch.Tensor, concated_labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = concated_embeddings.size(0)
        mask1 = concated_labels.unsqueeze(0).expand(batch_size, batch_size)
        mask2 = concated_labels.unsqueeze(1).expand(batch_size, batch_size)
        mask = 1 - torch.eye(batch_size, device=concated_embeddings.device)
        pos_mask = (mask1 == mask2).long()
        rep1 = concated_embeddings.unsqueeze(0).expand(batch_size, batch_size, -1)
        rep2 = concated_embeddings.unsqueeze(1).expand(batch_size, batch_size, -1)
        scores = self.score_fn(rep1, rep2)
        scores *= mask
        scores /= self.temp
        scores -= torch.max(scores).item()
        return scores, pos_mask * mask, 1 - pos_mask

    def calculate_loss(
        self, scores: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor, decoupled: bool
    ) -> torch.Tensor:
        if decoupled:
            pos_scores = scores * pos_mask
            pos_scores = pos_scores.sum(-1)
            pos_scores /= pos_mask.sum(-1) + self.eps
            neg_scores = torch.exp(scores) * neg_mask
            loss = -pos_scores + torch.log(neg_scores.sum(-1) + self.eps)
            masked_loss = loss[loss > 0]
        else:
            scores = torch.exp(scores)
            pos_scores = (scores * pos_mask).sum(-1)
            neg_scores = (scores * neg_mask).sum(-1)
            probs = pos_scores / (pos_scores + neg_scores)
            probs /= pos_mask.sum(-1) + self.eps
            loss = -torch.log(probs + self.eps)
            masked_loss = loss[loss > 0.3]
        return masked_loss.mean() if len(masked_loss) > 0 else torch.tensor(0.0, device=scores.device)

    def forward(self, output: ClassifierOutput, labels: torch.Tensor, *, decoupled: bool = False) -> torch.Tensor:
        embeddings = output.features
        curr_centers = self.calculate_centers()
        self.update_pools(embeddings, labels)

        concated_embeddings = torch.cat((embeddings, curr_centers), 0)
        pad_labels = torch.arange(self.num_classes, device=embeddings.device)
        concated_labels = torch.cat((labels, pad_labels), 0)

        scores, pos_mask, neg_mask = self.compute_scores_and_masks(concated_embeddings, concated_labels)
        return self.calculate_loss(scores, pos_mask, neg_mask, decoupled)


class SelfContrastiveLoss(nn.Module):
    def __init__(self, dims: Mapping[str, int], *, hidden_dim: int = 128):
        super().__init__()
        self.dims = dict(dims)
        self.hidden_dim = hidden_dim
        # sum_dim = sum(self.dims.values())
        self.named_projectors = nn.ModuleDict(
            {
                f"{name1}->{name2}": nn.Linear(dim1, dim2)
                for (name1, dim1), (name2, dim2) in itertools.product(self.dims.items(), repeat=2)
            }
        )

    def forward(self, output: ClassifierOutput, labels: torch.Tensor) -> torch.Tensor:
        embs_dict = output.embs_dict
        assert embs_dict is not None
        names = set(embs_dict.keys()) & set(self.dims.keys())

        total_loss = torch.tensor(0.0, device=output.logits.device)
        for name1, name2 in itertools.product(names, repeat=2):
            projector = self.named_projectors[f"{name1}->{name2}"]
            embs1 = embs_dict[name1]
            embs2 = embs_dict[name2]
            proj_embs1 = projector(embs1)
            loss = F.mse_loss(proj_embs1, embs2)
            total_loss += loss
        return total_loss
