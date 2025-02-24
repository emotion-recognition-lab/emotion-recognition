from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from recognize.model.inputs import MultimodalInput

from .basic import Projector

if TYPE_CHECKING:
    from recognize.model import ClassifierOutput, UnimodalModel


def pearson_correlation(a: torch.Tensor, b: torch.Tensor, *, eps=1e-8) -> torch.Tensor:
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


class SimSiamLoss(nn.Module):
    def __init__(self, a_dim: int, b_dim: int):
        super().__init__()
        hidden_dim = a_dim + b_dim
        self.projectors = nn.ModuleList([Projector(a_dim, hidden_dim), Projector(b_dim, hidden_dim)])
        self.predictors = nn.ModuleList([Projector(hidden_dim, hidden_dim), Projector(hidden_dim, hidden_dim)])

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h_a = self.projectors[0](a)
        h_b = self.projectors[1](b)
        p_b = self.predictors[0](h_a)
        p_a = self.predictors[1](h_b)
        loss = (
            F.cosine_similarity(p_b, h_b.detach(), dim=-1).mean()
            + F.cosine_similarity(p_a, h_a.detach(), dim=-1).mean()
        )
        return loss


class LogitLoss(nn.Module):
    def __init__(self, beta: float = 1.0, gamma: float = 1.0, temp: float = 4.0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.temp = temp

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor):
        y_s = (z_s / self.temp).softmax(dim=1)
        y_t = (z_t / self.temp).softmax(dim=1)
        inter_loss = self.temp**2 * inter_class_relation(y_s, y_t)
        intra_loss = self.temp**2 * intra_class_relation(y_s, y_t)
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        return kd_loss


class FeatureLoss(nn.Module):
    def __init__(self, temp: float = 1.0):
        super().__init__()
        self.temp = temp

    def forward(self, features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        target_features = F.normalize(target_features, p=2, dim=1)
        features = F.normalize(features, p=2, dim=1)
        log_q = torch.log_softmax(torch.matmul(target_features, features.transpose(0, 1)) / self.temp, dim=1)
        p = torch.softmax(torch.matmul(target_features, target_features.transpose(0, 1)) / self.temp, dim=1)
        return F.kl_div(log_q, p, reduction="batchmean")


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(input, target, reduction="none")
        p = torch.exp(-ce_loss)
        loss = (1 - p) ** self.gamma * ce_loss
        return loss.mean()


class DistillationLoss(nn.Module):
    def __init__(self, teacher_model: UnimodalModel):
        super().__init__()
        self.teacher_model = teacher_model
        self.logit_loss_fn = LogitLoss()

    def forward(self, input: MultimodalInput, output: ClassifierOutput):
        with torch.no_grad():
            teacher_output = self.teacher_model(input)
        # if all(input.labels != torch.argmax(output.logits, dim=1)):
        #     return torch.tensor(0.0, device=output.logits.device)
        loss = self.logit_loss_fn(output.logits, teacher_output.logits)
        return loss


class MultiLoss(nn.Module):
    def __init__(self, loss_num: int):
        super().__init__()
        self.loss_num = loss_num
        self.sigmas_dota = nn.Parameter(torch.randn(self.loss_num))

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        factor = 0.5 / self.sigmas_dota
        loss_part = factor @ losses
        regular_part = torch.sum(torch.log(self.sigmas_dota))
        loss = loss_part + regular_part
        return loss


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


class PrototypeContrastiveLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        *,
        temp: float = 0.08,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.temp = temp
        self.eps = eps

    def score_fn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (1 + F.cosine_similarity(x, y, dim=-1)) / 2 + self.eps


class SupervisedProtoContrastiveLoss(PrototypeContrastiveLoss):
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
        super().__init__(num_classes, hidden_dim, temp=temp, eps=eps)

        self.pool_size = pool_size
        self.support_set_size = support_set_size
        self.default_centers = generate_orthogonal_vectors(num_classes, hidden_dim)
        self.pools = {idx: [] for idx in range(num_classes)}

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

    def forward(self, input: MultimodalInput, output: ClassifierOutput, *, decoupled: bool = False) -> torch.Tensor:
        embeddings = output.features
        labels = input.labels
        assert labels is not None

        curr_centers = self.calculate_centers()
        self.update_pools(embeddings, labels)

        concated_embeddings = torch.cat((embeddings, curr_centers), 0)
        pad_labels = torch.arange(self.num_classes, device=embeddings.device)
        concated_labels = torch.cat((labels, pad_labels), 0)

        scores, pos_mask, neg_mask = self.compute_scores_and_masks(concated_embeddings, concated_labels)
        return self.calculate_loss(scores, pos_mask, neg_mask, decoupled)


class AdaptivePrototypeContrastiveLoss(PrototypeContrastiveLoss):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        *,
        temp: float = 0.08,
        eps: float = 1e-8,
        alpha: float = 0.5,
        gamma: float = 0.99,
    ):
        super().__init__(num_classes, hidden_dim, temp=temp, eps=eps)

        self.beta = alpha * (1 - gamma)
        self.gamma = gamma
        self.prototypes = generate_orthogonal_vectors(num_classes, hidden_dim)
        self.momentums = torch.zeros_like(self.prototypes)

    def update_prototypes(self, embeddings: torch.Tensor, labels: torch.Tensor):
        with torch.no_grad():
            for idx, label in enumerate(labels):
                label = label.item()
                assert isinstance(label, int), "label must be an integer"
                delta = embeddings[idx] - self.prototypes[label]
                self.momentums[label] = self.gamma * self.momentums[label] + self.beta * delta

            target = self.prototypes + self.momentums

            q_matrix, _ = torch.linalg.qr(target.T)
            self.prototypes = q_matrix.t()[: self.prototypes.shape[0]]

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

    def calculate_loss(self, scores: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor) -> torch.Tensor:
        pos_scores = scores * pos_mask
        pos_scores = pos_scores.sum(-1)
        pos_scores /= pos_mask.sum(-1) + self.eps
        neg_scores = torch.exp(scores) * neg_mask
        loss = -pos_scores + torch.log(neg_scores.sum(-1) + self.eps)
        masked_loss = loss[loss > 0]
        return masked_loss.mean() if len(masked_loss) > 0 else torch.tensor(0.0, device=scores.device)

    def forward(self, input: MultimodalInput, output: ClassifierOutput) -> torch.Tensor:
        features = output.features
        labels = input.labels
        assert labels is not None

        self.update_prototypes(features, labels)

        pad_labels = torch.arange(self.num_classes, device=features.device)

        concated_features = torch.cat((features, self.prototypes), 0)
        concated_labels = torch.cat((labels, pad_labels), 0)

        scores, pos_mask, neg_mask = self.compute_scores_and_masks(concated_features, concated_labels)
        return self.calculate_loss(scores, pos_mask, neg_mask)


class MultiParticleAdaptivePrototypeContrastiveLoss(PrototypeContrastiveLoss):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        *,
        temp: float = 0.08,
        eps: float = 1e-8,
        particle_num: int = 5,
    ):
        super().__init__(num_classes, hidden_dim, temp=temp, eps=eps)

        self.particle_num = particle_num
        self.prototypes = generate_orthogonal_vectors(particle_num * num_classes, hidden_dim).reshape(
            num_classes, particle_num, hidden_dim
        )
        self.momentums = torch.zeros_like(self.prototypes)

    def update_prototypes(self, embeddings: torch.Tensor, labels: torch.Tensor):
        embeddings = embeddings.view(-1, 1, self.hidden_dim)
        with torch.no_grad():
            for idx, label in enumerate(labels):
                label = label.item()
                assert isinstance(label, int), "label must be an integer"
                delta = embeddings[idx] - self.prototypes[label]
                self.momentums[label] = self.gamma * self.momentums[label] + self.beta * delta
            for center, momentum in zip(self.prototypes, self.momentums, strict=True):
                center += momentum

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

    def calculate_loss(self, scores: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor) -> torch.Tensor:
        pos_scores = scores * pos_mask
        pos_scores = pos_scores.sum(-1)
        pos_scores /= pos_mask.sum(-1) + self.eps
        neg_scores = torch.exp(scores) * neg_mask
        loss = -pos_scores + torch.log(neg_scores.sum(-1) + self.eps)
        masked_loss = loss[loss > 0]
        return masked_loss.mean() if len(masked_loss) > 0 else torch.tensor(0.0, device=scores.device)

    def forward(self, input: MultimodalInput, output: ClassifierOutput) -> torch.Tensor:
        features = output.features
        labels = input.labels
        assert labels is not None

        self.update_prototypes(features, labels)

        concated_features = torch.cat((features, self.prototypes), 0)
        pad_labels = torch.arange(self.num_classes, device=features.device)
        concated_labels = torch.cat((labels, pad_labels), 0)

        scores, pos_mask, neg_mask = self.compute_scores_and_masks(concated_features, concated_labels)
        return self.calculate_loss(scores, pos_mask, neg_mask)


class SelfContrastiveLoss(nn.Module):
    def __init__(self, dims: Mapping[str, int], main_modal: str, *, hidden_dim: int = 128):
        super().__init__()
        self.dims = dict(dims)
        self.hidden_dim = hidden_dim
        self.main_modal = main_modal
        self.feature_loss_fn = FeatureLoss()
        self.named_projectors = nn.ModuleDict(
            {
                f"{name1}->{name2}": nn.Linear(dim1, dim2)
                for (name1, dim1), (name2, dim2) in itertools.product(self.dims.items(), repeat=2)
            }
        )

    def forward(self, input: MultimodalInput, output: ClassifierOutput) -> torch.Tensor:
        embs_dict = output.embs_dict
        assert embs_dict is not None
        total_loss = torch.tensor(0.0, device=output.logits.device)
        modalities = set(embs_dict.keys()) & set(self.dims.keys())
        for modal in modalities - {self.main_modal}:
            total_loss += self.feature_loss_fn(embs_dict[modal], embs_dict[self.main_modal])

        return total_loss


class CrossModalContrastiveLoss(nn.Module):
    def __init__(self, dims: Mapping[str, int], main_modal: str):
        super().__init__()
        self.dims = dict(dims)
        self.main_modal = main_modal
        self.feature_loss_fn = FeatureLoss()

    def forward(self, input: MultimodalInput, output: ClassifierOutput) -> torch.Tensor:
        embs_dict = output.embs_dict
        assert embs_dict is not None
        total_loss = torch.tensor(0.0, device=output.logits.device)
        modalities = set(embs_dict.keys()) & set(self.dims.keys())
        main_modal_embs = embs_dict[self.main_modal].detach()
        for modal in modalities - {self.main_modal}:
            total_loss += self.feature_loss_fn(embs_dict[modal], main_modal_embs)
        return total_loss


class ReconstructionLoss(nn.Module):
    def __init__(self, dims: Mapping[str, int], feature_size: int, *, alpha: float = 0.5):
        super().__init__()
        self.dims = dict(dims)
        self.feature_size = feature_size
        self.alpha = alpha
        self.loss_fn = SimSiamLoss(feature_size, sum(self.dims.values()))

    def forward(self, input: MultimodalInput, output: ClassifierOutput) -> torch.Tensor:
        embs_dict = output.embs_dict
        assert embs_dict is not None
        if len(embs_dict) != len(self.dims):
            return torch.tensor(0.0, device=output.logits.device)
        concatenated_inputs = torch.cat([embs_dict[name] for name in self.dims.keys()], dim=1).detach()
        loss = self.loss_fn(
            output.features,
            concatenated_inputs,
        )
        return self.alpha * loss
