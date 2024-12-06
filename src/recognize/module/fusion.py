from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal, overload

import torch
import torch.nn.functional as F
from torch import nn

from .basic import CrossAttention, Projector, SelfAttentionProjector
from .loss import SimSiamLoss
from .router import NoiseRouter


@overload
def support_modal_missing(cls: type[FusionLayer]) -> type[FusionLayer]: ...
@overload
def support_modal_missing(cls: None = None) -> Callable[[type[FusionLayer]], type[FusionLayer]]: ...


def support_modal_missing(
    cls: type[FusionLayer] | None = None,
) -> type[FusionLayer] | Callable[[type[FusionLayer]], type[FusionLayer]]:
    def decorator(cls: type[FusionLayer]) -> type[FusionLayer]:
        class WrappedFusionLayer(cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.placeholders = nn.ParameterDict(
                    {name: nn.Parameter(torch.randn(1, dim)) for name, dim in self.dims.items()}
                )

            @torch.compile(dynamic=True)
            def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
                batch_size = next(iter(inputs.values())).size(0)
                wrapped_inputs = {
                    name: torch.broadcast_to(self.placeholders[name], (batch_size, dim))
                    if name not in inputs
                    else inputs[name]
                    for name, dim in self.dims.items()
                }
                return cls.forward(self, wrapped_inputs)

            def forward_with_loss(
                self, inputs: Mapping[str, torch.Tensor], label: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                batch_size = next(iter(inputs.values())).size(0)
                wrapped_inputs = {
                    name: torch.broadcast_to(self.placeholders[name], (batch_size, dim))
                    if name not in inputs
                    else inputs[name]
                    for name, dim in self.dims.items()
                }
                return cls.forward_with_loss(self, wrapped_inputs, label)

        return WrappedFusionLayer

    if cls is None:
        return decorator
    else:
        return decorator(cls)


class FusionLayer(nn.Module):
    __call__: Callable[[Mapping[str, torch.Tensor]], torch.Tensor]

    def __init__(self, dims: Mapping[str, int], output_size: int):
        super().__init__()
        self.dims = dict(dims)
        self.output_size = output_size

    @abstractmethod
    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor: ...

    def forward_with_loss(
        self, inputs: Mapping[str, torch.Tensor], label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(inputs), torch.tensor(0.0, device=label.device)


@support_modal_missing()
class VallinaFusionLayer(FusionLayer):
    def __init__(
        self,
        dims: Mapping[str, int],
        output_size: int,
        *,
        depth: int = 1,
        method: Literal["mean", "concat"] = "concat",
    ):
        super().__init__(dims, output_size)
        self.method = method
        if method == "mean":
            # TODO: support different input sizes
            self.fc = Projector(next(iter(dims.values())), output_size, depth=depth)
        else:
            self.fc = Projector(sum(dims.values()), output_size, depth=depth)

    def mean(self, inputs: Iterable[torch.Tensor]) -> torch.Tensor:
        filtered_embs_list = torch.stack([emb for emb in inputs if emb is not None])
        return torch.mean(filtered_embs_list, dim=0)

    def cat(self, inputs: Iterable[torch.Tensor]) -> torch.Tensor:
        return torch.cat([emb for emb in inputs if emb is not None], dim=1)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.method == "mean":
            return self.fc(self.mean(inputs.values()))
        else:
            return self.fc(self.cat(inputs.values()))


@support_modal_missing()
class SelfAttentionFusionLayer(FusionLayer):
    def __init__(
        self,
        dims: Mapping[str, int],
        output_size: int,
        *,
        num_experts: int = 1,
        head_num: int = 8,
    ):
        super().__init__(dims, output_size)
        self.head_num = head_num
        self.pooler = nn.Linear(sum(dims.values()), output_size)
        self.multihead_attn = nn.MultiheadAttention(sum(dims.values()), head_num)

    def forward(self, inputs: Mapping[str, torch.Tensor]):
        embeddings = torch.cat([emb for emb in inputs.values() if emb is not None], dim=1)
        new_embeddings, _ = self.multihead_attn(embeddings, embeddings, embeddings)
        return self.pooler(new_embeddings)


class CrossAttentionFusionLayer(FusionLayer):
    def __init__(self, dims: Mapping[str, int], *, head_num: int = 8):
        sum_dim = sum(dims.values())
        super().__init__(dims, sum_dim)

        self.named_attns = nn.ModuleDict(
            {name: CrossAttention(dim, sum_dim - dim, head_num=head_num) for name, dim in dims.items()}
        )

    def forward(self, inputs: Mapping[str, torch.Tensor]):
        new_embeddings = []
        for name in self.dims.keys():
            attn = self.named_attns[name]
            embeddings = torch.cat([emb for emb_name, emb in inputs.items() if emb_name != name], dim=1)
            embeddings = attn(inputs[name], embeddings)
            new_embeddings.append(embeddings)
        return torch.cat(new_embeddings, dim=1)


class TensorFusionLayer(FusionLayer):
    def __init__(self, text_size: int, audio_size: int, video_size: int):
        super().__init__(
            {"T": text_size, "A": audio_size, "V": video_size}, (text_size + 1) * (audio_size + 1) * (video_size + 1)
        )
        self.text_size = text_size
        self.audio_size = audio_size
        self.video_size = video_size
        self.augmentation = nn.Parameter(torch.ones(1))

    def forward(self, inputs: Mapping[str, torch.Tensor]):
        zl = inputs["T"]
        za = inputs.get("A", None)
        zv = inputs.get("V", None)

        augmentation = self.augmentation.broadcast_to(zl.size(0), 1)
        augmentation_zl = torch.cat((zl, augmentation), dim=1)
        if za is not None:
            augmentation_za = torch.cat((za, augmentation), dim=1)
        else:
            augmentation_za = augmentation.broadcast_to(zl.size(0), self.audio_size + 1)
        if zv is not None:
            augmentation_zv = torch.cat((zv, augmentation), dim=1)
        else:
            augmentation_zv = augmentation.broadcast_to(zl.size(0), self.video_size + 1)

        assert augmentation_zl.size(1) == self.text_size + 1, f"{augmentation_zl.size(1)} != {self.text_size + 1}"
        assert augmentation_za.size(1) == self.audio_size + 1, f"{augmentation_za.size(1)} != {self.audio_size + 1}"
        assert augmentation_zv.size(1) == self.video_size + 1, f"{augmentation_zv.size(1)} != {self.video_size + 1}"

        fusion_tensor = torch.einsum("bi,bj,bk->bijk", augmentation_zl, augmentation_za, augmentation_zv).view(
            zl.size(0), -1
        )
        return fusion_tensor


class LowRankFusionLayer(FusionLayer):
    def __init__(self, dims: Mapping[str, int], rank: int, output_size: int, *, trainable_placeholder: bool = True):
        super().__init__(dims, output_size)
        self.rank = rank
        self.low_rank_weights = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(rank, dim, output_size)) for name, dim in dims.items()}
        )
        if trainable_placeholder:
            self.placeholders = nn.ParameterDict(
                {name: nn.Parameter(torch.randn(1, dim)) for name, dim in dims.items()}
            )
        else:
            self.placeholders = {name: torch.ones(1, dim) for name, dim in dims.items()}

        self.output_layer = nn.Linear(rank, 1)

    def forward(self, inputs: Mapping[str, torch.Tensor]):
        # N*d x R*d*h => R*N*h ~reshape~> N*h*R -> N*h*1 ~squeeze~> N*h
        fusion_tensors = [
            torch.matmul(input, self.low_rank_weights[name])
            if input is not None and name in self.low_rank_weights
            else self.placeholders[name]
            for name, input in inputs.items()
        ]
        product_tensor = torch.prod(torch.stack(fusion_tensors), dim=0)
        output = self.output_layer(product_tensor.permute(1, 2, 0)).squeeze(dim=-1)
        return output


class MultiHeadFusionMoE(FusionLayer):
    def __init__(
        self,
        dims: Mapping[str, int],
        experts: Sequence[tuple[tuple[str, ...], nn.Module, int]],
        output_size: int,
    ):
        """
        Experts are expected to be a sequence of tuples of the form (needed_inputs, expert_module, output_dim).
        """
        super().__init__(dims, output_size)
        self.dim_names = sorted(dims.keys())
        # TODO: router maybe can be scaled up
        self.router = nn.ModuleDict({name: nn.Linear(dims[name], len(experts)) for name in self.dim_names})
        self.experts = nn.ModuleList(
            nn.Sequential(
                expert_module,
                nn.Linear(output_dim, output_size),
            )
            for _, expert_module, output_dim in experts
        )
        self.expert_needed_inputs = [needed_inputs for needed_inputs, _, _ in experts]

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        importances = torch.stack([self.router[name](inputs[name]) for name in self.dim_names if name in inputs])
        routing_weights = torch.softmax(torch.sum(importances, dim=0), dim=1)
        outputs = []
        for i, expert in enumerate(self.experts):
            expert_outputs = expert(inputs)
            outputs.append(routing_weights[:, i : i + 1] * expert_outputs)
        return torch.sum(torch.stack(outputs), dim=0)


@support_modal_missing()
class ConcatFusionMoE(FusionLayer):
    def __init__(self, dims: Mapping[str, int], output_size: int):
        super().__init__(dims, output_size)
        self.dim_names = sorted(dims.keys())
        self.router = nn.Sequential(
            nn.Linear(sum(dims.values()), len(dims) + 1),
            nn.Softmax(dim=-1),
        )
        self.experts = nn.ModuleList(
            [
                *[nn.Linear(dim, output_size) for dim in dims.values()],
                nn.Linear(sum(dims.values()), output_size),
            ]
        )

    def forward(self, inputs: Mapping[str, torch.Tensor]):
        concatenated_inputs = torch.cat(
            [inputs[name] for name in self.dim_names],
            dim=1,
        )
        routing_weights = self.router(concatenated_inputs)
        outputs = []
        for i, name in enumerate(self.dim_names):
            input = inputs.get(name, self.placeholders[name])
            if input is not None:
                input = input.view(input.size(0), -1)
                outputs.append(routing_weights[:, i : i + 1] * self.experts[i](input))

        return torch.sum(torch.stack(outputs), dim=0)


@support_modal_missing()
class DisentanglementFusion(FusionLayer):
    def __init__(
        self,
        dims: Mapping[str, int],
        private_feature_size: int,
        shared_feature_size: int,
    ):
        """
        Experts are expected to be a sequence of tuples of the form (needed_inputs, expert_module, output_dim).
        """
        num_modal = len(dims)
        super().__init__(dims, shared_feature_size * (num_modal - 1) + private_feature_size)
        self.dim_names = sorted(dims.keys())
        # TODO: dims should be the same, or use Pooler
        self.shared_projector = SelfAttentionProjector(
            next(iter(dims.values())),
            shared_feature_size * (num_modal - 1),
            head_num=4 * num_modal,
            depth=2 * num_modal,
        )
        self.named_cross_projectors = nn.ModuleDict(
            {
                name: SelfAttentionProjector(
                    dims[name], shared_feature_size, head_num=4 * (num_modal - 1), depth=2 ** (num_modal - 1)
                )
                for name in dims.keys()
            }
        )
        self.named_private_projectors = nn.ModuleDict(
            {
                name: SelfAttentionProjector(dims[name], private_feature_size, head_num=4, depth=2)
                for name in dims.keys()
            }
        )
        self.cross_reconstruction_predictor = nn.ModuleDict(
            {name: Projector(shared_feature_size * (num_modal - 1), dims[name], depth=4) for name in dims.keys()}
        )
        self.fusion_reconstruction_loss_fn = SimSiamLoss(
            shared_feature_size * (num_modal - 1) + private_feature_size, sum(dims.values())
        )
        self.shared_router = NoiseRouter(sum(dims.values()), num_modal)
        self.private_router = NoiseRouter(sum(dims.values()), num_modal)
        self.cross_attn = CrossAttentionFusionLayer(
            {"shared": shared_feature_size * (num_modal - 1), "private": private_feature_size}
        )

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        concatenated_inputs = torch.cat([inputs[modal] for modal in self.dim_names if modal in inputs], dim=1).detach()
        private_routing_weights = self.private_router(concatenated_inputs)
        private_features_dict = {modal: self.named_private_projectors[modal](inputs[modal]) for modal in self.dim_names}
        private_features = torch.einsum(
            "bn,bnk->bk",
            private_routing_weights,
            torch.stack([private_features_dict[modal] for modal in self.dim_names], 1),
        )
        shared_routing_weights = self.shared_router(concatenated_inputs)
        shared_features_dict = {modal: self.shared_projector(inputs[modal]) for modal in self.dim_names}
        shared_features = torch.einsum(
            "bn,bnk->bk",
            shared_routing_weights,
            torch.stack([shared_features_dict[modal] for modal in self.dim_names], 1),
        )
        fusion_features = self.cross_attn(
            {
                "shared": shared_features,
                "private": private_features,
            }
        )
        return fusion_features

    def forward_with_loss(
        self, inputs: Mapping[str, torch.Tensor], label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        concatenated_inputs = torch.cat([inputs[modal] for modal in self.dim_names if modal in inputs], dim=1).detach()
        private_routing_weights = self.private_router(concatenated_inputs)
        private_features_dict = {modal: self.named_private_projectors[modal](inputs[modal]) for modal in self.dim_names}
        private_features = torch.einsum(
            "bn,bnk->bk",
            private_routing_weights,
            torch.stack([private_features_dict[modal] for modal in self.dim_names], 1),
        )
        shared_routing_weights = self.shared_router(concatenated_inputs)
        shared_features_dict = {modal: self.shared_projector(inputs[modal]) for modal in self.dim_names}
        shared_features = torch.einsum(
            "bn,bnk->bk",
            shared_routing_weights,
            torch.stack([shared_features_dict[modal] for modal in self.dim_names], 1),
        )
        fusion_features = self.cross_attn(
            {
                "shared": shared_features,
                "private": private_features,
            }
        )
        cross_features_dict = {
            modal1: torch.cat(
                [self.named_cross_projectors[modal1](inputs[modal2]) for modal2 in self.dim_names if modal1 != modal2],
                dim=1,
            )
            for modal1 in self.dim_names
        }
        feature_loss = torch.sum(
            torch.stack(
                [
                    F.smooth_l1_loss(shared_features_dict[modal], cross_features_dict[modal].detach())
                    for modal in self.dim_names
                ]
            ),
            dim=0,
        )
        reconstruction_loss = self.fusion_reconstruction_loss_fn(
            fusion_features,
            concatenated_inputs,
        )
        for modal in self.dim_names:
            reconstruction_loss += F.smooth_l1_loss(
                self.cross_reconstruction_predictor[modal](cross_features_dict[modal]),
                inputs[modal].detach(),
            )
        return fusion_features, (feature_loss + reconstruction_loss) / 2
