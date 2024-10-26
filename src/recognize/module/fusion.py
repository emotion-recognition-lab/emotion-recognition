from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn.parameter import Parameter


class FusionLayer(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size

    @abstractmethod
    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor: ...

    __call__: Callable[[Mapping[str, torch.Tensor]], torch.Tensor]


class TensorFusionLayer(FusionLayer):
    def __init__(self, text_size: int, audio_size: int, video_size: int):
        super().__init__((text_size + 1) * (audio_size + 1) * (video_size + 1))
        self.text_size = text_size
        self.audio_size = audio_size
        self.video_size = video_size
        self.augmentation = Parameter(torch.ones(1))

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
    def __init__(self, dims: dict[str, int], rank: int, output_size: int, *, trainable_placeholder: bool = True):
        super().__init__(output_size)
        self.dims = dims
        self.rank = rank
        self.low_rank_weights = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(rank, dim, output_size)) for name, dim in dims.items()}
        )
        self.placeholders = nn.ParameterDict({name: nn.Parameter(torch.randn(1, dim)) for name, dim in dims.items()})
        if not trainable_placeholder:
            # TODO: if trainable_placeholder=False, then forward logic should be changed
            self.placeholders.requires_grad_(False)
        self.output_layer = nn.Linear(rank, 1)

    def forward(self, inputs: Mapping[str, torch.Tensor]):
        assert (
            0 < len(inputs) <= len(self.low_rank_weights)
        ), f"Number of inputs ({len(inputs)}) should be less equal to number of weights ({len(self.low_rank_weights)})"
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


class MeanEmbeddingsFusionLayer(FusionLayer):
    def mean_embs(self, embs_list: Iterable[torch.Tensor | None]) -> torch.Tensor:
        filtered_embs_list = torch.stack([emb for emb in embs_list if emb is not None])
        return torch.sum(filtered_embs_list, dim=0) / len(filtered_embs_list)

    def forward(self, inputs: Mapping[str, torch.Tensor]):
        return self.mean_embs(inputs.values())


class MultiHeadFusionMoE(FusionLayer):
    def __init__(
        self,
        dims: dict[str, int],
        experts: Sequence[tuple[tuple[str, ...], nn.Module, int]],
        output_size: int,
    ):
        """
        Experts are expected to be a sequence of tuples of the form (needed_inputs, expert_module, output_dim).
        """
        super().__init__(output_size)
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


class ConcatFusionMoE(FusionLayer):
    def __init__(self, dims: dict[str, int], output_size: int):
        super().__init__(output_size)
        self.dim_names = sorted(dims.keys())
        self.placeholders = nn.ParameterDict({name: nn.Parameter(torch.randn(1, dim)) for name, dim in dims.items()})
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
        batch_size = next(iter(inputs.values())).size(0)

        concatenated_inputs = torch.cat(
            [
                inputs[name] if name in inputs else torch.broadcast_to(self.placeholders[name], (batch_size, -1))
                for name in self.dim_names
            ],
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
