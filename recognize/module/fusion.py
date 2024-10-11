from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import torch
from torch import nn
from torch.nn.parameter import Parameter


class FusionLayer(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size

    __call__: Callable[[dict[str, torch.Tensor | None]], torch.Tensor]


class TensorFusionLayer(FusionLayer):
    def __init__(self, text_size: int, audio_size: int, video_size: int):
        super().__init__((text_size + 1) * (audio_size + 1) * (video_size + 1))
        self.text_size = text_size
        self.audio_size = audio_size
        self.video_size = video_size
        self.augmentation = Parameter(torch.ones(1))

    def forward(self, zl: torch.Tensor, za: torch.Tensor | None, zv: torch.Tensor | None):
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
    def __init__(self, dims: dict[str, int], rank: int, output_size: int, *, trainable_placeholder: bool = False):
        super().__init__(output_size)
        self.dims = dims
        self.rank = rank
        self.low_rank_weights = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(rank, dim, output_size)) for name, dim in dims.items()}
        )
        self.placeholders = nn.ParameterDict({name: nn.Parameter(torch.randn(1, dim)) for name, dim in dims.items()})
        if not trainable_placeholder:
            self.placeholders.requires_grad_(False)
        self.output_layer = nn.Linear(rank, 1)

    def forward(self, inputs: Mapping[str, torch.Tensor | None]):
        assert (
            0 < len(inputs) <= len(self.low_rank_weights)
        ), f"Number of inputs ({len(inputs)}) should be less equal to number of weights ({len(self.low_rank_weights)})"
        # N*d x R*d*h => R*N*h ~reshape~> N*h*R -> N*h*1 ~squeeze~> N*h
        fusion_tensors = [
            torch.matmul(i, self.low_rank_weights[n])
            if i is not None and n in self.low_rank_weights
            else self.placeholders[n]
            for n, i in inputs.items()
        ]
        product_tensor = torch.prod(torch.stack(fusion_tensors), dim=0)
        output = self.output_layer(product_tensor.permute(1, 2, 0)).squeeze(dim=-1)
        return output


class MeanEmbeddingsFusionLayer(FusionLayer):
    def mean_embs(self, embs_list: Iterable[torch.Tensor | None]):
        filtered_embs_list = [emb for emb in embs_list if emb is not None]
        return sum(filtered_embs_list) / len(filtered_embs_list)

    def forward(self, inputs: Mapping[str, torch.Tensor | None]):
        return self.mean_embs(inputs.values())
