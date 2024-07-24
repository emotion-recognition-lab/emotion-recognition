from __future__ import annotations

import torch
from torch import nn
from torch.nn.parameter import Parameter


class FusionLayer(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size


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

        assert (
            augmentation_zl.size(1) == self.text_size + 1
        ), f"{augmentation_zl.size(1)} != {self.text_size + 1}"
        assert (
            augmentation_za.size(1) == self.audio_size + 1
        ), f"{augmentation_za.size(1)} != {self.audio_size + 1}"
        assert (
            augmentation_zv.size(1) == self.video_size + 1
        ), f"{augmentation_zv.size(1)} != {self.video_size + 1}"

        fusion_tensor = torch.einsum(
            "bi,bj,bk->bijk", augmentation_zl, augmentation_za, augmentation_zv
        ).view(zl.size(0), -1)
        return fusion_tensor


class LowRankFusionLayer(FusionLayer):
    def __init__(self, dims: list[int], rank: int, output_size: int):
        super().__init__(output_size)
        self.dims = dims
        self.rank = rank
        self.low_rank_weights = nn.ParameterList(
            [nn.Parameter(torch.randn(rank, dim, output_size)) for dim in dims]
        )
        self.output_layer = nn.Linear(rank, 1)

    def forward(self, *inputs: torch.Tensor | None):
        assert len(inputs) <= len(
            self.low_rank_weights
        ), "Number of inputs should be less than or equal to number of weights"
        # N*d x R*d*h => R*N*h ~reshape~> N*h*R -> N*h*1 ~squeeze~> N*h
        fusion_tensors = [
            torch.matmul(i, w)
            for i, w in zip(inputs, self.low_rank_weights, strict=False)
            if i is not None
        ]
        product_tensor = torch.prod(torch.stack(fusion_tensors), dim=0)
        output = self.output_layer(product_tensor.permute(1, 2, 0)).squeeze(dim=-1)
        return output


def gen_fusion_layer(fusion: str) -> FusionLayer:
    fusion = eval(
        fusion,
        {
            "TensorFusionLayer": TensorFusionLayer,
            "LowRankFusionLayer": LowRankFusionLayer,
        },
    )
    assert isinstance(fusion, FusionLayer), f"{fusion} is not a FusionLayer"
    return fusion
