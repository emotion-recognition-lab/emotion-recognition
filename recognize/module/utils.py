from __future__ import annotations

from recognize.module.fusion import LowRankFusionLayer, TensorFusionLayer
from recognize.module.moe import MoELowRankFusionLayer
from recognize.typing import FusionLayerLike, ModalType


def gen_fusion_layer(fusion: str, modals: list[ModalType], feature_sizes: list[int]) -> FusionLayerLike:
    fusion = eval(
        fusion,
        {
            "TensorFusionLayer": TensorFusionLayer,
            "LowRankFusionLayer": LowRankFusionLayer,
            "MoELowRankFusionLayer": MoELowRankFusionLayer,
            "feature_sizes_dict": dict(zip(modals, feature_sizes, strict=True)),
        },
    )

    assert isinstance(fusion, FusionLayerLike), f"{fusion} is not a FusionLayerLike, but {type(fusion)}"
    return fusion
