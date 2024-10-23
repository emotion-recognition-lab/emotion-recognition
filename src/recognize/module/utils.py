from __future__ import annotations

from recognize.module.fusion import ConcatFusionMoE, LowRankFusionLayer, MultiHeadFusionMoE, TensorFusionLayer
from recognize.module.moe import MultimodalMoE
from recognize.typing import FusionLayerLike, ModalType


def get_feature_sizes_dict(modals: list[ModalType], feature_sizes: list[int]) -> dict[str, int]:
    return dict(zip(modals, feature_sizes, strict=True))


def gen_fusion_layer(fusion: str, feature_sizes_dict: dict[str, int]) -> FusionLayerLike:
    fusion = eval(
        fusion,
        {
            "TensorFusionLayer": TensorFusionLayer,
            "LowRankFusionLayer": LowRankFusionLayer,
            "MultimodalMoE": MultimodalMoE,
            "MultiHeadFusionMoE": MultiHeadFusionMoE,
            "ConcatFusionMoE": ConcatFusionMoE,
            "feature_sizes_dict": feature_sizes_dict,
        },
    )

    assert isinstance(fusion, FusionLayerLike), f"{fusion} is not a FusionLayerLike, but {type(fusion)}"
    return fusion
