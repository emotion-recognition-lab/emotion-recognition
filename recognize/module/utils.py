from __future__ import annotations

from recognize.module.fusion import FusionLayer, LowRankFusionLayer
from recognize.typing import ModalType


def gen_fusion_layer(fusion: str, modals: list[ModalType], feature_sizes: list[int]) -> FusionLayer:
    fusion = eval(
        fusion,
        {
            # "TensorFusionLayer": TensorFusionLayer,
            "LowRankFusionLayer": LowRankFusionLayer,
            "feature_sizes_dict": dict(zip(modals, feature_sizes, strict=True)),
        },
    )

    assert isinstance(fusion, FusionLayer), f"{fusion} is not a FusionLayer, but {type(fusion)}"
    return fusion
