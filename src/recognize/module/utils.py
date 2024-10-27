from __future__ import annotations

from recognize.config import ModelEncoderConfig
from recognize.module.fusion import ConcatFusionMoE, LowRankFusionLayer, MultiHeadFusionMoE, TensorFusionLayer
from recognize.module.moe import MultimodalMoE
from recognize.typing import FusionLayerLike, ModalType


def get_feature_sizes_dict(config_model_encoder: dict[ModalType, ModelEncoderConfig]) -> dict[str, int]:
    return {modal: config.feature_size for modal, config in config_model_encoder.items()}


def gen_fusion_layer(fusion: str, feature_sizes_dict: dict[str, int]) -> FusionLayerLike:
    # TODO: split fusion to fusion_cls and fusion_args
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
