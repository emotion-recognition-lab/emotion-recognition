from __future__ import annotations

from recognize.config import ModelEncoderConfig
from recognize.module.fusion import (
    ConcatFusionMoE,
    CrossAttentionFusionLayer,
    DisentanglementFusion,
    LowRankFusionLayer,
    MultiHeadFusionMoE,
    PrivateFeatureFusion,
    SelfAttentionFusionLayer,
    SharedFeatureFusion,
    TensorFusionLayer,
    VallinaFusionLayer,
)
from recognize.typing import FusionLayerLike, ModalType


def get_feature_sizes_dict(config_model_encoder: dict[ModalType, ModelEncoderConfig]) -> dict[str, int]:
    return {modal: config.feature_size for modal, config in config_model_encoder.items()}


def gen_fusion_layer(fusion: str, feature_sizes_dict: dict[str, int]) -> FusionLayerLike:
    # TODO: split fusion to fusion_cls and fusion_args
    fusion = eval(
        fusion,
        {
            "VallinaFusionLayer": VallinaFusionLayer,
            "SelfAttentionFusionLayer": SelfAttentionFusionLayer,
            "CrossAttentionFusionLayer": CrossAttentionFusionLayer,
            "TensorFusionLayer": TensorFusionLayer,
            "LowRankFusionLayer": LowRankFusionLayer,
            "MultiHeadFusionMoE": MultiHeadFusionMoE,
            "ConcatFusionMoE": ConcatFusionMoE,
            "DisentanglementFusion": DisentanglementFusion,
            "SharedFeatureFusion": SharedFeatureFusion,
            "PrivateFeatureFusion": PrivateFeatureFusion,
            "feature_sizes_dict": feature_sizes_dict,
        },
    )
    assert isinstance(fusion, FusionLayerLike), f"{fusion} is not a FusionLayerLike, but {type(fusion)}"
    return fusion
