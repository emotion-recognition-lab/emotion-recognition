from __future__ import annotations

import torch

from recognize.config import load_inference_config
from recognize.model import (
    LazyMultimodalInput,
    MultimodalBackbone,
    MultimodalModel,
)
from recognize.model.unimodal import UnimodalModel
from recognize.module import gen_fusion_layer
from recognize.preprocessor import Preprocessor


class EmotionEstimator:
    def __init__(self, checkpoint: str) -> None:
        config = load_inference_config(f"{checkpoint}/config.toml")

        backbone = MultimodalBackbone.from_checkpoint(
            f"{checkpoint}/backbones",
            use_cache=False,
        )
        if config.model.fusion is None:
            self.emotion_model = (
                UnimodalModel.from_checkpoint(
                    checkpoint,
                    backbone,
                )
                .cuda()
                .eval()
            )
        else:
            fusion_layer = gen_fusion_layer(config.model.fusion, config.model.modals, config.model.feature_sizes)
            self.emotion_model = (
                MultimodalModel.from_checkpoint(
                    checkpoint,
                    backbone,
                    fusion_layer,
                )
                .cuda()
                .eval()
            )

        self.preprocessor = Preprocessor.from_pretrained(f"{checkpoint}/preprocessor")

    def emotion_estimate(
        self,
        text: str | None = None,
        video_path: str | None = None,
        audio_path: str | None = None,
    ) -> float:
        inputs = LazyMultimodalInput(
            preprocessor=self.preprocessor,
            texts=[text] if text is not None else None,
            audio_paths=[audio_path] if audio_path is not None else None,
            video_paths=[video_path] if video_path is not None else None,
        ).cuda()
        outputs = self.emotion_model(inputs)
        logits = outputs.logits[0]
        probabilities = torch.softmax(logits, 0)
        expected_value = torch.sum(
            probabilities
            * torch.arange(self.emotion_model.num_classes, dtype=probabilities.dtype, device=probabilities.device)
        )
        score = (expected_value / 3) * 100
        return score.item()
