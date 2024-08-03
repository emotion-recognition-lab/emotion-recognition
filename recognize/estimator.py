from __future__ import annotations

import numpy as np
from loguru import logger

from recognize.dataset import Preprocessor
from recognize.model import (
    LazyMultimodalInput,
    MultimodalBackbone,
    MultimodalModel,
)
from recognize.module import LowRankFusionLayer


class EmotionEstimator:
    def __init__(self, checkpoint: str = "./public/models/emotion/"):
        model_checkpoint = f"{checkpoint}/"
        text_feature_size, audio_feature_size, video_feature_size = 128, 16, 1
        backbone = MultimodalBackbone.from_checkpoint(
            f"{model_checkpoint}/backbones",
            use_cache=False,
        )
        fusion_layer = LowRankFusionLayer(
            [text_feature_size, audio_feature_size, video_feature_size], 16, 128
        )

        self.preprocessor = Preprocessor.from_pretrained(f"{checkpoint}/preprocessor")
        self.emotion_model = (
            MultimodalModel.from_checkpoint(
                model_checkpoint, backbone=backbone, fusion_layer=fusion_layer
            )
            .cuda()
            .eval()
        )

    def emotion_estimate(
        self,
        text: str | None = None,
        video_path: str | None = None,
        audio_path: str | None = None,
    ) -> int:
        inputs = LazyMultimodalInput(
            preprocessor=self.preprocessor,
            texts=[text] if text is not None else None,
            audio_paths=[audio_path] if audio_path is not None else None,
            video_paths=[video_path] if video_path is not None else None,
        ).cuda()
        outputs = self.emotion_model(inputs)
        emotion_level = 1 / (1 + np.exp(-outputs.logits[0][0].cpu().numpy()))
        emotion_level = 50 + emotion_level * 50
        return emotion_level

    def extrct_text(self, audio_path: str = "tmp_audio.wav"):
        print(123123123)
        segments, info = self.whisper_model.transcribe(audio_path, language="zh-cn")
        text = "。".join(seg.text for seg in segments)
        logger.debug(f"Extracted text: {text}")
        return text
