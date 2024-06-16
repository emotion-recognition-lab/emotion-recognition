from __future__ import annotations

import numpy as np
import whisperx
from loguru import logger
from opencc import OpenCC

from recognize.dataset import Preprocessor
from recognize.model import (
    LowRankFusionLayer,
    MultimodalBackbone,
    MultimodalModel,
)


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

        self.whisper_model = whisperx.load_model(
            "medium", device="cuda", download_root="./public/models/whisper"
        )
        self.preprocessor = Preprocessor.from_pretrained(f"{checkpoint}/preprocessor")
        self.emotion_model = (
            MultimodalModel.from_checkpoint(
                model_checkpoint, backbone=backbone, fusion_layer=fusion_layer
            )
            .cuda()
            .eval()
        )
        self.cc_model = OpenCC("t2s")

    def emotion_estimate(
        self,
        video_path: str = "./public/media/video.mp4",
        audio_path: str = "./public/media/audio.wav",
    ) -> int:
        from recognize.model import (
            LazyMultimodalInput,
        )

        text = self.extrct_text(audio_path)
        inputs = LazyMultimodalInput(
            preprocessor=self.preprocessor,
            texts=[text],
            # audio_paths=[audio_path],
            video_paths=[video_path],
        ).cuda()
        outputs = self.emotion_model(inputs)
        emotion_level = 1 / (1 + np.exp(-outputs.logits[0][0].cpu().numpy()))
        emotion_level = 50 + emotion_level * 50
        return emotion_level

    def extrct_text(self, audio_path: str = "tmp_audio.wav"):
        result = self.whisper_model.transcribe(audio_path, language="zh")
        if len(result["segments"]) == 0:
            return ""
        text = "。".join(seg["text"] for seg in result["segments"])
        text = self.cc_model.convert(text)
        logger.debug(f"Extracted text: {text}")
        return text
