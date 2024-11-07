from __future__ import annotations

from pathlib import Path

import torch
from typing_extensions import deprecated

from recognize.config import load_inference_config
from recognize.model import (
    LazyMultimodalInput,
    MultimodalBackbone,
    MultimodalModel,
    UnimodalModel,
)
from recognize.module import gen_fusion_layer, get_feature_sizes_dict
from recognize.preprocessor import Preprocessor


class EmotionEstimator:
    def __init__(self, checkpoint: Path) -> None:
        config = load_inference_config(checkpoint / "inference.toml")

        backbone = MultimodalBackbone.from_checkpoint(
            Path(f"{checkpoint}/backbone"),
            use_cache=False,
        )
        preprocessor = Preprocessor.from_pretrained(checkpoint / "preprocessor")

        if config.model.fusion is None:
            model = (
                UnimodalModel.from_checkpoint(
                    checkpoint,
                    backbone,
                )
                .cuda()
                .eval()
            )
        else:
            feature_sizes_dict = get_feature_sizes_dict(config.model.encoder)
            fusion_layer = gen_fusion_layer(config.model.fusion, feature_sizes_dict)
            model = (
                MultimodalModel.from_checkpoint(
                    checkpoint,
                    backbone,
                    fusion_layer,
                )
                .cuda()
                .eval()
            )

        self.preprocessor = preprocessor
        self.emotion_model = model

    def compute_logits(
        self,
        text: str | None = None,
        video_path: Path | None = None,
        audio_path: Path | None = None,
    ) -> torch.Tensor:
        inputs = LazyMultimodalInput(
            preprocessor=self.preprocessor,
            texts=[text] if text is not None else None,
            audio_paths=[audio_path.as_posix()] if audio_path is not None and audio_path.exists() else None,
            video_paths=[video_path.as_posix()] if video_path is not None and video_path.exists() else None,
        ).cuda()
        return self.emotion_model(inputs).logits[0]

    def classify(
        self,
        text: str | None = None,
        video_path: Path | None = None,
        audio_path: Path | None = None,
    ) -> int:
        logits = self.compute_logits(text, video_path, audio_path)
        cls = torch.argmax(logits, dim=-1).item()
        assert isinstance(cls, int)
        return cls

    def estimate(
        self,
        text: str | None = None,
        video_path: Path | None = None,
        audio_path: Path | None = None,
    ) -> float:
        logits = self.compute_logits(text, video_path, audio_path)
        probabilities = torch.softmax(logits, 0)
        expected_value = torch.sum(
            probabilities
            * torch.arange(self.emotion_model.num_classes, dtype=probabilities.dtype, device=probabilities.device)
        )

        score = (1 - expected_value / 3) * 100
        return score.item()

    @deprecated("use `estimate` instead")
    def emotion_estimate(
        self,
        text: str | None = None,
        video_path: Path | None = None,
        audio_path: Path | None = None,
    ) -> float:
        return self.estimate(text, video_path, audio_path)
