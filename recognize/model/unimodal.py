from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import torch

from recognize.config import load_config

from .base import ClassifierModel, ClassifierOutput
from .multimodal import MultimodalBackbone, MultimodalInput


class UnimodalModel(ClassifierModel[MultimodalBackbone]):
    def __init__(
        self,
        backbone: MultimodalBackbone,
        *,
        feature_size: int = 128,
        num_classes: int = 2,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__(
            backbone,
            feature_size,
            num_classes=num_classes,
            class_weights=class_weights,
        )
        self.feature_size = feature_size

    def get_hyperparameter(self):
        return {
            "num_classes": self.num_classes,
        }

    def forward(self, inputs: MultimodalInput) -> ClassifierOutput:
        pooled_embs = self.backbone(inputs)["text"]
        return self.classify(pooled_embs, inputs.labels)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        backbone: MultimodalBackbone,
        *,
        class_weights: torch.Tensor | None = None,
    ):
        checkpoint_path = Path(checkpoint_path)
        with open(checkpoint_path / "config.json") as f:
            model_config = json.load(f)
        config = load_config(checkpoint_path / "config.toml")

        return cls(
            backbone,
            feature_size=config.model.feature_sizes[0],
            num_classes=model_config["num_classes"],
            class_weights=class_weights,
        )

    __call__: Callable[[MultimodalInput], ClassifierOutput]
