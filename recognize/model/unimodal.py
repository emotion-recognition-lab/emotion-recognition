from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch

from .base import ClassifierModel, ClassifierOutput, Pooler
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
        self.pooler = Pooler(self.backbone.output_size, feature_size)

    def get_hyperparameter(self):
        return {
            "feature_size": self.feature_size,
            "num_classes": self.num_classes,
        }

    def pool_embs(
        self,
        embs: torch.Tensor,
    ) -> torch.Tensor:
        return self.pooler(embs)

    def forward(self, inputs: MultimodalInput) -> ClassifierOutput:
        embs = self.backbone(inputs)[0]
        pooled_embs = self.pool_embs(embs)
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
        with open(checkpoint_path / "config.json", "r") as f:
            model_config = json.load(f)
        return cls(backbone, **model_config, class_weights=class_weights)

    __call__: Callable[[MultimodalInput], ClassifierOutput]
