from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch

from recognize.config import load_inference_config

from .base import ClassifierModel, ClassifierOutput
from .multimodal import MultimodalBackbone, MultimodalInput


class UnimodalModel(ClassifierModel[MultimodalBackbone]):
    def __init__(
        self,
        backbone: MultimodalBackbone,
        *,
        feature_size: int = 128,
        num_classes: int = 2,
        num_experts: int = 1,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__(
            backbone,
            feature_size,
            num_classes=num_classes,
            num_experts=num_experts,
            class_weights=class_weights,
        )
        self.feature_size = feature_size

    def get_hyperparameter(self):
        return {
            "num_classes": self.num_classes,
        }

    def forward(self, inputs: MultimodalInput) -> ClassifierOutput:
        pooled_embs: dict[str, torch.Tensor] = self.backbone(inputs)
        if len(pooled_embs) != 1:
            # NOTE: dataset maybe not have some modalities
            assert inputs.labels is not None, "labels is required"
            return ClassifierOutput(
                logits=torch.zeros((inputs.labels.shape[0], self.num_classes), device=inputs.device),
            )
        return self.classify(next(iter(pooled_embs.values())), inputs.labels)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        backbone: MultimodalBackbone,
        *,
        class_weights: torch.Tensor | None = None,
    ):
        from recognize.utils import load_model

        checkpoint_path = Path(checkpoint_path)
        config = load_inference_config(checkpoint_path / "inference.toml")

        model = cls(
            backbone,
            feature_size=config.model.feature_sizes[0],
            num_classes=config.num_classes,
            num_experts=config.model.num_experts,
            class_weights=class_weights,
        )
        load_model(checkpoint_path, model)

        return model

    __call__: Callable[[MultimodalInput], ClassifierOutput]
