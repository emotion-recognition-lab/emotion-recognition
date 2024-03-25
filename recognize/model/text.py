from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from .base import Backbone, ClassifierModel, ClassifierOutput, Pooler
from .multimodal import MultimodalInput


class TextBackbone(Backbone):
    def __init__(
        self,
        text_backbone: nn.Module,
    ):
        super().__init__(text_backbone.config.hidden_size)
        self.text_backbone = self.pretrained_module(text_backbone)
        self.pooler = Pooler(self.hidden_size)

    def forward(self, inputs: MultimodalInput):
        text_outputs = self.text_backbone(inputs.text_input_ids, attention_mask=inputs.text_attention_mask)
        text_embs = text_outputs.last_hidden_state
        text_pooled_embs = self.pooler(text_embs)

        return text_pooled_embs


class TextModel(ClassifierModel):
    def __init__(
        self,
        text_backbone: nn.Module,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__(
            TextBackbone(text_backbone),
            num_classes=num_classes,
            class_weights=class_weights,
        )

    # @torch.compile()
    def forward(self, inputs: MultimodalInput) -> ClassifierOutput:
        features = self.backbone(inputs)
        return self.classify(features, inputs.labels)

    __call__: Callable[[MultimodalInput], ClassifierOutput]
