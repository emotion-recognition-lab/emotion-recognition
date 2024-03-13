from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from .base import Backbone, ClassifierModel, ClassifierOutput
from .multimodal import MultimodalInput


class TextBackbone(Backbone):
    def __init__(
        self,
        text_backbone: nn.Module,
    ):
        super().__init__(text_backbone.config.hidden_size)
        self.projector = nn.Linear(self.hidden_size, self.hidden_size)
        self.text_backbone = text_backbone

    def forward(self, inputs: MultimodalInput):
        text_outputs = self.text_backbone(inputs.text_input_ids, attention_mask=inputs.text_attention_mask)
        text_embs = text_outputs.last_hidden_state
        if inputs.audio_attention_mask is not None:
            text_pooled_embs = torch.stack(
                [sent_embs[atn_mask].mean(dim=0) for sent_embs, atn_mask in zip(text_embs, inputs.audio_attention_mask)]
            )
        else:
            text_pooled_embs = text_embs.mean(dim=1)

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

    @torch.compile()
    def forward(self, inputs: MultimodalInput) -> ClassifierOutput:
        features = self.backbone(inputs)
        return self.classify(features, inputs.labels)

    __call__: Callable[[MultimodalInput], ClassifierOutput]
