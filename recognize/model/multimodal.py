from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from .base import Backbone, ClassifierModel, ClassifierOutput, ModelInput


class MultimodalInput(ModelInput):
    text_input_ids: torch.Tensor
    audio_input_values: torch.Tensor | None = None

    text_attention_mask: torch.Tensor | None = None
    audio_attention_mask: torch.Tensor | None = None

    labels: torch.Tensor | None = None


class MultimodalBackbone(Backbone):
    def __init__(
        self,
        text_backbone: nn.Module,
        audio_backbone: nn.Module,
    ):
        assert (
            text_backbone.config.hidden_size == audio_backbone.config.hidden_size
        ), "Hidden size of text and audio backbones must be the same"

        super().__init__(text_backbone.config.hidden_size)
        self.projector = nn.Linear(self.hidden_size, self.hidden_size)
        self.text_backbone = text_backbone
        self.audio_backbone = audio_backbone

    def forward(self, inputs: MultimodalInput):
        text_outputs = self.text_backbone(inputs.text_input_ids, attention_mask=inputs.text_attention_mask)
        text_embs = text_outputs.last_hidden_state
        if inputs.audio_attention_mask is not None:
            text_pooled_embs = torch.stack(
                [sent_embs[atn_mask].mean(dim=0) for sent_embs, atn_mask in zip(text_embs, inputs.audio_attention_mask)]
            )
        else:
            text_pooled_embs = text_embs.mean(dim=1)

        if inputs.audio_input_values is not None:
            audio_outputs = self.audio_backbone(inputs.audio_input_values, attention_mask=inputs.audio_attention_mask)
            audio_pooled_embs = audio_outputs.last_hidden_state.mean(dim=1)
            output_embs = (text_pooled_embs + audio_pooled_embs) / 2
        else:
            output_embs = text_pooled_embs

        return output_embs


class MultimodalModel(ClassifierModel):
    def __init__(
        self,
        text_backbone: nn.Module,
        audio_backbone: nn.Module,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__(
            MultimodalBackbone(text_backbone, audio_backbone),
            num_classes=num_classes,
            class_weights=class_weights,
        )

    def forward(self, inputs: MultimodalInput) -> ClassifierOutput:
        features = self.backbone(inputs)
        return self.classify(features, inputs.labels)

    __call__: Callable[[MultimodalInput], ClassifierOutput]
