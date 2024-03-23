from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from .base import Backbone, ClassifierModel, ClassifierOutput, ModelInput, Pooler


class MultimodalInput(ModelInput):
    text_input_ids: torch.Tensor
    text_attention_mask: torch.Tensor | None = None

    audio_input_values: torch.Tensor | None = None
    audio_attention_mask: torch.Tensor | None = None

    video_pixel_values: torch.Tensor | None = None
    video_head_mask: torch.Tensor | None = None

    labels: torch.Tensor | None = None


class MultimodalBackbone(Backbone):
    def __init__(
        self,
        text_backbone: nn.Module,
        audio_backbone: nn.Module,
        video_backbone: nn.Module | None = None,
    ):
        assert (
            text_backbone.config.hidden_size == audio_backbone.config.hidden_size
        ), "Hidden size of text and audio backbones must be the same"

        super().__init__(text_backbone.config.hidden_size)
        self.text_backbone = text_backbone
        self.audio_backbone = audio_backbone
        self.video_backbone = video_backbone

        self.text_pooler = Pooler(self.hidden_size)
        self.audio_pooler = Pooler(self.hidden_size)
        self.video_pooler = Pooler(self.hidden_size)

    def compute_pooled_embs(self, inputs: MultimodalInput):
        text_outputs = self.text_backbone(inputs.text_input_ids, attention_mask=inputs.text_attention_mask)
        text_embs = text_outputs.last_hidden_state
        text_pooled_embs = self.text_pooler(text_embs)

        if inputs.audio_input_values is not None:
            audio_outputs = self.audio_backbone(inputs.audio_input_values, attention_mask=inputs.audio_attention_mask)
            audio_pooled_embs = self.audio_pooler(audio_outputs.last_hidden_state)
        else:
            audio_pooled_embs = None

        if inputs.video_pixel_values is not None and self.video_backbone is not None:
            video_outputs = self.video_backbone(inputs.video_pixel_values)
            video_pooled_embs = self.video_pooler(video_outputs.last_hidden_state)
        else:
            video_pooled_embs = None

        return text_pooled_embs, audio_pooled_embs, video_pooled_embs

    def compute_loss(self, inputs: MultimodalInput):
        text_pooled_embs, audio_pooled_embs, video_pooled_embs = self.compute_pooled_embs(inputs)

    def forward(self, inputs: MultimodalInput):
        text_pooled_embs, audio_pooled_embs, video_pooled_embs = self.compute_pooled_embs(inputs)
        if audio_pooled_embs is not None and video_pooled_embs is not None:
            output_embs = (text_pooled_embs + audio_pooled_embs + video_pooled_embs) / 3
        elif audio_pooled_embs is not None:
            output_embs = (text_pooled_embs + audio_pooled_embs) / 2
        elif video_pooled_embs is not None:
            output_embs = (text_pooled_embs + video_pooled_embs) / 2
        else:
            output_embs = text_pooled_embs

        return output_embs


class MultimodalModel(ClassifierModel):
    def __init__(
        self,
        text_backbone: nn.Module,
        audio_backbone: nn.Module,
        # video_backbone: nn.Module,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__(
            MultimodalBackbone(text_backbone, audio_backbone),
            num_classes=num_classes,
            class_weights=class_weights,
        )

    @torch.compile()
    def forward(self, inputs: MultimodalInput) -> ClassifierOutput:
        features = self.backbone(inputs)
        return self.classify(features, inputs.labels)

    __call__: Callable[[MultimodalInput], ClassifierOutput]
