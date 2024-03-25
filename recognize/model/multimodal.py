from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.nn.parameter import Parameter

from .base import Backbone, ClassifierModel, ClassifierOutput, ModelInput, Pooler


class MultimodalInput(ModelInput):
    text_input_ids: torch.Tensor
    text_attention_mask: torch.Tensor | None = None

    audio_input_values: torch.Tensor | None = None
    audio_attention_mask: torch.Tensor | None = None

    video_pixel_values: torch.Tensor | None = None
    video_head_mask: torch.Tensor | None = None

    labels: torch.Tensor | None = None


def compute_fusion_tensor(zl: torch.Tensor, zv: torch.Tensor | None, za: torch.Tensor | None):
    zl = torch.cat((zl, torch.ones(zl.size(0), 1).cuda()), dim=1)
    if zv is not None:
        zv = torch.cat((zv, torch.ones(zv.size(0), 1).cuda()), dim=1)
    else:
        zv = torch.ones(zl.size(0), 2).cuda()
    # za = torch.cat((za, torch.ones(za.size(0), 1)), dim=1)
    # fusion_tensor = torch.einsum("bi,bj,bk->bijk", zl, zv, za).view(zl.size(0), -1)
    fusion_tensor = torch.einsum("bi,bj->bij", zl, zv).view(zl.size(0), -1)
    return fusion_tensor


class FusionTensorNet(nn.Module):
    def __init__(self, text_size: int, audio_size: int, video_size: int):
        super().__init__()
        self.text_size = text_size
        self.audio_size = audio_size
        self.video_size = video_size
        self.augmentation = Parameter(torch.ones(1))

    def forward(self, zl: torch.Tensor, za: torch.Tensor | None, zv: torch.Tensor | None):
        augmentation = self.augmentation.broadcast_to(zl.size(0), 1)

        augmentation_zl = torch.cat((zl, augmentation), dim=1)
        if za is not None:
            augmentation_za = torch.cat((za, augmentation), dim=1)
        else:
            augmentation_za = augmentation.broadcast_to(zl.size(0), self.audio_size + 1)
        if zv is not None:
            augmentation_zv = torch.cat((zv, augmentation), dim=1)
        else:
            augmentation_zv = augmentation.broadcast_to(zl.size(0), self.video_size + 1)

        assert augmentation_zl.size(1) == self.text_size + 1
        assert augmentation_za.size(1) == self.audio_size + 1
        assert augmentation_zv.size(1) == self.video_size + 1

        fusion_tensor = torch.einsum("bi,bj,bk->bijk", zl, zv, za).view(zl.size(0), -1)
        return fusion_tensor


class MultimodalBackbone(Backbone):
    def __init__(
        self,
        text_backbone: nn.Module,
        audio_backbone: nn.Module | None = None,
        video_backbone: nn.Module | None = None,
    ):
        assert (audio_backbone is None or text_backbone.config.hidden_size == audio_backbone.config.hidden_size) and (
            video_backbone is None or text_backbone.config.hidden_size == video_backbone.config.hidden_size
        ), "Hidden size of text, audio and video backbones must be the same"
        self.backbone_hidden_size = text_backbone.config.hidden_size
        hidden_size = 128 + 1
        if audio_backbone is not None:
            hidden_size *= 1 + 1
        if video_backbone is not None:
            hidden_size *= 1 + 1
        super().__init__(768)
        self.text_backbone = self.pretrained_module(text_backbone)
        self.audio_backbone = self.pretrained_module(audio_backbone)
        self.video_backbone = self.pretrained_module(video_backbone)

        self.text_pooler = Pooler(self.backbone_hidden_size, 768)
        self.audio_pooler = Pooler(self.backbone_hidden_size, 1)
        self.video_pooler = Pooler(self.backbone_hidden_size, 1)

    def compute_pooled_embs(self, inputs: MultimodalInput):
        text_outputs = self.text_backbone(inputs.text_input_ids, attention_mask=inputs.text_attention_mask)
        text_embs = text_outputs.last_hidden_state
        text_pooled_embs = self.text_pooler(text_embs)

        if inputs.audio_input_values is not None and self.audio_backbone is not None:
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

    def mean_embs(self, embs_list: list[torch.Tensor | None]):
        filtered_embs_list = [emb for emb in embs_list if emb is not None]
        return sum(filtered_embs_list) / len(filtered_embs_list)

    def forward(self, inputs: MultimodalInput):
        # text_pooled_embs, audio_pooled_embs, video_pooled_embs = self.compute_pooled_embs(inputs)
        # return self.mean_embs([text_pooled_embs, audio_pooled_embs, video_pooled_embs])
        text_outputs = self.text_backbone(inputs.text_input_ids, attention_mask=inputs.text_attention_mask)
        text_embs = text_outputs.last_hidden_state
        text_pooled_embs = self.text_pooler(text_embs)
        return text_pooled_embs

    # def forward(self, inputs: MultimodalInput):
    #     text_pooled_embs, audio_pooled_embs, video_pooled_embs = self.compute_pooled_embs(inputs)
    #     fusion_tensor = compute_fusion_tensor(text_pooled_embs, audio_pooled_embs, video_pooled_embs)
    #     return fusion_tensor

    def compute_loss(self, inputs: MultimodalInput):
        text_pooled_embs, audio_pooled_embs, video_pooled_embs = self.compute_pooled_embs(inputs)


class MultimodalModel(ClassifierModel):
    def __init__(
        self,
        text_backbone: nn.Module,
        audio_backbone: nn.Module | None = None,
        video_backbone: nn.Module | None = None,
        num_classes: int = 2,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__(
            MultimodalBackbone(text_backbone, audio_backbone, video_backbone),
            num_classes=num_classes,
            class_weights=class_weights,
        )

    def forward(self, inputs: MultimodalInput) -> ClassifierOutput:
        features = self.backbone(inputs)
        return self.classify(features, inputs.labels)

    __call__: Callable[[MultimodalInput], ClassifierOutput]
