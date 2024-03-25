from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.nn.parameter import Parameter

from ..cache import load_cached_tensors, save_cached_tensors
from .base import Backbone, ClassifierModel, ClassifierOutput, ModelInput, Pooler


class MultimodalInput(ModelInput):
    text_input_ids: torch.Tensor
    text_attention_mask: torch.Tensor | None = None

    audio_input_values: torch.Tensor | None = None
    audio_attention_mask: torch.Tensor | None = None

    video_pixel_values: torch.Tensor | None = None
    video_head_mask: torch.Tensor | None = None

    labels: torch.Tensor | None = None
    unique_id: str | list[str] = ""

    def __getitem__(self, index: int | list[int] | slice) -> MultimodalInput:
        assert isinstance(self.unique_id, list), "unique_id must be a list"

        text_input_ids = self.text_input_ids[index]
        text_attention_mask = self.text_attention_mask[index] if self.text_attention_mask is not None else None

        audio_input_values = self.audio_input_values[index] if self.audio_input_values is not None else None
        audio_attention_mask = self.audio_attention_mask[index] if self.audio_attention_mask is not None else None

        video_pixel_values = self.video_pixel_values[index] if self.video_pixel_values is not None else None
        video_head_mask = self.video_head_mask[index] if self.video_head_mask is not None else None

        labels = self.labels[index] if self.labels is not None else None
        unique_id = [self.unique_id[i] for i in index] if isinstance(index, list) else self.unique_id[index]

        return MultimodalInput(
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            audio_input_values=audio_input_values,
            audio_attention_mask=audio_attention_mask,
            video_pixel_values=video_pixel_values,
            video_head_mask=video_head_mask,
            labels=labels,
            unique_id=unique_id,
        )


class TensorFusionLayer(nn.Module):
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

        fusion_tensor = torch.einsum("bi,bj,bk->bijk", augmentation_zl, augmentation_za, augmentation_zv).view(
            zl.size(0), -1
        )
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
        hidden_size *= 1 + 1
        hidden_size *= 1 + 1
        super().__init__(hidden_size)
        self.text_backbone = self.pretrained_module(text_backbone)
        self.audio_backbone = self.pretrained_module(audio_backbone)
        self.video_backbone = self.pretrained_module(video_backbone)

        self.tensor_fusion_layer = TensorFusionLayer(128, 1, 1)

        self.text_pooler = Pooler(self.backbone_hidden_size, self.tensor_fusion_layer.text_size)
        self.audio_pooler = Pooler(self.backbone_hidden_size, self.tensor_fusion_layer.audio_size)
        self.video_pooler = Pooler(self.backbone_hidden_size, self.tensor_fusion_layer.video_size)

    def pool_embs(self, text_embs: torch.Tensor, audio_embs: torch.Tensor | None, video_embs: torch.Tensor | None):
        text_pooled_embs = self.text_pooler(text_embs)
        if audio_embs is not None:
            audio_pooled_embs = self.audio_pooler(audio_embs)
        else:
            audio_pooled_embs = None

        if video_embs is not None:
            video_pooled_embs = self.video_pooler(video_embs)
        else:
            video_pooled_embs = None

        return text_pooled_embs, audio_pooled_embs, video_pooled_embs

    def cached_compute_pooled_embs(self, inputs: MultimodalInput):
        assert isinstance(inputs.unique_id, list), "unique_id must be a list"
        cached_list, cached_index_list, no_cached_index_list = load_cached_tensors(inputs.unique_id)

        if len(no_cached_index_list) != 0:
            no_cached_inputs = inputs[no_cached_index_list]
            with torch.no_grad():
                text_embs, audio_embs, video_embs = self.compute_embs(no_cached_inputs)

            if self.is_frozen:
                save_cached_tensors(
                    inputs.unique_id,
                    {
                        "text_embs": text_embs,
                        "audio_embs": audio_embs,
                        "video_embs": video_embs,
                    },
                )

            cached_list, cached_index_list, no_cached_index_list = load_cached_tensors(inputs.unique_id)
        assert len(no_cached_index_list) == 0, "All tensors should be cached"

        embs_list_dict: dict[str, list[torch.Tensor]] = {}
        for cache in cached_list:
            for k, v in cache.items():
                if k not in embs_list_dict:
                    embs_list_dict[k] = []
                embs_list_dict[k].append(v)
        embs_dict: dict[str, torch.Tensor] = {
            k: torch.stack(v).to(inputs.text_input_ids.device) for k, v in embs_list_dict.items()
        }
        return self.pool_embs(
            embs_dict["text_embs"], embs_dict.get("audio_embs", None), embs_dict.get("video_embs", None)
        )

    def compute_embs(self, inputs: MultimodalInput):
        assert isinstance(inputs.unique_id, list), "unique_id must be a list"

        text_outputs = self.text_backbone(inputs.text_input_ids, attention_mask=inputs.text_attention_mask)
        text_embs = text_outputs.last_hidden_state[:, 0]

        if inputs.audio_input_values is not None and self.audio_backbone is not None:
            audio_outputs = self.audio_backbone(inputs.audio_input_values, attention_mask=inputs.audio_attention_mask)
            audio_embs = audio_outputs.last_hidden_state[:, 0]
        else:
            audio_embs = None

        if inputs.video_pixel_values is not None and self.video_backbone is not None:
            video_outputs = self.video_backbone(inputs.video_pixel_values)
            video_embs = video_outputs.last_hidden_state[:, 0]
        else:
            video_embs = None

        return text_embs, audio_embs, video_embs

    def compute_pooled_embs(self, inputs: MultimodalInput):
        assert isinstance(inputs.unique_id, list), "unique_id must be a list"

        text_outputs = self.text_backbone(inputs.text_input_ids, attention_mask=inputs.text_attention_mask)
        text_embs = text_outputs.last_hidden_state[:, 0]
        text_pooled_embs = self.text_pooler(text_embs)

        if inputs.audio_input_values is not None and self.audio_backbone is not None:
            audio_outputs = self.audio_backbone(inputs.audio_input_values, attention_mask=inputs.audio_attention_mask)
            audio_embs = audio_outputs.last_hidden_state[:, 0]
            audio_pooled_embs = self.audio_pooler(audio_embs)
        else:
            audio_embs = None
            audio_pooled_embs = None

        if inputs.video_pixel_values is not None and self.video_backbone is not None:
            video_outputs = self.video_backbone(inputs.video_pixel_values)
            video_embs = video_outputs.last_hidden_state[:, 0]
            video_pooled_embs = self.video_pooler(video_embs)
        else:
            video_embs = None
            video_pooled_embs = None

        return text_pooled_embs, audio_pooled_embs, video_pooled_embs

    def mean_embs(self, embs_list: list[torch.Tensor | None]):
        filtered_embs_list = [emb for emb in embs_list if emb is not None]
        return sum(filtered_embs_list) / len(filtered_embs_list)

    # def forward(self, inputs: MultimodalInput):
    #     text_pooled_embs, audio_pooled_embs, video_pooled_embs = self.compute_pooled_embs(inputs)
    #     return self.mean_embs([text_pooled_embs, audio_pooled_embs, video_pooled_embs])

    def forward(self, inputs: MultimodalInput):
        if self.is_frozen:
            text_pooled_embs, audio_pooled_embs, video_pooled_embs = self.cached_compute_pooled_embs(inputs)
        else:
            text_pooled_embs, audio_pooled_embs, video_pooled_embs = self.compute_pooled_embs(inputs)
        fusion_tensor = self.tensor_fusion_layer(text_pooled_embs, audio_pooled_embs, video_pooled_embs)
        return fusion_tensor

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
