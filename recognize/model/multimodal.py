from __future__ import annotations

from typing import Callable

import torch
from loguru import logger
from pydantic import Field
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence

from ..cache import load_cached_tensors, save_cached_tensors
from ..dataset import Preprocessor
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

    @property
    def device(self):
        if self.text_input_ids is not None:
            return self.text_input_ids.device
        elif self.audio_input_values is not None:
            return self.audio_input_values.device
        elif self.video_pixel_values is not None:
            return self.video_pixel_values.device
        else:
            return torch.device("cpu")

    @classmethod
    def collate_fn(cls, batch: list[MultimodalInput]):
        field_dict = {}
        for field, field_info in cls.model_fields.items():
            if field_info.annotation == torch.Tensor:
                attr = [getattr(item, field) for item in batch]
            elif field_info.annotation == torch.Tensor | None:
                attr = cls.merge(batch, field)
            elif field_info.annotation == str | list[str]:
                attr = [getattr(item, field) for item in batch]
            else:
                logger.debug(f"Field [red]{field}[/] will use first item in batch")
                attr = getattr(batch[0], field)
            if field == "labels":
                attr = torch.tensor(attr, dtype=torch.int64)
            elif isinstance(attr, torch.Tensor):
                attr = pad_sequence(attr, batch_first=True)
            elif isinstance(attr, list) and isinstance(attr[0], torch.Tensor):
                attr = pad_sequence(attr, batch_first=True)
            field_dict[field] = attr
        return cls(**field_dict)


class LazyMultimodalInput(ModelInput):
    preprocessor: Preprocessor

    texts: list[str] | None = Field(default_factory=list)
    audio_paths: list[str] | None = Field(default_factory=list)
    video_paths: list[str] | None = Field(default_factory=list)

    labels: torch.Tensor | None = None
    unique_id: list[str] = Field(default_factory=list)

    # The following fields will be computed lazily
    text_input_ids: torch.Tensor | None = None
    text_attention_mask: torch.Tensor | None = None
    audio_input_values: torch.Tensor | None = None
    audio_attention_mask: torch.Tensor | None = None
    video_pixel_values: torch.Tensor | None = None
    video_head_mask: torch.Tensor | None = None

    def __getattribute__(self, __name: str):
        if (
            __name in ["text_input_ids", "text_attention_mask"]
            and super().__getattribute__("text_input_ids") is None
            and self.texts is not None
        ):
            self.text_input_ids, self.text_attention_mask = self.preprocessor.load_texts(self.texts)
        elif (
            __name in ["audio_input_values", "audio_attention_mask"]
            and super().__getattribute__("audio_input_values") is None
            and self.audio_paths is not None
        ):
            self.audio_input_values, self.audio_attention_mask = self.preprocessor.load_audios(self.audio_paths)
        elif (
            __name == "video_pixel_values"
            and super().__getattribute__("video_pixel_values") is None
            and self.video_paths is not None
        ):
            self.video_pixel_values = self.preprocessor.load_videos(self.video_paths)

        return super().__getattribute__(__name)

    def __getitem__(self, index: int | list[int] | slice) -> LazyMultimodalInput:
        assert isinstance(self.unique_id, list), "unique_id must be a list"
        if isinstance(index, slice):
            texts = self.texts[index] if self.texts is not None else None
            audio_paths = self.audio_paths[index] if self.audio_paths is not None else None
            video_paths = self.video_paths[index] if self.video_paths is not None else None
            labels = self.labels[index] if self.labels is not None else None
            unique_id = self.unique_id[index]
        else:
            if isinstance(index, int):
                index = [index]
            texts = [self.texts[i] for i in index] if self.texts is not None else None
            audio_paths = [self.audio_paths[i] for i in index] if self.audio_paths is not None else None
            video_paths = [self.video_paths[i] for i in index] if self.video_paths is not None else None
            labels = self.labels[index] if self.labels is not None else None
            unique_id = [self.unique_id[i] for i in index]

        text_input_ids = self.text_input_ids[index] if self.text_input_ids is not None else None
        text_attention_mask = self.text_attention_mask[index] if self.text_attention_mask is not None else None
        audio_input_values = self.audio_input_values[index] if self.audio_input_values is not None else None
        audio_attention_mask = self.audio_attention_mask[index] if self.audio_attention_mask is not None else None
        video_pixel_values = self.video_pixel_values[index] if self.video_pixel_values is not None else None
        video_head_mask = self.video_head_mask[index] if self.video_head_mask is not None else None

        return LazyMultimodalInput(
            preprocessor=self.preprocessor,
            texts=texts,
            audio_paths=audio_paths,
            video_paths=video_paths,
            labels=labels,
            unique_id=unique_id,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            audio_input_values=audio_input_values,
            audio_attention_mask=audio_attention_mask,
            video_pixel_values=video_pixel_values,
            video_head_mask=video_head_mask,
        )

    @property
    def device(self):
        if self.text_input_ids is not None:
            return self.text_input_ids.device
        elif self.audio_input_values is not None:
            return self.audio_input_values.device
        elif self.video_pixel_values is not None:
            return self.video_pixel_values.device
        else:
            return torch.device("cpu")

    @classmethod
    def collate_fn(cls, batch: list[LazyMultimodalInput]):
        field_dict = {}
        for field in cls.model_fields.keys():
            if field in [
                "text_input_ids",
                "text_attention_mask",
                "audio_input_values",
                "audio_attention_mask",
                "video_pixel_values",
                "video_head_mask",
            ]:
                continue
            if field == "labels":
                attr = cls.merge(batch, field)
                if attr is not None:
                    attr = torch.cat(attr)
            elif field == "preprocessor":
                attr = batch[0].preprocessor
            else:
                attr = []
                for item in batch:
                    itme_attr = getattr(item, field)
                    if itme_attr is None:
                        break
                    assert isinstance(itme_attr, list), f"{field} must be a list"
                    attr.extend(itme_attr)
            field_dict[field] = attr
        return cls(**field_dict)


class TensorFusionLayer(nn.Module):
    def __init__(self, text_size: int, audio_size: int, video_size: int):
        super().__init__()
        self.text_size = text_size
        self.audio_size = audio_size
        self.video_size = video_size
        self.augmentation = Parameter(torch.ones(1))
        self.output_size = (text_size + 1) * (audio_size + 1) * (video_size + 1)

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

        assert augmentation_zl.size(1) == self.text_size + 1, f"{augmentation_zl.size(1)} != {self.text_size + 1}"
        assert augmentation_za.size(1) == self.audio_size + 1, f"{augmentation_za.size(1)} != {self.audio_size + 1}"
        assert augmentation_zv.size(1) == self.video_size + 1, f"{augmentation_zv.size(1)} != {self.video_size + 1}"

        fusion_tensor = torch.einsum("bi,bj,bk->bijk", augmentation_zl, augmentation_za, augmentation_zv).view(
            zl.size(0), -1
        )
        return fusion_tensor


class LowRankFusionLayer(nn.Module):
    def __init__(self, dims: list[int], rank: int, output_size: int):
        super().__init__()
        self.dims = dims
        self.rank = rank
        self.output_size = output_size

        self.low_rank_weights = nn.ParameterList([nn.Parameter(torch.randn(rank, dim, output_size)) for dim in dims])
        self.output_layer = nn.Linear(rank, 1)

    def forward(self, *inputs: torch.Tensor | None):
        assert len(inputs) <= len(
            self.low_rank_weights
        ), "Number of inputs should be less than or equal to number of weights"
        # N*d x R*d*h => R*N*h ~reshape~> N*h*R -> N*h*1 ~squeeze~> N*h
        fusion_tensors = [
            torch.matmul(i, w) for i, w in zip(inputs, self.low_rank_weights, strict=False) if i is not None
        ]
        product_tensor = torch.prod(torch.stack(fusion_tensors), dim=0)
        output = self.output_layer(product_tensor.permute(1, 2, 0)).squeeze(dim=-1)
        return output


class MultimodalBackbone(Backbone):
    def __init__(
        self,
        text_backbone: nn.Module,
        audio_backbone: nn.Module | None = None,
        video_backbone: nn.Module | None = None,
        *,
        use_cache: bool = True,
    ):
        assert (audio_backbone is None or text_backbone.config.hidden_size == audio_backbone.config.hidden_size) and (
            video_backbone is None or text_backbone.config.hidden_size == video_backbone.config.hidden_size
        ), "Hidden size of text, audio and video backbones must be the same"
        super().__init__(text_backbone.config.hidden_size)
        self.text_backbone = self.pretrained_module(text_backbone)
        self.audio_backbone = self.pretrained_module(audio_backbone)
        self.video_backbone = self.pretrained_module(video_backbone)

        self.use_cache = use_cache

    def compute_embs(
        self, inputs: LazyMultimodalInput
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
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

    def forward(self, inputs: LazyMultimodalInput):
        if self.is_frozen and self.use_cache:
            return self.cached_forward(inputs)
        else:
            return self.compute_embs(inputs)

    def cached_forward(self, inputs: LazyMultimodalInput):
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
        embs_dict: dict[str, torch.Tensor] = {k: torch.stack(v).to(inputs.device) for k, v in embs_list_dict.items()}

        return embs_dict["text_embs"], embs_dict.get("audio_embs", None), embs_dict.get("video_embs", None)

    def mean_embs(self, embs_list: list[torch.Tensor | None]):
        filtered_embs_list = [emb for emb in embs_list if emb is not None]
        return sum(filtered_embs_list) / len(filtered_embs_list)

    # def forward(self, inputs: MultimodalInput):
    #     text_pooled_embs, audio_pooled_embs, video_pooled_embs = self.compute_pooled_embs(inputs)
    #     return self.mean_embs([text_pooled_embs, audio_pooled_embs, video_pooled_embs])

    # def forward(self, inputs: MultimodalInput):
    #     if self.is_frozen:
    #         text_pooled_embs, audio_pooled_embs, video_pooled_embs = self.cached_compute_pooled_embs(inputs)
    #     else:
    #         text_pooled_embs, audio_pooled_embs, video_pooled_embs = self.compute_pooled_embs(inputs)
    #     fusion_tensor = self.tensor_fusion_layer(text_pooled_embs, audio_pooled_embs, video_pooled_embs)
    #     return fusion_tensor


class MultimodalModel(ClassifierModel):
    def __init__(
        self,
        text_backbone: nn.Module,
        audio_backbone: nn.Module | None = None,
        video_backbone: nn.Module | None = None,
        *,
        text_feature_size: int = 128,
        audio_feature_size: int = 16,
        video_feature_size: int = 1,
        num_classes: int = 2,
        class_weights: torch.Tensor | None = None,
    ):
        # fusion_layer = TensorFusionLayer(text_feature_size, audio_feature_size, video_feature_size)
        fusion_layer = LowRankFusionLayer([text_feature_size, audio_feature_size, video_feature_size], 16, 128)
        super().__init__(
            MultimodalBackbone(text_backbone, audio_backbone, video_backbone),
            fusion_layer.output_size,
            num_classes=num_classes,
            class_weights=class_weights,
        )
        self.poolers = nn.ModuleList(
            [
                Pooler(self.backbone.output_size, text_feature_size),
                Pooler(self.backbone.output_size, audio_feature_size),
                Pooler(self.backbone.output_size, video_feature_size),
            ]
        )
        self.fusion_layer = fusion_layer

    def pool_embs(
        self, text_embs: torch.Tensor, audio_embs: torch.Tensor | None, video_embs: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        text_pooler, audio_pooler, video_pooler = self.poolers

        text_pooled_embs = text_pooler(text_embs)
        if audio_embs is not None and text_embs.size(0) == audio_embs.size(0):
            audio_pooled_embs = audio_pooler(audio_embs)
        else:
            audio_pooled_embs = None

        if video_embs is not None and text_embs.size(0) == video_embs.size(0):
            video_pooled_embs = video_pooler(video_embs)
        else:
            video_pooled_embs = None

        return text_pooled_embs, audio_pooled_embs, video_pooled_embs

    def forward(self, inputs: LazyMultimodalInput) -> ClassifierOutput:
        text_embs, audio_embs, video_embs = self.backbone(inputs)
        text_pooled_embs, audio_pooled_embs, video_pooled_embs = self.pool_embs(text_embs, audio_embs, video_embs)
        fusion_features = self.fusion_layer(text_pooled_embs, audio_pooled_embs, video_pooled_embs)
        return self.classify(fusion_features, inputs.labels)

    __call__: Callable[[LazyMultimodalInput], ClassifierOutput]
