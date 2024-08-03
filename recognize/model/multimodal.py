from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Self, Sequence

import torch
from loguru import logger
from safetensors.torch import load_file
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModel

from ..dataset import Preprocessor
from ..module.fusion import FusionLayer
from .base import Backbone, ClassifierModel, ClassifierOutput, ModelInput, Pooler


class MultimodalInput(ModelInput):
    text_input_ids: torch.Tensor | None = None
    text_attention_mask: torch.Tensor | None = None

    audio_input_values: torch.Tensor | None = None
    audio_attention_mask: torch.Tensor | None = None

    video_pixel_values: torch.Tensor | None = None
    video_head_mask: torch.Tensor | None = None

    labels: torch.Tensor | None = None

    def __getitem__(self, index: int | list[int] | slice) -> MultimodalInput:
        text_input_ids = self.text_input_ids[index] if self.text_input_ids is not None else None
        text_attention_mask = (
            self.text_attention_mask[index] if self.text_attention_mask is not None else None
        )
        audio_input_values = (
            self.audio_input_values[index] if self.audio_input_values is not None else None
        )
        audio_attention_mask = (
            self.audio_attention_mask[index] if self.audio_attention_mask is not None else None
        )
        video_pixel_values = (
            self.video_pixel_values[index] if self.video_pixel_values is not None else None
        )
        video_head_mask = self.video_head_mask[index] if self.video_head_mask is not None else None

        labels = self.labels[index] if self.labels is not None else None

        if self.unique_ids is None:
            unique_ids = None
        elif isinstance(index, list):
            unique_ids = [self.unique_ids[i] for i in index]
        elif isinstance(index, int):
            unique_ids = [self.unique_ids[index]]
        elif isinstance(index, slice):
            unique_ids = self.unique_ids[index]

        return MultimodalInput(
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            audio_input_values=audio_input_values,
            audio_attention_mask=audio_attention_mask,
            video_pixel_values=video_pixel_values,
            video_head_mask=video_head_mask,
            labels=labels,
            unique_ids=unique_ids,
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
            logger.warning("No tensor is loaded, use CPU as default")
            return torch.device("cpu")

    @classmethod
    def collate_fn(cls, batch: Sequence[Self]) -> Self:
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


class LazyMultimodalInput(MultimodalInput):
    preprocessor: Preprocessor

    texts: list[str] | None = None
    audio_paths: list[str] | None = None
    video_paths: list[str] | None = None

    def __getattribute__(self, __name: str):
        if (
            __name in ["text_input_ids", "text_attention_mask"]
            and self.texts is None
            and self.audio_paths is not None
        ):
            logger.debug("Texts is not provided, but audio_paths is provided, so use audio_paths")
            self.texts = [self.recoginize_audio(audio_path) for audio_path in self.audio_paths]
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
            self.audio_input_values, self.audio_attention_mask = self.preprocessor.load_audios(
                self.audio_paths
            )
        elif (
            __name == "video_pixel_values"
            and super().__getattribute__("video_pixel_values") is None
            and self.video_paths is not None
        ):
            pass
            # TODO: load video too slow
            self.video_pixel_values = self.preprocessor.load_videos(self.video_paths)

        return super().__getattribute__(__name)

    @classmethod
    def collate_fn(cls, batch: Sequence[Self]) -> Self:
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
                # those field is lazy
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
                        attr = None
                        break
                    assert isinstance(
                        itme_attr, list
                    ), f"{field} must be a list, but got {type(itme_attr)}"
                    attr.extend(itme_attr)
            field_dict[field] = attr
        return cls(**field_dict)


class MultimodalBackbone(Backbone[MultimodalInput]):
    def __init__(
        self,
        text_backbone: nn.Module | None = None,
        audio_backbone: nn.Module | None = None,
        video_backbone: nn.Module | None = None,
        *,
        use_cache: bool = True,
        is_frozen: bool = True,
        use_peft: bool = False,
        backbones_dir: str | Path = "./checkpoints/backbones",
    ):
        super().__init__(
            {
                "text_backbone": text_backbone,
                "audio_backbone": audio_backbone,
                "video_backbone": video_backbone,
            },
            embedding_num=3,
            use_cache=use_cache,
            is_frozen=is_frozen,
            use_peft=use_peft,
            backbones_dir=backbones_dir,
        )

    def compute_embs(
        self, inputs: MultimodalInput
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if "text_backbone" in self.backbones and inputs.text_input_ids is not None:
            text_outputs = self.backbones["text_backbone"](
                inputs.text_input_ids, attention_mask=inputs.text_attention_mask
            )
            text_embs = text_outputs.last_hidden_state[:, 0]
        else:
            text_embs = None

        if "audio_backbone" in self.backbones and inputs.audio_input_values is not None:
            audio_outputs = self.backbones["audio_backbone"](
                inputs.audio_input_values, attention_mask=inputs.audio_attention_mask
            )
            audio_embs = audio_outputs.last_hidden_state[:, 0]
        else:
            audio_embs = None

        if "video_backbone" in self.backbones and inputs.video_pixel_values is not None:
            video_outputs = self.backbones["video_backbone"](inputs.video_pixel_values)
            video_embs = video_outputs.last_hidden_state[:, 0]
        else:
            video_embs = None

        return text_embs, audio_embs, video_embs

    def forward(self, inputs: MultimodalInput):
        if self.is_frozen and self.use_cache and inputs.unique_ids is not None:
            return self.cached_forward(inputs)
        else:
            return self.compute_embs(inputs)

    def cached_forward(self, inputs: MultimodalInput) -> tuple[torch.Tensor | None, ...]:
        cached_list, no_cached_index_list = self.load_cache(inputs)
        if len(no_cached_index_list) != 0:
            no_cached_inputs = inputs[no_cached_index_list]
            self.save_cache(no_cached_inputs)
            cached_list, no_cached_index_list = self.load_cache(inputs)
        assert len(no_cached_index_list) == 0, "All tensors should be cached"

        return self.merge_cache(cached_list, inputs.device)

    # def mean_embs(self, embs_list: list[torch.Tensor | None]):
    #     filtered_embs_list = [emb for emb in embs_list if emb is not None]
    #     return sum(filtered_embs_list) / len(filtered_embs_list)

    # def forward(self, inputs: MultimodalInput):
    #     text_pooled_embs, audio_pooled_embs, video_pooled_embs = self.compute_pooled_embs(inputs)
    #     return self.mean_embs([text_pooled_embs, audio_pooled_embs, video_pooled_embs])

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        use_cache: bool = True,
        is_frozen: bool = True,
        use_peft: bool = False,
        backbones_dir: str | Path = "./checkpoints/backbones",
    ):
        checkpoint_path = Path(checkpoint_path)
        backbones: list[nn.Module] = []
        for name in ["text_backbone", "audio_backbone", "video_backbone"]:
            if not (checkpoint_path / name).exists():
                logger.warning(f"{name} not found in {checkpoint_path}")
                continue
            config = AutoConfig.from_pretrained(checkpoint_path / name)
            model = AutoModel.from_config(config)
            model.load_state_dict(load_file(checkpoint_path / f"{name}.safetensors"))
            backbones.append(model)

        return cls(
            *backbones,
            use_cache=use_cache,
            is_frozen=is_frozen,
            use_peft=use_peft,
            backbones_dir=backbones_dir,
        )


class MultimodalModel(ClassifierModel[MultimodalBackbone]):
    def __init__(
        self,
        backbone: MultimodalBackbone,
        fusion_layer: FusionLayer,
        feature_sizes: Sequence[int],
        *,
        num_classes: int = 2,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__(
            backbone,
            fusion_layer.output_size,
            num_classes=num_classes,
            class_weights=class_weights,
        )
        self.feature_size = list(feature_sizes)

        self.poolers = nn.ModuleList(
            [Pooler(self.backbone.output_size, feature_size) for feature_size in feature_sizes]
        )
        self.fusion_layer = fusion_layer

    def get_hyperparameter(self):
        return {
            "feature_size": self.feature_size,
            "num_classes": self.num_classes,
        }

    def pool_embs(
        self,
        text_embs: torch.Tensor | None,
        audio_embs: torch.Tensor | None,
        video_embs: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        embs_list = [text_embs, audio_embs, video_embs]
        pool_embs_list = [
            pooler(embs) if embs is not None else None
            for pooler, embs in zip(self.poolers, embs_list, strict=False)
        ]
        return tuple(pool_embs_list)

    def forward(self, inputs: MultimodalInput) -> ClassifierOutput:
        text_embs, audio_embs, video_embs = self.backbone(inputs)
        pooled_embs_tuple = self.pool_embs(text_embs, audio_embs, video_embs)
        fusion_features = self.fusion_layer(*pooled_embs_tuple)
        return self.classify(fusion_features, inputs.labels)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        backbone: MultimodalBackbone,
        fusion_layer: FusionLayer,
        *,
        class_weights: torch.Tensor | None = None,
    ):
        checkpoint_path = Path(checkpoint_path)
        with open(checkpoint_path / "config.json", "r") as f:
            model_config = json.load(f)
        return cls(backbone, fusion_layer, **model_config, class_weights=class_weights)

    __call__: Callable[[MultimodalInput], ClassifierOutput]
