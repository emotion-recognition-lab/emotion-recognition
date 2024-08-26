from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Self

import torch
from loguru import logger
from torch.nn.utils.rnn import pad_sequence

from recognize.config import load_inference_config
from recognize.module.fusion import FusionLayer
from recognize.preprocessor import Preprocessor

from .base import Backbone, ClassifierModel, ClassifierOutput, ModelInput


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
        text_attention_mask = self.text_attention_mask[index] if self.text_attention_mask is not None else None
        audio_input_values = self.audio_input_values[index] if self.audio_input_values is not None else None
        audio_attention_mask = self.audio_attention_mask[index] if self.audio_attention_mask is not None else None
        video_pixel_values = self.video_pixel_values[index] if self.video_pixel_values is not None else None
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
        if __name in ["text_input_ids", "text_attention_mask"] and self.texts is None and self.audio_paths is not None:
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
            self.audio_input_values, self.audio_attention_mask = self.preprocessor.load_audios(self.audio_paths)
        elif (
            __name == "video_pixel_values"
            and super().__getattribute__("video_pixel_values") is None
            and self.video_paths is not None
        ):
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
                    assert isinstance(itme_attr, list), f"{field} must be a list, but got {type(itme_attr)}"
                    attr.extend(itme_attr)
            field_dict[field] = attr
        return cls(**field_dict)


class MultimodalBackbone(Backbone[MultimodalInput]):
    def compute_embs(self, inputs: MultimodalInput) -> dict[str, torch.Tensor | None]:
        if "text" in self.encoders and inputs.text_input_ids is not None:
            text_outputs = self.encoders["text"](inputs.text_input_ids, attention_mask=inputs.text_attention_mask)
            text_embs = text_outputs.last_hidden_state[:, 0]
        else:
            text_embs = None

        if "audio" in self.encoders and inputs.audio_input_values is not None:
            audio_outputs = self.encoders["audio"](
                inputs.audio_input_values, attention_mask=inputs.audio_attention_mask
            )
            audio_embs = audio_outputs.last_hidden_state[:, 0]
        else:
            audio_embs = None

        if "video" in self.encoders and inputs.video_pixel_values is not None:
            video_outputs = self.encoders["video"](inputs.video_pixel_values)
            video_embs = video_outputs.last_hidden_state[:, 0]
        else:
            video_embs = None

        return {
            "text": text_embs,
            "audio": audio_embs,
            "video": video_embs,
        }


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
        self.feature_sizes = list(feature_sizes)
        self.fusion_layer = fusion_layer

    def get_hyperparameter(self):
        return {
            "num_classes": self.num_classes,
        }

    def forward(self, inputs: MultimodalInput) -> ClassifierOutput:
        embs_dict = self.backbone(inputs)
        fusion_features = self.fusion_layer(embs_dict)
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
        config = load_inference_config(checkpoint_path / "inference.toml")
        return cls(
            backbone,
            fusion_layer,
            feature_sizes=config.model.feature_sizes,
            num_classes=config.num_classes,
            class_weights=class_weights,
        )

    __call__: Callable[[MultimodalInput], ClassifierOutput]
