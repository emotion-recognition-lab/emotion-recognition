from __future__ import annotations

import pickle
from collections.abc import Sequence
from typing import Self

import torch
from loguru import logger
from pydantic import BaseModel, ConfigDict
from torch.nn.utils.rnn import pad_sequence

from recognize.cache import hash_bytes
from recognize.preprocessor import Preprocessor
from recognize.typing import ModelInputT


class ModelInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    labels: torch.Tensor | None = None

    def cuda(self) -> Self:
        for field in self.model_fields.keys():
            field_value = getattr(self, field)
            if isinstance(field_value, torch.Tensor):
                setattr(self, field, field_value.cuda())
        return self

    def pin_memory(self) -> Self:
        for field in self.model_fields.keys():
            field_value = getattr(self, field)
            if isinstance(field_value, torch.Tensor):
                setattr(self, field, field_value.pin_memory().cuda())
        return self

    def hash(self) -> str:
        return hash_bytes(pickle.dumps(self.model_dump(mode="json")))

    def __hash__(self) -> int:
        return int(self.hash(), 16)

    def __getitem__(self, index: int | list[int] | slice) -> Self:
        raise NotImplementedError("__getitem__ method must be implemented in subclass")

    def get_unique_keys(self) -> dict[str, list[str]]:
        raise NotImplementedError("You should use LazyMultimodalInput instead")

    @staticmethod
    def merge(batch: Sequence[ModelInputT], attr_name: str):
        attr: list[torch.Tensor] = []
        for item in batch:
            if getattr(item, attr_name) is not None:
                attr.append(getattr(item, attr_name))
            else:
                return None
        return attr

    @property
    def device(self) -> torch.device:
        raise NotImplementedError("device property must be implemented in subclass")


class MultimodalInput(ModelInput):
    text_input_ids: torch.Tensor | None = None
    text_attention_mask: torch.Tensor | None = None

    audio_input_values: torch.Tensor | None = None
    audio_attention_mask: torch.Tensor | None = None

    video_pixel_values: torch.Tensor | None = None
    video_head_mask: torch.Tensor | None = None

    def __getitem__(self, index: int | list[int] | slice) -> MultimodalInput:
        labels = self.labels[index] if self.labels is not None else None
        if isinstance(self, LazyMultimodalInput):
            assert not isinstance(index, list), "LazyMultimodalInput does not support list index"
            if isinstance(index, int):
                texts = [self.texts[index]] if self.texts is not None else None
                audio_paths = [self.audio_paths[index]] if self.audio_paths is not None else None
                video_paths = [self.video_paths[index]] if self.video_paths is not None else None
            else:
                texts = self.texts[index] if self.texts is not None else None
                audio_paths = self.audio_paths[index] if self.audio_paths is not None else None
                video_paths = self.video_paths[index] if self.video_paths is not None else None
            return LazyMultimodalInput(
                texts=texts,
                audio_paths=audio_paths,
                video_paths=video_paths,
                preprocessor=self.preprocessor,
                labels=labels,
            )

        else:
            text_input_ids = self.text_input_ids[index] if self.text_input_ids is not None else None
            text_attention_mask = self.text_attention_mask[index] if self.text_attention_mask is not None else None
            audio_input_values = self.audio_input_values[index] if self.audio_input_values is not None else None
            audio_attention_mask = self.audio_attention_mask[index] if self.audio_attention_mask is not None else None
            video_pixel_values = self.video_pixel_values[index] if self.video_pixel_values is not None else None
            video_head_mask = self.video_head_mask[index] if self.video_head_mask is not None else None

            return MultimodalInput(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                audio_input_values=audio_input_values,
                audio_attention_mask=audio_attention_mask,
                video_pixel_values=video_pixel_values,
                video_head_mask=video_head_mask,
                labels=labels,
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

    _cuda: bool = False
    _pin_memory: bool = False
    # TODO: add model_validator

    # TODO: support cuda and pin_memory
    # 如果不处理logit就变成了不是cuda
    def cuda(self) -> Self:
        self._cuda = True
        if self.labels is not None:
            self.labels = self.labels.cuda()
        return self

    def pin_memory(self) -> Self:
        self._pin_memory = True
        self._cuda = True
        if self.labels is not None:
            self.labels = self.labels.pin_memory().cuda()
        return self

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if self._cuda else "cpu")

    def __getattribute__(self, __name: str):
        match __name:
            case "text_input_ids" | "text_attention_mask":
                if self.texts is None and self.audio_paths is not None:
                    logger.debug("Texts is not provided, but audio_paths is provided, so use audio_paths")
                    self.texts = [self.preprocessor.recoginize_audio(audio_path) for audio_path in self.audio_paths]
                if super().__getattribute__("text_input_ids") is None and self.texts is not None:
                    self.text_input_ids, self.text_attention_mask = self.preprocessor.load_texts(
                        self.texts, device=self.device
                    )
            case "audio_input_values" | "audio_attention_mask":
                if super().__getattribute__("audio_input_values") is None and self.audio_paths is not None:
                    self.audio_input_values, self.audio_attention_mask = self.preprocessor.load_audios(
                        self.audio_paths, device=self.device
                    )
            case "video_pixel_values":
                if super().__getattribute__("video_pixel_values") is None and self.video_paths is not None:
                    self.video_pixel_values = self.preprocessor.load_videos(self.video_paths, device=self.device)
        return super().__getattribute__(__name)

    def get_unique_keys(self) -> dict[str, list[str]]:
        input_metas = {
            "T": self.texts,
            "A": self.audio_paths,
            "V": self.video_paths,
        }
        unique_keys = {
            modal: [hash_bytes(pickle.dumps((modal, v_))) for v_ in metas]
            for modal, metas in input_metas.items()
            if metas is not None
        }
        return unique_keys

    def hash(self) -> str:
        return hash_bytes(pickle.dumps((self.texts, self.audio_paths, self.video_paths)))

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
