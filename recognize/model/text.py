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

from ..cache import load_cached_tensors, save_cached_tensors
from ..dataset import Preprocessor
from .base import Backbone, ClassifierModel, ClassifierOutput, ModelInput, Pooler


class TextInput(ModelInput):
    text_input_ids: torch.Tensor
    text_attention_mask: torch.Tensor | None = None

    labels: torch.Tensor | None = None
    unique_ids: list[str] | None = None

    def __getitem__(self, index: int | list[int] | slice) -> TextInput:
        text_input_ids = self.text_input_ids[index]
        text_attention_mask = (
            self.text_attention_mask[index] if self.text_attention_mask is not None else None
        )
        labels = self.labels[index] if self.labels is not None else None

        if self.unique_ids is None:
            unique_ids = None
        else:
            if isinstance(index, list):
                unique_ids = [self.unique_ids[i] for i in index]
            elif isinstance(index, int):
                unique_ids = [self.unique_ids[index]]
            elif isinstance(index, slice):
                unique_ids = self.unique_ids[index]
            else:
                raise ValueError(f"Unsupported index type {type(index)}")

        return TextInput(
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            labels=labels,
            unique_ids=unique_ids,
        )

    @property
    def device(self):
        if self.text_input_ids is not None:
            return self.text_input_ids.device
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


class LazyTextInput(TextInput):
    preprocessor: Preprocessor
    texts: list[str]

    def __getattribute__(self, __name: str):
        if (
            __name in ["text_input_ids", "text_attention_mask"]
            and super().__getattribute__("text_input_ids") is None
        ):
            self.text_input_ids, self.text_attention_mask = self.preprocessor.load_texts(self.texts)  # type: ignore

        return super().__getattribute__(__name)

    @classmethod
    def collate_fn(cls, batch: Sequence[Self]) -> Self:
        field_dict = {}
        for field in cls.model_fields.keys():
            if field in [
                "text_input_ids",
                "text_attention_mask",
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


class TextBackbone(Backbone):
    def __init__(
        self,
        text_backbone: nn.Module,
        *,
        use_cache: bool = True,
        is_frozen: bool = True,
        use_peft: bool = False,
        backbones_dir: str | Path = "./checkpoints/backbones",
    ):
        super().__init__(
            [text_backbone],
            use_cache=use_cache,
            is_frozen=is_frozen,
            use_peft=use_peft,
            backbones_dir=backbones_dir,
        )
        self.text_backbone = self.pretrained_module(text_backbone)

    def compute_embs(self, inputs: TextInput) -> torch.Tensor:
        text_outputs = self.text_backbone(
            inputs.text_input_ids, attention_mask=inputs.text_attention_mask
        )
        text_embs = text_outputs.last_hidden_state[:, 0]
        return text_embs

    def forward(self, inputs: TextInput):
        if self.is_frozen and self.use_cache and inputs.unique_ids is not None:
            return self.cached_forward(inputs)
        else:
            return self.compute_embs(inputs)

    def cached_forward(self, inputs: TextInput):
        assert isinstance(inputs.unique_ids, list), "unique_ids must be a list"
        cached_list, cached_index_list, no_cached_index_list = load_cached_tensors(
            inputs.unique_ids
        )
        if len(no_cached_index_list) != 0:
            no_cached_inputs = inputs[no_cached_index_list]
            assert isinstance(no_cached_inputs.unique_ids, list), "unique_ids must be a list"
            with torch.no_grad():
                text_embs, audio_embs, video_embs = self.compute_embs(no_cached_inputs)

            if self.is_frozen:
                save_cached_tensors(
                    no_cached_inputs.unique_ids,
                    {
                        "text_embs": text_embs,
                    },
                )

            cached_list, cached_index_list, no_cached_index_list = load_cached_tensors(
                inputs.unique_ids
            )
        assert len(no_cached_index_list) == 0, "All tensors should be cached"

        embs_list_dict: dict[str, list[torch.Tensor]] = {}
        for cache in cached_list:
            for k, v in cache.items():
                if k not in embs_list_dict:
                    embs_list_dict[k] = []
                embs_list_dict[k].append(v)
        embs_dict: dict[str, torch.Tensor] = {
            k: torch.stack(v).to(inputs.device) for k, v in embs_list_dict.items()
        }
        return embs_dict.get("text_embs", None)

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
        for name in ["text_backbone"]:
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


class TextModel(ClassifierModel):
    def __init__(
        self,
        backbone: TextBackbone,
        *,
        text_feature_size: int = 128,
        num_classes: int = 2,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__(
            backbone,
            text_feature_size,
            num_classes=num_classes,
            class_weights=class_weights,
        )
        self.text_feature_size = text_feature_size

        self.text_pooler = Pooler(self.backbone.output_size, text_feature_size)

    def get_hyperparameter(self):
        return {
            "text_feature_size": self.text_feature_size,
            "num_classes": self.num_classes,
        }

    def pool_embs(
        self,
        text_embs: torch.Tensor,
    ) -> torch.Tensor:
        return self.text_pooler(text_embs)

    def forward(self, inputs: TextInput) -> ClassifierOutput:
        text_embs = self.backbone(inputs)
        text_pooled_embs, audio_pooled_embs, video_pooled_embs = self.pool_embs(text_embs)
        fusion_features = self.fusion_layer(text_pooled_embs, audio_pooled_embs, video_pooled_embs)
        return self.classify(fusion_features, inputs.labels)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        backbone: TextBackbone,
        *,
        class_weights: torch.Tensor | None = None,
    ):
        checkpoint_path = Path(checkpoint_path)
        with open(checkpoint_path / "config.json", "r") as f:
            model_config = json.load(f)
        return cls(backbone, **model_config, class_weights=class_weights)

    __call__: Callable[[TextInput], ClassifierOutput]
