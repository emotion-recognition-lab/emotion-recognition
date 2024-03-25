from __future__ import annotations

from functools import cached_property
from typing import Callable, TypeVar, overload

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel

ModelInputT = TypeVar("ModelInputT", bound="ModelInput")
ClassifierModelT = TypeVar("ClassifierModelT", bound="ClassifierModel")


class ModelOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ClassifierOutput(ModelOutput):
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class ModelInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def cuda(self):
        for field in self.model_fields_set:
            field_value = getattr(self, field)
            if isinstance(field_value, torch.Tensor):
                setattr(self, field, field_value.cuda())
        return self

    def pin_memory(self):
        for field in self.model_fields_set:
            field_value = getattr(self, field)
            if isinstance(field_value, torch.Tensor):
                setattr(self, field, field_value.pin_memory().cuda())
        return self

    @staticmethod
    def merge(batch: list[ModelInputT], attr_name: str):
        attr: list[torch.Tensor] = []
        for item in batch:
            if getattr(item, attr_name) is not None:
                attr.append(getattr(item, attr_name))
            else:
                return None
        return attr

    @classmethod
    def collate_fn(cls, batch: list[ModelInputT]):
        field_dict = {}
        for field, field_info in cls.model_fields.items():
            if field_info.annotation == torch.Tensor:
                attr = [getattr(item, field) for item in batch]
            elif field_info.annotation == torch.Tensor | None:
                attr = cls.merge(batch, field)
            else:
                raise NotImplementedError(f"Field {field} has unsupported type {field_info.annotation}")
            if field == "labels":
                attr = torch.tensor(attr, dtype=torch.int64)
            elif attr is not None:
                attr = pad_sequence(attr, batch_first=True)
            field_dict[field] = attr
        return cls(**field_dict)


class Pooler(nn.Module):
    def __init__(self, in_features: int, out_features: int | None = None, bias: bool = True):
        super().__init__()
        if out_features is None:
            out_features = in_features
        self.dense = nn.Linear(in_features, out_features, bias=bias)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class Backbone(nn.Module):
    def __init__(self, hidden_size: int, *, is_frozen: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_frozen = is_frozen

    def freeze(self):
        self.is_frozen = True

    def unfreeze(self):
        self.is_frozen = False

    @overload
    def pretrained_module(self, module: nn.Module) -> nn.Module: ...

    @overload
    def pretrained_module(self, module: None) -> None: ...

    def pretrained_module(self, module: nn.Module | None) -> nn.Module | None:
        if module is None:
            return None

        def call(*args, **kwargs):
            if self.is_frozen:
                with torch.no_grad():
                    return module.forward(*args, **kwargs)
            else:
                return module.forward(*args, **kwargs)

        module.__call__ = call
        return module


class __Backbone(nn.Module):
    def __init__(self, pretrained_model: nn.Module, hidden_size: int, *, is_frozen: bool = True):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.hidden_size = hidden_size
        self.is_frozen = is_frozen

    def freeze(self):
        self.is_frozen = True

    def unfreeze(self):
        self.is_frozen = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *, is_frozen: bool = True):
        model = AutoModel.from_pretrained(pretrained_model_name_or_path)
        return cls(
            model,
            model.config.hidden_size,
            is_frozen=is_frozen,
        )


class ClassifierModel(nn.Module):
    __call__: Callable[..., ClassifierOutput]

    def __init__(self, backbone: Backbone, *, num_classes: int, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.hidden_size = backbone.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            # nn.Linear(self.hidden_size, self.hidden_size),
            # nn.Tanh(),
            # nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_classes),
        )

    def freeze_backbone(self):
        self.backbone.freeze()

    def unfreeze_backbone(self):
        self.backbone.unfreeze()

    @cached_property
    def sample_weights(self):
        return 1 / self.class_weights if self.class_weights is not None else None

    @torch.compile()
    def classify(self, features: torch.Tensor, labels: torch.Tensor | None) -> ClassifierOutput:
        logits = self.classifier(features)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=self.sample_weights)
            loss = loss_fct(logits, labels)
        return ClassifierOutput(logits=logits, loss=loss)
