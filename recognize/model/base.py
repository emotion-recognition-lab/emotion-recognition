from __future__ import annotations

from functools import cached_property
from typing import Callable, TypeVar, overload

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence

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
            elif field_info.annotation == str | list[str]:
                attr = [getattr(item, field) for item in batch]
            else:
                raise NotImplementedError(f"Field {field} has unsupported type {field_info.annotation}")
            if field == "labels":
                attr = torch.tensor(attr, dtype=torch.int64)
            elif isinstance(attr, torch.Tensor):
                attr = pad_sequence(attr, batch_first=True)
            elif isinstance(attr, list) and isinstance(attr[0], torch.Tensor):
                attr = pad_sequence(attr, batch_first=True)
            field_dict[field] = attr
        return cls(**field_dict)


class Pooler(nn.Module):
    def __init__(self, in_features: int, out_features: int | None = None, bias: bool = True):
        super().__init__()
        if out_features is None:
            out_features = in_features
        self.pool = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, out_features, bias=bias), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled_output = self.pool(x)
        return pooled_output


class Backbone(nn.Module):
    def __init__(self, output_size: int, *, is_frozen: bool = True):
        super().__init__()
        self.output_size = output_size
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
        module_forward = module.forward

        def forward(*args, **kwargs):
            if self.is_frozen:
                with torch.no_grad():
                    return module_forward(*args, **kwargs)
            else:
                return module_forward(*args, **kwargs)

        module.forward = forward
        return module


class ClassifierModel(nn.Module):
    __call__: Callable[..., ClassifierOutput]

    def __init__(
        self, backbone: Backbone, hidden_size: int, *, num_classes: int, class_weights: torch.Tensor | None = None
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.hidden_size = hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_size, num_classes),
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
