from __future__ import annotations

from typing import Callable, TypeVar

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
            else:
                raise NotImplementedError(f"Field {field} has unsupported type {field_info.annotation}")
            if field == "labels":
                attr = torch.tensor(attr, dtype=torch.int64)
            elif attr is not None:
                attr = pad_sequence(attr, batch_first=True)
            field_dict[field] = attr
        return cls(**field_dict)


class Backbone(nn.Module):
    def __init__(self, hidden_size: int, is_frozen: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_frozen = is_frozen

    def freeze(self):
        self.is_frozen = True

    def unfreeze(self):
        self.is_frozen = False

    def __call__(self, inputs: ModelInput) -> torch.Tensor:
        if self.is_frozen:
            with torch.no_grad():
                return super().__call__(inputs)
        else:
            return super().__call__(inputs)


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

    @torch.compile()
    def classify(self, features: torch.Tensor, labels: torch.Tensor | None) -> ClassifierOutput:
        logits = self.classifier(features)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits, labels)
        return ClassifierOutput(logits=logits, loss=loss)
