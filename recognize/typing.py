from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, runtime_checkable

import torch
from torch import nn
from typing_extensions import TypeVar

if TYPE_CHECKING:
    from .model.base import Backbone, ClassifierModel, ModelInput


StateDict: TypeAlias = dict[str, torch.Tensor]
StateDicts: TypeAlias = dict[str, StateDict]
ModelInputT = TypeVar("ModelInputT", bound="ModelInput")
BackboneT = TypeVar("BackboneT", bound="Backbone")
ClassifierModelT = TypeVar("ClassifierModelT", bound="ClassifierModel")
ModuleT = TypeVar("ModuleT", bound=nn.Module)

ModalType = Literal["T", "A", "V"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

DatasetSplit = Literal["train", "valid", "test", "dev"]
DatasetLabelType = Literal["emotion", "sentiment"]
SupportedFusionLayer = Literal["TensorFusionLayer", "LowRankFusionLayer", "MultimodalMoE", "MultiHeadFusionMoE"]


@runtime_checkable
class FusionLayerLike(Protocol):
    output_size: int

    def __call__(self, inputs: dict[str, torch.Tensor | None]) -> torch.Tensor: ...
