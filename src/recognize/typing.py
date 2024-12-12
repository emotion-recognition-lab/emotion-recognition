from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

from typing_extensions import TypeVar

if TYPE_CHECKING:
    import torch
    from torch import nn

    from .model import Backbone, ClassifierModel, ModelInput


StateDict: TypeAlias = dict[str, "torch.Tensor"]
StateDicts: TypeAlias = dict[str, StateDict]
ModelInputT = TypeVar("ModelInputT", bound="ModelInput")
BackboneT = TypeVar("BackboneT", bound="Backbone", covariant=True)
ClassifierModelT = TypeVar("ClassifierModelT", bound="ClassifierModel")
ModuleT = TypeVar("ModuleT", bound="nn.Module")

ModalType = Literal["T", "A", "V"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

DatasetClass = Literal["MELDDataset", "PilotDataset", "SIMSDataset", "IEMOCAPDataset"]
DatasetSplit = Literal["train", "valid", "test", "dev"]
DatasetLabelType = Literal["emotion", "sentiment"]
