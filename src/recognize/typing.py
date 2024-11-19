from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, runtime_checkable

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


# TODO: deprecated
@runtime_checkable
class FusionLayerLike(Protocol):
    output_size: int

    def __call__(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor: ...

    def forward_with_loss(
        self, inputs: Mapping[str, torch.Tensor], label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
