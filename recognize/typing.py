from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

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
