from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, TypeAlias, TypeVar

import torch
from torch import nn

if TYPE_CHECKING:
    from .model.base import ClassifierModel, ModelInput


StateDict: TypeAlias = dict[str, torch.Tensor]
StateDicts: TypeAlias = dict[str, StateDict]
ModelInputT = TypeVar("ModelInputT", bound="ModelInput")
ClassifierModelT = TypeVar("ClassifierModelT", bound="ClassifierModel")
ModuleT = TypeVar("ModuleT", bound=nn.Module)


class ModalType(str, Enum):
    TEXT = "T"
    AUDIO = "A"
    VIDEO = "V"
    TEXT_AUDIO = "T+A"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
