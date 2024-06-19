from __future__ import annotations

from enum import Enum


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
