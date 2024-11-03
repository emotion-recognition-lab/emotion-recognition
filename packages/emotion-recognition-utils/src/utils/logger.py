from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from loguru import logger
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def init_logger(log_level: LogLevel, log_dir: Path | None = None):
    handler = RichHandler(highlighter=NullHighlighter(), markup=True)
    logger.remove()
    if log_dir is not None:
        log_file = log_dir / f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt"
        logger.add(log_file, format="{message}", level=log_level)
    logger.add(handler, format="{message}", level=log_level)
