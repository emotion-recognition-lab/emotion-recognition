from __future__ import annotations

import shutil
from pathlib import Path

import typer
from loguru import logger
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler

from recognize.typing import LogLevel
from recognize.utils import find_best_model


def init_logger(log_level: LogLevel):
    handler = RichHandler(highlighter=NullHighlighter(), markup=True)
    logger.remove()
    logger.add(handler, format="{message}", level=log_level)


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def generate_inference_checkpoints(
    checkpoint: Path,
    inference_checkpoint: Path,
    log_level: LogLevel = LogLevel.DEBUG,
):
    init_logger(log_level)
    if inference_checkpoint.exists():
        logger.warning(f"{inference_checkpoint} already exists, will be overwritten")
        shutil.rmtree(inference_checkpoint)

    inference_checkpoint.mkdir(parents=True, exist_ok=True)
    best_epoch = find_best_model(checkpoint)
    shutil.copytree(checkpoint / "preprocessor", inference_checkpoint / "preprocessor")
    for subpath in (checkpoint / f"{best_epoch}").glob("*"):
        if subpath.is_file():
            shutil.copy2(subpath, inference_checkpoint / subpath.name)
        else:
            shutil.copytree(subpath, inference_checkpoint / subpath.name)
    logger.info(f"Inference checkpoints generated: [blue]{inference_checkpoint}")


if __name__ == "__main__":
    app()
