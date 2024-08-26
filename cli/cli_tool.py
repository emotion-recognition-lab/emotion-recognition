from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from loguru import logger
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler

if TYPE_CHECKING:
    from recognize.typing import LogLevel


def init_logger(log_level: LogLevel):
    handler = RichHandler(highlighter=NullHighlighter(), markup=True)
    logger.remove()
    logger.add(handler, format="{message}", level=log_level)


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def create_standalone(
    checkpoint: Path,
    target_checkpoint: Path,
):
    from recognize.config import load_inference_config, load_training_config
    from recognize.utils import find_best_model

    training_config = load_training_config(checkpoint / "training.toml")
    inference_config = load_inference_config(checkpoint / "inference.toml")
    init_logger(training_config.log_level)

    if target_checkpoint.exists():
        logger.warning(f"{target_checkpoint} already exists, will be overwritten")
        shutil.rmtree(target_checkpoint)
    target_checkpoint.mkdir(parents=True, exist_ok=True)

    best_epoch = find_best_model(checkpoint)
    logger.info(f"Best epoch found: [blue]{best_epoch}")
    shutil.copytree(checkpoint / "preprocessor", target_checkpoint / "preprocessor")
    for subpath in (checkpoint / f"{best_epoch}").glob("*"):
        if subpath.is_file():
            shutil.copy2(subpath, target_checkpoint / subpath.name)
        else:
            shutil.copytree(subpath, target_checkpoint / subpath.name)
    logger.info(f"Inference checkpoints generated: [blue]{target_checkpoint}")


@app.command()
def prune(
    checkpoint: Path,
    pruned_checkpoint: Path,
):
    from recognize.config import load_training_config

    config = load_training_config(checkpoint / "config.toml")
    init_logger(config.log_level)


if __name__ == "__main__":
    app()
