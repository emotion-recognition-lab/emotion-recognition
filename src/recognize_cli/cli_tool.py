from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from loguru import logger
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler
from rich.table import Table

from recognize.config import load_training_config

if TYPE_CHECKING:
    from recognize.typing import LogLevel


def init_logger(log_level: LogLevel):
    handler = RichHandler(highlighter=NullHighlighter(), markup=True)
    logger.remove()
    logger.add(handler, format="{message}", level=log_level)


app = typer.Typer(pretty_exceptions_show_locals=False)


def count_symlinks(target_path: str | Path, search_root: str | Path):
    symlink_count = 0
    target_inode = os.stat(target_path).st_ino

    for root, dirs, files in os.walk(search_root):
        for name in dirs + files:
            path = os.path.join(root, name)
            try:
                if os.path.islink(path):
                    real_path = os.path.realpath(path)
                    if os.path.exists(real_path) and os.stat(real_path).st_ino == target_inode:
                        symlink_count += 1
                        logger.debug(f"Found symlink: {path} -> {real_path}")
            except (PermissionError, OSError) as e:
                logger.warning(f"Error accessing {path}: {e}")
                continue

    return symlink_count


def format_bytes(size):
    power = 2**10
    n = 0
    power_labels = {0: "", 1: "K", 2: "M", 3: "G", 4: "T"}
    while size > power:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}B"


@app.command()
def create_standalone(
    checkpoint: Path,
    target_checkpoint: Path,
):
    from recognize.config import load_training_config
    from recognize.utils import find_best_model

    training_config = load_training_config(checkpoint / "training.toml")
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
def info(
    path: Path,
    sort_by: str = "epoch",
    filter: str = "",
    max_show: int = 5,
):
    # TODO: add more information
    from rich import print

    from recognize.trainer import EarlyStopper

    init_logger("INFO")

    for subpath in path.glob("*"):
        if subpath.is_file():
            continue
        if not (subpath / "stopper.yaml").exists():
            continue
        if filter and filter not in str(subpath):
            continue
        stopper = EarlyStopper.from_file(subpath / "stopper.yaml")
        config = load_training_config(subpath / "training.toml")
        logger.info(f"info of [blue]{subpath}[/](finished: {stopper.finished})")
        logger.info(f"encoders: [blue]{config.model.encoders}[/]")
        logger.info(f"fusion: [blue]{config.model.fusion}[/]")
        result: dict[int, dict] = {}
        score_names = set()
        best_epoch: dict[str, int] = {}
        for epoch, record in stopper.history:
            result.setdefault(epoch, {})
            result[epoch].update(record)
            score_names.update(record.keys())
            for k, v in record.items():
                if k not in best_epoch or best_epoch[k] < v:
                    best_epoch[k] = epoch
        table = Table(show_header=False, show_lines=True)
        table.add_row("epoch", *[f"[bold]{k}[/]" for k in score_names])
        if sort_by == "epoch":
            columns = sorted(
                result.items(),
                key=lambda r: r[0],
            )
        else:
            columns = sorted(
                result.items(),
                key=lambda r: r[1].get(sort_by, 0),
                reverse=True,
            )

        for epoch, v in columns[:max_show]:
            str_row = [f"{v[k]:.4f}" if k in v else "" for k in score_names]
            table.add_row(str(epoch), *str_row)
        print(table)


@app.command()
def clean(
    checkpoint_dir: Path = typer.Argument(Path("checkpoints"), help="The checkpoint directory to clean"),
):
    init_logger("INFO")
    cleaned_size = 0
    cleaned_count = 0
    encoder_dir = checkpoint_dir / "encoders"
    for subpath in os.listdir(encoder_dir):
        symlink_count = count_symlinks(encoder_dir / subpath, checkpoint_dir)
        if subpath.endswith("safetensors") and symlink_count == 0:
            logger.info(f"Removing {subpath} for no symlinks")
            cleaned_size += os.path.getsize(encoder_dir / subpath)
            os.remove(encoder_dir / subpath)
            cleaned_count += 1
    readable_size = format_bytes(cleaned_size)
    logger.info(f"Cleaned {readable_size} of {cleaned_count} files")


@app.command()
def prune(
    checkpoint: Path,
    pruned_checkpoint: Path,
):
    from recognize.config import load_training_config

    config = load_training_config(checkpoint / "config.toml")
    init_logger(config.log_level)
    raise NotImplementedError("Pruning is not implemented yet")


if __name__ == "__main__":
    app()
