from __future__ import annotations

import os
import shutil
from pathlib import Path

import typer
from loguru import logger
from rich.table import Table
from utils import OrderedSet, count_symlinks, format_bytes, init_logger

app = typer.Typer(pretty_exceptions_show_locals=False)


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
def extract_encoders(
    checkpoint: Path,
    target_checkpoint: Path,
):
    import rtoml

    from recognize.config import ModelEncoderConfig, load_training_config
    from recognize.utils import find_best_model

    training_config = load_training_config(checkpoint / "training.toml")
    init_logger(training_config.log_level)

    if target_checkpoint.exists():
        logger.warning(f"{target_checkpoint} already exists, will be overwritten")
        shutil.rmtree(target_checkpoint)
    target_checkpoint.mkdir(parents=True, exist_ok=True)

    best_epoch = find_best_model(checkpoint)
    logger.info(f"Best epoch found: [blue]{best_epoch}")
    backbone_dir = checkpoint / f"{best_epoch}/backbone"
    logger.info(f"Loading backbone meta from [blue]{backbone_dir}/meta.toml")
    with open(backbone_dir / "meta.toml") as f:
        feature_sizes = rtoml.load(f)["feature_sizes"]

    encoder_configs: dict[str, dict] = {}
    for name in feature_sizes.keys():
        logger.info(f"Extracting encoder [blue]{name}[/]")
        shutil.copytree(backbone_dir / name, target_checkpoint / name)
        shutil.copy2(
            backbone_dir / f"{name}.safetensors",
            target_checkpoint / f"{name}.safetensors",
            follow_symlinks=False,
        )
        encoder_config = ModelEncoderConfig(
            model=name,
            feature_size=feature_sizes[name],
            checkpoint=target_checkpoint,
        )
        encoder_configs[name] = encoder_config.model_dump(mode="json")

    # TODO: to support combinations
    (target_checkpoint / "configs").mkdir(parents=True, exist_ok=True)
    for name, encoder_config in encoder_configs.items():
        with open(target_checkpoint / f"configs/{name}.toml", "w") as f:
            rtoml.dump({"model": {"encoder": encoder_config}}, f, none_value=None)
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

    from recognize.config import load_training_config
    from recognize.trainer import EarlyStopper

    init_logger("INFO")
    for stopper_path in path.glob("**/stopper.yaml"):
        if not stopper_path.is_file():
            continue
        subpath = stopper_path.parent
        if not stopper_path.exists():
            continue
        if filter and filter not in str(subpath):
            continue
        stopper = EarlyStopper.from_file(stopper_path)
        config = load_training_config(subpath / "training.toml")
        logger.info(f"find checkpoint: [blue]{subpath}[/]")
        logger.info(f"finished: [blue]{stopper.finished}[/]")
        logger.info(f"encoder: [blue]{config.model.encoder}[/]")
        logger.info(f"fusion: [blue]{config.model.fusion}[/]")
        result: dict[int, dict] = {}
        score_names = OrderedSet[str]()
        best_epoch: dict[str, int] = {}
        for epoch, record in stopper.history:
            result.setdefault(epoch, {})
            result[epoch].update(record)
            score_names |= record.keys()
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
    # clean encoder files with no symlinks
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
