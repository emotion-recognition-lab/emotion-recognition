from __future__ import annotations

import os
import shutil
from pathlib import Path

import humanize
import typer
from loguru import logger
from rich.table import Table
from utils import OrderedSet, count_symlinks, init_logger

from recognize.trainer import EarlyStopper

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
    # shutil.copytree(checkpoint / "preprocessor", target_checkpoint / "preprocessor")
    shutil.copy2(checkpoint / "training.toml", target_checkpoint / "training.toml")
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
    """show checkpoint information"""
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
        logger.info(f"checkpoint: [blue]{subpath}[/]")
        logger.info(f"last epoch: [blue]{stopper.history[-1][0]}[/], finished: [blue]{stopper.finished}[/]")
        logger.info(f"model.encoder: [blue]{config.model.encoder}[/]")
        logger.info(f"model.fusion: [blue]{config.model.fusion}[/]")
        logger.info(f"loss: [blue]{config.loss}[/]")
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
            str_row = [f"{v[k]:.4f}%" if k in v else "" for k in score_names]
            table.add_row(str(epoch), *str_row)
        print(table)


@app.command()
def clean(
    checkpoint_dir: Path = typer.Argument(Path("checkpoints"), help="The checkpoint directory to clean"),
    keep_best_only: bool = typer.Option(
        True, help="Only keep the best checkpoints (based on test_f1 and test_accuracy)"
    ),
):
    """Clean checkpoints directory by removing unused files and optionally keeping only the best checkpoints"""

    init_logger("INFO")
    cleaned_size = 0
    cleaned_count = 0

    if keep_best_only:
        # clean checkpoints, only keep the best
        for stopper_path in checkpoint_dir.glob("**/stopper.yaml"):
            if not stopper_path.is_file():
                continue
            checkpoint_subdir = stopper_path.parent
            stopper = EarlyStopper.from_file(stopper_path)
            if not stopper.finished:
                continue

            # get the best epoch
            best_epochs = set()
            for metric, epoch in stopper.best_epoch.items():
                if metric in ["test_f1", "test_accuracy"]:
                    best_epochs.add(epoch)
                    logger.info(f"Found best epoch {epoch} for metric {metric} in {checkpoint_subdir}")

            # delete non-best checkpoints
            for subpath in checkpoint_subdir.glob("[0-9]*"):
                if subpath.is_dir() and int(subpath.name) not in best_epochs:
                    size = sum(f.stat().st_size for f in subpath.rglob("*") if f.is_file())
                    shutil.rmtree(subpath)
                    cleaned_size += size
                    cleaned_count += 1
                    logger.info(f"Removed non-best checkpoint: {subpath}")

            # delete optimizer.pt file
            optimizer_path = checkpoint_subdir / "optimizer.pt"
            if optimizer_path.exists():
                optimizer_size = optimizer_path.stat().st_size
                optimizer_path.unlink()
                cleaned_size += optimizer_size
                readable_size = humanize.naturalsize(optimizer_size)
                logger.info(f"Removed optimizer.pt from {checkpoint_subdir}, freed {readable_size}")

    # clean encoder files without symlinks
    encoder_dir = checkpoint_dir / "encoders"
    if encoder_dir.exists():
        for subpath in os.listdir(encoder_dir):
            symlink_count = count_symlinks(encoder_dir / subpath, checkpoint_dir)
            if subpath.endswith("safetensors") and symlink_count == 0:
                logger.info(f"Removing unused encoder: {subpath}")
                cleaned_size += os.path.getsize(encoder_dir / subpath)
                os.remove(encoder_dir / subpath)
                cleaned_count += 1
        if cleaned_count > 0:
            readable_size = humanize.naturalsize(cleaned_size)
            logger.info(f"Cleaned {cleaned_count} unused encoder files, freed {readable_size}")


@app.command()
def analysis_dataset(
    dataset_dir: Path = typer.Argument(Path("datasets"), help="The dataset directory to analysis"),
    output_dir: Path = typer.Option(None, help="The output directory for plots"),
):
    from utils.visualization import collect_emotion_distribution_data, plot_emotion_distribution

    from recognize.dataset import IEMOCAPDataset, MELDDataset

    init_logger("INFO")

    name_to_dataset: dict[str, type[MELDDataset | IEMOCAPDataset]] = {
        "MELD": MELDDataset,
        "IEMOCAP": IEMOCAPDataset,
    }
    available_split = {"train", "valid", "test"}

    emotion_mapping = {
        "IEMOCAP": ["中性", "愤怒", "兴奋", "沮丧", "高兴", "悲伤"],
        "MELD": ["中性", "高兴", "悲伤", "愤怒", "恐惧", "厌恶", "惊讶"],
    }

    df = collect_emotion_distribution_data(
        dataset_dir=dataset_dir,
        name_to_dataset=name_to_dataset,
        available_split=available_split,
        emotion_mapping=emotion_mapping,
    )
    plot_emotion_distribution(df, output_dir)


@app.command()
def prune(
    checkpoint: Path,
    pruned_checkpoint: Path,
):
    """not implemented yet"""
    from recognize.config import load_training_config

    config = load_training_config(checkpoint / "config.toml")
    init_logger(config.log_level)
    raise NotImplementedError("Pruning is not implemented yet")


if __name__ == "__main__":
    app()
