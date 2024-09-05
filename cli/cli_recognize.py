from __future__ import annotations

import os
import random
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import typer
from loguru import logger
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from recognize.config import InferenceConfig, load_training_config, save_config
from recognize.dataset import MELDDataset, PilotDataset
from recognize.evaluate import TrainingResult
from recognize.model import (
    LazyMultimodalInput,
    MultimodalBackbone,
    MultimodalModel,
    UnimodalModel,
)
from recognize.module import gen_fusion_layer
from recognize.preprocessor import Preprocessor
from recognize.trainer import EarlyStopper
from recognize.typing import DatasetLabelType
from recognize.utils import (
    find_best_model,
    load_best_model,
    train_and_eval,
    train_and_eval_distill,
)

if TYPE_CHECKING:
    from recognize.dataset.base import MultimodalDataset
    from recognize.typing import LogLevel, ModalType


def init_logger(log_level: LogLevel):
    handler = RichHandler(highlighter=NullHighlighter(), markup=True)
    logger.remove()
    logger.add(handler, format="{message}", level=log_level)


def seed_everything(seed: int | None = None):
    if seed is None:
        return
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def provide_meld_datasets(
    dataset_path: Path,
    label_type: DatasetLabelType = "emotion",
    custom_unique_id: str = "T",
    dataset_class_str: str = "MELDDataset",
):
    dataset_class = {
        "MELDDataset": MELDDataset,
        "PilotDataset": PilotDataset,
    }[dataset_class_str]
    custom_unique_id = f"{dataset_class_str}--{custom_unique_id}"
    train_dataset = dataset_class(
        dataset_path.as_posix(),
        split="train",
        label_type=label_type,
        custom_unique_id=custom_unique_id,
    )
    dev_dataset = dataset_class(
        dataset_path.as_posix(),
        split="valid",
        label_type=label_type,
        custom_unique_id=custom_unique_id,
    )
    test_dataset = dataset_class(
        dataset_path.as_posix(),
        split="test",
        label_type=label_type,
        custom_unique_id=custom_unique_id,
    )
    return train_dataset, dev_dataset, test_dataset


def symlink(src: Path | str, dst: Path | str, target_is_directory: bool = False):
    dst = Path(dst)
    src = Path(src)
    dst.unlink(missing_ok=True)
    logger.info(f"Create symlink: {dst} -> {src}")
    dst.symlink_to(src.absolute(), target_is_directory=target_is_directory)


def generate_model_label(modal: list[ModalType], freeze: bool, label_type: str):
    model_label = "+".join(modal)
    if label_type == "sentiment":
        model_label += "--S"
    else:
        model_label += "--E"
    if freeze:
        model_label += "F"
    else:
        model_label += "T"
    return model_label


def generate_preprocessor(encoders: list[str], modals: list[ModalType], checkpoints: Sequence[Path] = ()):
    from transformers import (
        AutoFeatureExtractor,
        AutoImageProcessor,
        AutoTokenizer,
        VivitImageProcessor,
    )

    for checkpoint in checkpoints:
        if checkpoint.exists():
            preprocessor = Preprocessor.from_pretrained(checkpoint)
            logger.info(f"Load preprocessor from [blue]./{checkpoint}[/]")
            break
    else:
        preprocessor = Preprocessor(device="cuda")

    for modal, model_name in zip(modals, encoders, strict=True):
        if modal == "T":
            preprocessor.tokenizer = preprocessor.tokenizer or AutoTokenizer.from_pretrained(model_name)
        elif modal == "A":
            preprocessor.feature_extractor = preprocessor.feature_extractor or AutoFeatureExtractor.from_pretrained(
                model_name
            )
        elif modal == "V":
            preprocessor.image_processor = preprocessor.image_processor or VivitImageProcessor.from_pretrained(
                model_name
            )
        else:
            preprocessor.image_processor = preprocessor.image_processor or AutoImageProcessor.from_pretrained(
                model_name
            )

    return preprocessor


def generate_backbone(
    encoders: list[str], modals: list[ModalType], feature_sizes: list[int], checkpoints: Sequence[Path] = ()
):
    from transformers import AutoModel

    for checkpoint in checkpoints:
        if checkpoint.exists():
            backbone = MultimodalBackbone.from_checkpoint(checkpoint)
            logger.info(f"Load backbone from [blue]./{checkpoint}[/]")
            break
    else:
        backbone = MultimodalBackbone(
            {
                modal: (AutoModel.from_pretrained(model_name), feature_size)
                for modal, model_name, feature_size in zip(modals, encoders, feature_sizes, strict=True)
            }
        )

    return backbone


def generate_preprocessor_and_backbone(
    encoders: list[str],
    modals: list[ModalType],
    feature_sizes: list[int],
    datasets: list[MultimodalDataset],
    checkpoints: Sequence[Path] = (),
):
    from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoModel, AutoTokenizer, VivitImageProcessor

    for checkpoint in checkpoints:
        preprocessor_path = checkpoint / "preprocessor"
        if preprocessor_path.exists():
            preprocessor = Preprocessor.from_pretrained(preprocessor_path)
            logger.info(f"Load preprocessor from [blue]./{checkpoint}[/]")
            break
    else:
        preprocessor = Preprocessor(device="cuda")

    for modal, model_name in zip(modals, encoders, strict=True):
        if modal == "T":
            preprocessor.tokenizer = preprocessor.tokenizer or AutoTokenizer.from_pretrained(model_name)
        elif modal == "A":
            preprocessor.feature_extractor = preprocessor.feature_extractor or AutoFeatureExtractor.from_pretrained(
                model_name
            )
        elif modal == "V":
            preprocessor.image_processor = preprocessor.image_processor or VivitImageProcessor.from_pretrained(
                model_name
            )
        else:
            preprocessor.image_processor = preprocessor.image_processor or AutoImageProcessor.from_pretrained(
                model_name
            )
    for dataset in datasets:
        dataset.set_preprocessor(preprocessor)
    for checkpoint in checkpoints:
        backbone_path = checkpoint / "backbones"
        if backbone_path.exists():
            backbone = MultimodalBackbone.from_checkpoint(backbone_path)
            logger.info(f"Load backbone from [blue]./{checkpoint}[/]")
            break
    else:
        backbone = MultimodalBackbone(
            {
                modal: (AutoModel.from_pretrained(model_name), feature_size)
                for modal, model_name, feature_size in zip(modals, encoders, feature_sizes, strict=True)
            },
            init_hook=datasets[0].special_process,
        )
    return preprocessor, backbone


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def distill(
    config_path: Path,
    teacher_checkpoint: Path = typer.Option(..., help="The checkpoint of the teacher model"),
    checkpoint: Path | None = None,
    seed: int | None = None,
) -> None:
    """
    When using knowledge distillation,
    it is essential to ensure that the preprocessor for both the teacher model and the student model is the same.
    Generally, you can use the one corresponding to the student model,
    as the student model typically has more modalities.
    """

    seed_everything(seed)

    config = load_training_config(config_path)
    config_freeze = config.model.freeze_backbone
    config_encoders = config.model.encoders
    config_modals = config.model.modals
    config_feature_sizes = config.model.feature_sizes
    config_fusion = config.model.fusion
    config_dataset = config.dataset

    init_logger(config.log_level)

    model_label = f"distill--{config.generate_model_label()}"
    batch_size = config.batch_size

    assert config_fusion is not None
    assert Path(f"./{teacher_checkpoint}/stopper.json").exists()
    assert Path(f"./{teacher_checkpoint}/preprocessor").exists()

    preprocessor = generate_preprocessor(
        config_encoders,
        config_modals,
        checkpoints=[
            Path(f"./checkpoints/training/{model_label}/preprocessor"),
            Path(f"./{checkpoint}/preprocessor"),
            Path(f"./{teacher_checkpoint}/preprocessor"),
        ],
    )

    Path(f"./checkpoints/training/{model_label}").mkdir(exist_ok=True)
    preprocessor_dir = Path(f"./checkpoints/training/{model_label}/preprocessor")
    if not preprocessor_dir.exists():
        preprocessor.save_pretrained(preprocessor_dir)

    teacher_backbone = MultimodalBackbone.from_checkpoint(
        Path(f"{teacher_checkpoint}/backbones"),
    )
    logger.info(f"Load backbone from [blue]./{teacher_checkpoint}/backbones[/]")
    backbone = generate_backbone(
        config_encoders,
        config_modals,
        config_feature_sizes,
        # checkpoints=[Path(f"./checkpoints/training/{model_label}/backbones"), Path(f"./{checkpoint}/backbones")],
    )
    # TODO: maybe those will be covered by load_best_model
    backbone.named_encoders.update(dict(teacher_backbone.named_encoders))
    backbone.named_poolers.update(dict(teacher_backbone.named_poolers))

    train_dataset, dev_dataset, test_dataset = provide_meld_datasets(
        config_dataset.path,
        label_type=config_dataset.label_type,
        custom_unique_id="+".join(config.model.modals),
        dataset_class_str=config_dataset.dataset_class,
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
    )
    dev_data_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
    )
    class_weights = torch.tensor(train_dataset.class_weights, dtype=torch.float32).cuda()

    symlink(
        config_path,
        f"./checkpoints/training/{model_label}/training.toml",
    )
    inference_config = InferenceConfig(num_classes=train_dataset.num_classes, model=config.model)
    save_config(inference_config, f"./checkpoints/training/{model_label}/inference.toml")

    teacher_model = UnimodalModel(
        teacher_backbone,
        feature_size=config.model.feature_sizes[0],
        num_classes=train_dataset.num_classes,
        class_weights=class_weights,
    ).cuda()

    load_best_model(teacher_checkpoint, teacher_model)
    result = TrainingResult.auto_compute(teacher_model, test_data_loader)
    logger.info("Test result in best [green]teacher[/] model:")
    result.print()

    fusion_layer = gen_fusion_layer(config_fusion, config_modals, config_feature_sizes)
    model = MultimodalModel(
        backbone,
        fusion_layer,
        feature_sizes=config.model.feature_sizes,
        num_classes=train_dataset.num_classes,
        class_weights=class_weights,
    ).cuda()

    if not config_freeze:
        model.unfreeze_backbone()

    if checkpoint is not None and not os.path.exists(f"./checkpoints/training/{model_label}/stopper.json"):
        load_best_model(checkpoint, model)
        result = TrainingResult.auto_compute(model, test_data_loader)
        result.print()

    result: TrainingResult = train_and_eval_distill(
        teacher_model,
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        num_epochs=200,
        model_label=model_label,
    )

    logger.info("Test result in best model:")
    result.print()


@app.command()
def train(
    config_path: Path,
    checkpoint: Path | None = None,
    seed: int | None = None,
) -> None:
    seed_everything(seed)

    config = load_training_config(config_path)
    init_logger(config.log_level)
    config_freeze = config.model.freeze_backbone
    config_encoders = config.model.encoders
    config_feature_sizes = config.model.feature_sizes
    config_modals = config.model.modals
    config_dataset = config.dataset

    model_label = config.generate_model_label()
    batch_size = config.batch_size

    checkpoint_dir = Path(f"./checkpoints/training/{model_label}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, dev_dataset, test_dataset = provide_meld_datasets(
        config_dataset.path,
        label_type=config_dataset.label_type,
        custom_unique_id="+".join(config.model.modals),
        dataset_class_str=config_dataset.dataset_class,
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
    )
    dev_data_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
    )
    preprocessor, backbone = generate_preprocessor_and_backbone(
        config_encoders,
        config_modals,
        config_feature_sizes,
        datasets=[train_dataset, dev_dataset, test_dataset],
        checkpoints=[Path(f"./checkpoints/training/{model_label}"), Path(f"./{checkpoint}")],
    )

    symlink(
        config_path,
        f"./checkpoints/training/{model_label}/training.toml",
    )
    inference_config = InferenceConfig(num_classes=train_dataset.num_classes, model=config.model)
    save_config(inference_config, f"./checkpoints/training/{model_label}/inference.toml")

    class_weights = torch.tensor(train_dataset.class_weights, dtype=torch.float32).cuda()
    if not os.path.exists(f"./checkpoints/training/{model_label}/preprocessor"):
        preprocessor.save_pretrained(f"./checkpoints/training/{model_label}/preprocessor")

    if config.model.fusion is None:
        model = UnimodalModel(
            backbone,
            feature_size=config.model.feature_sizes[0],
            num_classes=train_dataset.num_classes,
            class_weights=class_weights,
        ).cuda()
    else:
        fusion_layer = gen_fusion_layer(config.model.fusion, config_modals, config_feature_sizes)
        model = MultimodalModel(
            backbone,
            fusion_layer,
            feature_sizes=config.model.feature_sizes,
            num_classes=train_dataset.num_classes,
            class_weights=class_weights,
        ).cuda()

    if not config_freeze:
        model.unfreeze_backbone()

    if checkpoint is not None and not os.path.exists(f"./checkpoints/training/{model_label}/stopper.json"):
        load_best_model(checkpoint, model)
        result = TrainingResult.auto_compute(model, test_data_loader)
        result.print()

    result: TrainingResult = train_and_eval(
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        stopper=EarlyStopper(patience=20),
        num_epochs=200,
        model_label=model_label,
    )
    logger.info("Test result in best model:")
    result.print()


@app.command()
def evaluate(
    checkpoint: Path,
    seed: int | None = None,
) -> None:
    seed_everything(seed)

    config = load_training_config(checkpoint / "training.toml")
    init_logger(config.log_level)
    config_freeze = config.model.freeze_backbone
    config_encoders = config.model.encoders
    config_feature_sizes = config.model.feature_sizes
    config_modals = config.model.modals
    config_dataset = config.dataset

    model_label = config.generate_model_label()
    batch_size = config.batch_size

    preprocessor = generate_preprocessor(
        config_encoders,
        config_modals,
        checkpoints=[Path(f"./checkpoints/training/{model_label}/preprocessor"), Path(f"./{checkpoint}/preprocessor")],
    )

    backbone = generate_backbone(
        config_encoders,
        config_modals,
        config_feature_sizes,
        checkpoints=[Path(f"./checkpoints/training/{model_label}/backbones"), Path(f"./{checkpoint}/backbones")],
    )

    train_dataset, dev_dataset, test_dataset = provide_meld_datasets(
        config_dataset.path,
        label_type=config_dataset.label_type,
        custom_unique_id="+".join(config.model.modals),
        dataset_class_str=config_dataset.dataset_class,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
    )

    if not os.path.exists(f"./checkpoints/training/{model_label}/preprocessor"):
        preprocessor.save_pretrained(f"./checkpoints/training/{model_label}/preprocessor")

    if config.model.fusion is None:
        model = UnimodalModel(
            backbone,
            feature_size=config.model.feature_sizes[0],
            num_classes=train_dataset.num_classes,
        ).cuda()
    else:
        fusion_layer = gen_fusion_layer(config.model.fusion, config_modals, config_feature_sizes)
        model = MultimodalModel(
            backbone,
            fusion_layer,
            feature_sizes=config.model.feature_sizes,
            num_classes=train_dataset.num_classes,
        ).cuda()

    if not config_freeze:
        model.unfreeze_backbone()

    load_best_model(checkpoint, model)
    model.parameters()
    result = TrainingResult.auto_compute(model, test_data_loader)
    result.print()


@app.command()
def inference(
    checkpoint: Path,
    text: str | None = None,
    audio_path: Path | None = None,
    video_path: Path | None = None,
):
    config = load_training_config(f"{checkpoint}/training.toml")
    config_modals = config.model.modals
    config_feature_sizes = config.model.feature_sizes
    config_fusion = config.model.fusion
    init_logger(config.log_level)

    preprocessor = Preprocessor.from_pretrained(f"{checkpoint}/preprocessor")
    model_checkpoint = f"{checkpoint}/{find_best_model(checkpoint)}"

    backbone = MultimodalBackbone.from_checkpoint(Path(f"{model_checkpoint}/backbones"))
    if config_fusion is None:
        model = UnimodalModel.from_checkpoint(model_checkpoint, backbone=backbone).cuda()
    else:
        fusion_layer = gen_fusion_layer(config_fusion, config_modals, config_feature_sizes)
        model = MultimodalModel.from_checkpoint(model_checkpoint, backbone=backbone, fusion_layer=fusion_layer).cuda()
    model.eval()
    inputs = LazyMultimodalInput(
        preprocessor=preprocessor,
        texts=[text] if text is not None else None,
        audio_paths=[audio_path.as_posix()] if audio_path is not None else None,
        video_paths=[video_path.as_posix()] if video_path is not None else None,
    ).cuda()
    outputs = model(inputs)
    print("Predicted logits:", outputs.logits)
    print("Predicted labels:", torch.argmax(outputs.logits, dim=-1).item())


if __name__ == "__main__":
    app()
