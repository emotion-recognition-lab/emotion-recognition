from __future__ import annotations

import os
import random
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import typer
from loguru import logger
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from recognize.config import InferenceConfig, load_training_config, save_config
from recognize.dataset import MELDDataset, PilotDataset, SIMSDataset
from recognize.evaluate import TrainingResult
from recognize.model import (
    LazyMultimodalInput,
    MultimodalBackbone,
    MultimodalModel,
    UnimodalModel,
)
from recognize.module import gen_fusion_layer, get_feature_sizes_dict
from recognize.preprocessor import Preprocessor
from recognize.typing import DatasetLabelType
from recognize.utils import (
    find_best_model,
    load_best_model,
    train_and_eval,
)

if TYPE_CHECKING:
    from recognize.dataset import MultimodalDataset
    from recognize.typing import LogLevel, ModalType


def init_logger(log_level: LogLevel, label: str):
    handler = RichHandler(highlighter=NullHighlighter(), markup=True)
    logger.remove()
    logger.add(f"./logs/{label}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt")
    logger.add(handler, format="{message}", level=log_level)


def init_torch():
    torch.set_float32_matmul_precision("high")


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


def provide_datasets(
    dataset_path: Path,
    label_type: DatasetLabelType = "emotion",
    dataset_class_str: Literal["MELDDataset", "PilotDataset", "SIMSDataset"] = "MELDDataset",
) -> tuple[MultimodalDataset, MultimodalDataset, MultimodalDataset]:
    dataset_class: type[MELDDataset | PilotDataset | SIMSDataset] = {
        "MELDDataset": MELDDataset,
        "PilotDataset": PilotDataset,
        "SIMSDataset": SIMSDataset,
    }[dataset_class_str]
    train_dataset = dataset_class(
        dataset_path.as_posix(),
        split="train",
        label_type=label_type,
    )
    dev_dataset = dataset_class(
        dataset_path.as_posix(),
        split="valid",
        label_type=label_type,
    )
    test_dataset = dataset_class(
        dataset_path.as_posix(),
        split="test",
        label_type=label_type,
    )
    return train_dataset, dev_dataset, test_dataset


def symlink(src: Path | str, dst: Path | str, target_is_directory: bool = False):
    dst = Path(dst)
    src = Path(src)
    dst.unlink(missing_ok=True)
    logger.info(f"Create symlink: {dst} -> {src}")
    dst.symlink_to(src.absolute(), target_is_directory=target_is_directory)


def generate_preprocessor_and_backbone(
    encoders: list[str],
    modals: list[ModalType],
    feature_sizes: list[int],
    datasets: list[MultimodalDataset],
    checkpoints: Sequence[Path] = (),
):
    from transformers import (
        AutoConfig,
        AutoFeatureExtractor,
        AutoImageProcessor,
        AutoModel,
        AutoTokenizer,
        VivitImageProcessor,
    )

    for checkpoint in checkpoints:
        preprocessor_path = checkpoint / "preprocessor"
        if preprocessor_path.exists():
            preprocessor = Preprocessor.from_pretrained(preprocessor_path)
            logger.info(f"Load preprocessor from [blue]{checkpoint}[/]")
            break
    else:
        preprocessor = Preprocessor(device="cuda")

    for modal, encoder_config in zip(modals, encoders, strict=True):
        # TODO: use more elegant way to handle extra information
        encoder_name = encoder_config.split("@")[0]
        if modal == "T":
            preprocessor.tokenizer = preprocessor.tokenizer or AutoTokenizer.from_pretrained(encoder_name)
        elif modal == "A":
            preprocessor.feature_extractor = preprocessor.feature_extractor or AutoFeatureExtractor.from_pretrained(
                encoder_name
            )
        elif modal == "V":
            preprocessor.image_processor = preprocessor.image_processor or VivitImageProcessor.from_pretrained(
                encoder_name
            )
        else:
            preprocessor.image_processor = preprocessor.image_processor or AutoImageProcessor.from_pretrained(
                encoder_name
            )
    for dataset in datasets:
        dataset.set_preprocessor(preprocessor)
    for checkpoint in checkpoints:
        backbone_path = checkpoint / "backbone"
        if backbone_path.exists():
            backbone = MultimodalBackbone.from_checkpoint(backbone_path)
            logger.info(f"Load backbone from [blue]{checkpoint}[/]")
            break
    else:
        backbone_encoders = {}
        for modal, encoder_config, feature_size in zip(modals, encoders, feature_sizes, strict=True):
            # TODO: use more elegant way to handle extra information
            encoder_name, *encoder_extra = encoder_config.split("@")
            if len(encoder_extra) > 0:
                logger.info(f"Load {modal} encoder from [blue]{encoder_extra[0]}[/]")
                checkpoint_path = Path(encoder_extra[0])
                config = AutoConfig.from_pretrained(checkpoint_path / f"backbone/{modal}/config.json")
                backbone_encoder = AutoModel.from_config(config)
                backbone_encoder.load_state_dict(load_file(checkpoint_path / f"backbone/{modal}.safetensors"))
            else:
                backbone_encoder = AutoModel.from_pretrained(encoder_name)
            backbone_encoders[modal] = (backbone_encoder, feature_size)
        backbone = MultimodalBackbone(
            backbone_encoders,
            init_hook=datasets[0].special_process if len(datasets) > 0 else None,
        )
    return preprocessor, backbone


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def distill(
    config_path: list[Path],
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
    init_torch()

    config = load_training_config(*config_path)
    config_training_mode = config.model.training_mode
    config_encoders = config.model.encoders
    config_modals = config.model.modals
    config_feature_sizes = config.model.feature_sizes
    config_fusion = config.model.fusion
    config_dataset = config.dataset

    init_logger(config.log_level, config.label)

    model_label = f"distill--{config.model_label}"
    dataset_label = config.dataset_label
    batch_size = config.batch_size

    assert config_training_mode != "lora", "Lora is not supported in distillation"
    assert config_fusion is not None
    assert (teacher_checkpoint / "stopper.yaml").exists()
    assert (teacher_checkpoint / "preprocessor").exists()

    if checkpoint is not None:
        checkpoint_dir = checkpoint
    else:
        checkpoint_dir = Path(f"./checkpoints/training/{dataset_label}/{model_label}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, dev_dataset, test_dataset = provide_datasets(
        config_dataset.path,
        label_type=config_dataset.label_type,
        dataset_class_str=config_dataset.dataset_class,
    )
    preprocessor, backbone = generate_preprocessor_and_backbone(
        config_encoders,
        config_modals,
        config_feature_sizes,
        datasets=[train_dataset, dev_dataset, test_dataset],
        checkpoints=[checkpoint_dir],
    )

    teacher_backbone = MultimodalBackbone.from_checkpoint(teacher_checkpoint / "backbone")
    teacher_preprocessor = Preprocessor.from_pretrained(teacher_checkpoint / "preprocessor")
    # TODO: maybe those will be covered by load_best_model
    backbone.named_encoders.load_state_dict(teacher_backbone.named_encoders.state_dict(), strict=False)
    backbone.named_poolers.load_state_dict(teacher_backbone.named_poolers.state_dict(), strict=False)
    if preprocessor.tokenizer is not None:
        teacher_preprocessor.tokenizer = preprocessor.tokenizer
    if preprocessor.feature_extractor is not None:
        teacher_preprocessor.feature_extractor = preprocessor.feature_extractor
    if preprocessor.image_processor is not None:
        teacher_preprocessor.image_processor = preprocessor.image_processor

    preprocessor_dir = checkpoint_dir / "preprocessor"
    if not preprocessor_dir.exists():
        preprocessor.save_pretrained(preprocessor_dir)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
    )
    dev_data_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
    )
    class_weights = torch.tensor(train_dataset.class_weights, dtype=torch.float32).cuda()

    inference_config = InferenceConfig(num_classes=train_dataset.num_classes, model=config.model)
    save_config(config, checkpoint_dir / "training.toml")
    save_config(inference_config, checkpoint_dir / "inference.toml")

    teacher_model = UnimodalModel(
        teacher_backbone,
        feature_size=config.model.feature_sizes[0],
        num_classes=train_dataset.num_classes,
        class_weights=class_weights,
    ).cuda()

    load_best_model(teacher_checkpoint, teacher_model)
    result = TrainingResult.auto_compute(teacher_model, test_data_loader)
    logger.info("Test result in [green]best teacher model[/]:")
    result.print()

    feature_sizes_dict = get_feature_sizes_dict(config_modals, config_feature_sizes)
    fusion_layer = gen_fusion_layer(config_fusion, feature_sizes_dict)
    model = MultimodalModel(
        backbone,
        fusion_layer,
        feature_sizes_dict=feature_sizes_dict,
        num_classes=train_dataset.num_classes,
        class_weights=class_weights,
    ).cuda()

    if config_training_mode == "trainable":
        model.unfreeze_backbone()

    if checkpoint is not None and not (checkpoint_dir / "stopper.yaml").exists():
        load_best_model(checkpoint, model)
        result = TrainingResult.auto_compute(model, test_data_loader)
        result.print()

    result: TrainingResult = train_and_eval(
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        teacher_model=teacher_model,
        checkpoint_dir=checkpoint_dir,
        num_epochs=200,
        model_label=model_label,
    )

    logger.info(f"Test result in best model({model_label}):")
    result.print()


@app.command()
def train(
    config_path: list[Path],
    checkpoint: Path | None = None,
    seed: int | None = None,
) -> None:
    seed_everything(seed)
    init_torch()

    config = load_training_config(*config_path)
    init_logger(config.log_level, config.label)
    config_training_mode = config.model.training_mode
    config_encoders = config.model.encoders
    config_feature_sizes = config.model.feature_sizes
    config_modals = config.model.modals
    config_dataset = config.dataset

    model_label = config.model_label
    dataset_label = config.dataset_label
    batch_size = config.batch_size

    assert config_training_mode != "lora", "Lora is not supported in training"

    if checkpoint is not None:
        checkpoint_dir = checkpoint
    else:
        checkpoint_dir = Path(f"./checkpoints/training/{dataset_label}/{model_label}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, dev_dataset, test_dataset = provide_datasets(
        config_dataset.path,
        label_type=config_dataset.label_type,
        dataset_class_str=config_dataset.dataset_class,
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
    )
    dev_data_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
    )
    preprocessor, backbone = generate_preprocessor_and_backbone(
        config_encoders,
        config_modals,
        config_feature_sizes,
        datasets=[train_dataset, dev_dataset, test_dataset],
        checkpoints=[checkpoint_dir],
    )
    if not (checkpoint_dir / "preprocessor").exists():
        preprocessor.save_pretrained(checkpoint_dir / "preprocessor")

    inference_config = InferenceConfig(num_classes=train_dataset.num_classes, model=config.model)
    save_config(config, checkpoint_dir / "training.toml")
    save_config(inference_config, checkpoint_dir / "inference.toml")

    class_weights = torch.tensor(train_dataset.class_weights, dtype=torch.float32).cuda()

    if config.model.fusion is None:
        assert len(config_modals) == 1, "Multiple modals must give a fusion layer"
        model = UnimodalModel(
            backbone,
            feature_size=config.model.feature_sizes[0],
            num_classes=train_dataset.num_classes,
            class_weights=class_weights,
        ).cuda()
    else:
        feature_sizes_dict = get_feature_sizes_dict(config_modals, config_feature_sizes)
        fusion_layer = gen_fusion_layer(config.model.fusion, feature_sizes_dict)
        model = MultimodalModel(
            backbone,
            fusion_layer,
            feature_sizes_dict=feature_sizes_dict,
            num_classes=train_dataset.num_classes,
            class_weights=class_weights,
        ).cuda()
    if config_training_mode == "trainable":
        model.unfreeze_backbone()
    if checkpoint is not None and not (checkpoint_dir / "stopper.yaml").exists():
        load_best_model(checkpoint, model)
        result = TrainingResult.auto_compute(model, test_data_loader)
        result.print()

    result: TrainingResult = train_and_eval(
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        checkpoint_dir=checkpoint_dir,
        num_epochs=200,
        model_label=model_label,
    )
    logger.info(f"Test result in best model({model_label}):")
    result.print()


@app.command()
def evaluate(checkpoint: Path) -> None:
    config = load_training_config(checkpoint / "training.toml")
    init_logger(config.log_level, config.label)
    config_encoders = config.model.encoders
    config_feature_sizes = config.model.feature_sizes
    config_modals = config.model.modals
    config_dataset = config.dataset

    batch_size = config.batch_size

    assert (checkpoint / "preprocessor").exists(), "Preprocessor not found, the checkpoint is not valid"

    _, _, test_dataset = provide_datasets(
        config_dataset.path,
        label_type=config_dataset.label_type,
        dataset_class_str=config_dataset.dataset_class,
    )

    _, backbone = generate_preprocessor_and_backbone(
        config_encoders,
        config_modals,
        config_feature_sizes,
        datasets=[test_dataset],
        checkpoints=[checkpoint],
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
    )
    if config.model.fusion is None:
        model = UnimodalModel(
            backbone,
            feature_size=config.model.feature_sizes[0],
            num_classes=test_dataset.num_classes,
        ).cuda()
    else:
        feature_sizes_dict = get_feature_sizes_dict(config_modals, config_feature_sizes)
        fusion_layer = gen_fusion_layer(config.model.fusion, feature_sizes_dict)
        model = MultimodalModel(
            backbone,
            fusion_layer,
            feature_sizes_dict=feature_sizes_dict,
            num_classes=test_dataset.num_classes,
        ).cuda()

    model.freeze_backbone()
    load_best_model(checkpoint, model)
    result = TrainingResult.auto_compute(model, test_data_loader)
    result.print()


@app.command()
def inference(
    checkpoint: Path,
    text: str | None = None,
    audio_path: Path | None = None,
    video_path: Path | None = None,
):
    config = load_training_config(checkpoint / "training.toml")
    config_modals = config.model.modals
    config_feature_sizes = config.model.feature_sizes
    config_fusion = config.model.fusion
    init_logger(config.log_level, config.label)

    preprocessor = Preprocessor.from_pretrained(f"{checkpoint}/preprocessor")
    model_checkpoint = checkpoint / str(find_best_model(checkpoint))

    backbone = MultimodalBackbone.from_checkpoint(Path(f"{model_checkpoint}/backbone"))
    if config_fusion is None:
        model = UnimodalModel.from_checkpoint(model_checkpoint, backbone=backbone).cuda()
    else:
        feature_sizes_dict = get_feature_sizes_dict(config_modals, config_feature_sizes)
        fusion_layer = gen_fusion_layer(config_fusion, feature_sizes_dict)
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
