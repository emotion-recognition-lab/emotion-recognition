from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import typer
from loguru import logger
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from recognize.config import load_config
from recognize.dataset import DatasetSplit, MELDDataset, MELDDatasetLabelType
from recognize.evaluate import TrainingResult
from recognize.model import (
    LazyMultimodalInput,
    MultimodalBackbone,
    MultimodalModel,
    UnimodalModel,
)
from recognize.module.fusion import gen_fusion_layer
from recognize.preprocessor import Preprocessor
from recognize.typing import LogLevel, ModalType
from recognize.utils import (
    find_best_model,
    load_best_model,
    train_and_eval,
    train_and_eval_distill,
)


def init_logger(log_level: LogLevel):
    handler = RichHandler(highlighter=NullHighlighter(), markup=True)
    logger.remove()
    logger.add(handler, format="{message}", level=log_level)


def seed_everything(seed: int = 666):
    import random

    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def provide_meld_datasets(dataset_path: Path, preprocessor: Preprocessor, label_type: MELDDatasetLabelType = "emotion"):
    train_dataset = MELDDataset(
        dataset_path.as_posix(),
        preprocessor,
        split=DatasetSplit.TRAIN,
        label_type=label_type,
        custom_unique_id="T",
    )
    dev_dataset = MELDDataset(
        dataset_path.as_posix(),
        preprocessor,
        split=DatasetSplit.VALID,
        label_type=label_type,
        custom_unique_id="T",
    )
    test_dataset = MELDDataset(
        dataset_path.as_posix(),
        preprocessor,
        split=DatasetSplit.TEST,
        label_type=label_type,
        custom_unique_id="T",
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


modal2name = {
    "T": "text",
    "V": "video",
    "A": "audio",
}


def generate_preprocessor(encoders: list[str], modals: list[ModalType], preprocessor_checkpoint: Path | None = None):
    from transformers import (
        AutoFeatureExtractor,
        AutoImageProcessor,
        AutoTokenizer,
        VivitImageProcessor,
    )

    assert len(encoders) == len(modals), "The number of encoders and modals should be the same"

    if preprocessor_checkpoint is not None:
        preprocessor = Preprocessor.from_pretrained(preprocessor_checkpoint)
        logger.info(f"Load preprocessor from [blue]./{preprocessor_checkpoint}[/]")
    else:
        preprocessor = Preprocessor()

    for modal, model_name in zip(modals, encoders, strict=True):
        if modal == "T":
            preprocessor.tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif modal == "A":
            preprocessor.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        elif modal == "V":
            preprocessor.image_processor = VivitImageProcessor.from_pretrained(model_name)
        else:
            preprocessor.image_processor = AutoImageProcessor.from_pretrained(model_name)

    return preprocessor


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def distill(
    teacher_checkpoint: Path,
    config_path: Path = Path("configs/default.toml"),
    checkpoint: Optional[Path] = None,
) -> None:
    """
    When using knowledge distillation,
    it is essential to ensure that the preprocessor for both the teacher model and the student model is the same.
    Generally, you can use the one corresponding to the student model,
    as the student model typically has more modalities.
    """
    from transformers import (
        AutoModel,
    )

    config = load_config(config_path)
    init_logger(config.log_level)
    config_freeze = config.model.freeze_backbone
    config_label_type = config.dataset.label_type
    config_encoders = config.model.encoders
    config_modals = config.model.modals
    config_feature_sizes = config.model.feature_sizes
    config_fusion = config.model.fusion

    model_label = f"distill--{generate_model_label(config_modals, config_freeze, config_label_type)}"
    batch_size = 64 if config_freeze else 2

    assert os.path.exists(f"./{teacher_checkpoint}/preprocessor")
    assert config_fusion is not None

    preprocessor = generate_preprocessor(
        config_encoders,
        config_modals,
        preprocessor_checkpoint=Path(f"./{teacher_checkpoint}/preprocessor"),
    )

    Path(f"./checkpoints/{model_label}").mkdir(exist_ok=True)
    preprocessor_dir = Path(f"./checkpoints/{model_label}/preprocessor")
    if not preprocessor_dir.exists():
        preprocessor.save_pretrained(preprocessor_dir)

    symlink(
        config_path,
        f"./checkpoints/{model_label}/config.toml",
    )
    teacher_backbone = MultimodalBackbone.from_checkpoint(
        f"{teacher_checkpoint}/backbones",
    )
    logger.info("load backbone from checkpoint")

    train_dataset, dev_dataset, test_dataset = provide_meld_datasets(
        config.dataset.path, preprocessor, label_type=config_label_type
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

    teacher_model = UnimodalModel(
        teacher_backbone,
        feature_size=config.model.feature_sizes[0],
        num_classes=train_dataset.num_classes,
        class_weights=class_weights,
    ).cuda()

    if os.path.exists(f"./{teacher_checkpoint}/stopper.json"):
        load_best_model(teacher_checkpoint, teacher_model)
        result = TrainingResult.auto_compute(teacher_model, test_data_loader)
        logger.info("Test result(best model):")
        result.print()

    backbone = MultimodalBackbone(
        {
            modal2name[modal]: (AutoModel.from_pretrained(model_name), feature_size)
            for modal, model_name, feature_size in zip(
                config_modals, config_encoders, config_feature_sizes, strict=True
            )
        }
    )
    backbone.encoders["text"] = teacher_backbone.encoders["text"]
    backbone.poolers["text"] = teacher_backbone.poolers["text"]
    fusion_layer = gen_fusion_layer(config_fusion)
    model = MultimodalModel(
        backbone,
        fusion_layer,
        feature_sizes=config.model.feature_sizes,
        num_classes=train_dataset.num_classes,
        class_weights=class_weights,
    ).cuda()

    if not config_freeze:
        model.unfreeze_backbone()

    result: TrainingResult = train_and_eval_distill(
        teacher_model,
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        num_epochs=200,
        model_label=model_label,
    )


@app.command()
def train(
    config_path: Path = Path("configs/default.toml"),
    checkpoint: Optional[Path] = None,
) -> None:
    from transformers import (
        AutoModel,
    )

    config = load_config(config_path)
    init_logger(config.log_level)
    freeze = config.model.freeze_backbone
    label_type = config.dataset.label_type
    encoders = config.model.encoders
    encoders = config.model.encoders
    config_feature_sizes = config.model.feature_sizes
    modals = config.model.modals

    model_label = generate_model_label(modals, freeze, label_type)
    batch_size = 64 if freeze else 2

    if os.path.exists(f"./checkpoints/{model_label}/preprocessor"):
        preprocessor_checkpoint = Path(f"./checkpoints/{model_label}/preprocessor")
    elif os.path.exists(f"./{checkpoint}/preprocessor"):
        preprocessor_checkpoint = Path(f"./{checkpoint}/preprocessor")
    else:
        preprocessor_checkpoint = None

    preprocessor = generate_preprocessor(encoders, modals, preprocessor_checkpoint=preprocessor_checkpoint)
    if os.path.exists(f"./{checkpoint}/backbones"):
        backbone = MultimodalBackbone.from_checkpoint(f"./{checkpoint}/backbones")
        logger.info(f"load backbone({backbone.hash}) from checkpoint")
    else:
        backbone = MultimodalBackbone(
            {
                modal2name[modal]: (AutoModel.from_pretrained(model_name), feature_size)
                for modal, model_name, feature_size in zip(modals, encoders, config_feature_sizes, strict=True)
            }
        )

    checkpoint_dir = Path(f"./checkpoints/{model_label}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    symlink(
        config_path,
        f"./checkpoints/{model_label}/config.toml",
    )

    train_dataset, dev_dataset, test_dataset = provide_meld_datasets(
        config.dataset.path, preprocessor, label_type=label_type
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
    if checkpoint is None and not os.path.exists(f"./{checkpoint}/preprocessor") and preprocessor.tokenizer is not None:
        preprocessor.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": train_dataset.speakers  # type: ignore
            }
        )
        backbone.encoders["text"].resize_token_embeddings(len(preprocessor.tokenizer))
    if not os.path.exists(f"./checkpoints/{model_label}/preprocessor"):
        preprocessor.save_pretrained(f"./checkpoints/{model_label}/preprocessor")

    if config.model.fusion is None:
        model = UnimodalModel(
            backbone,
            feature_size=config.model.feature_sizes[0],
            num_classes=train_dataset.num_classes,
            class_weights=class_weights,
        ).cuda()
    else:
        fusion_layer = gen_fusion_layer(config.model.fusion)
        model = MultimodalModel(
            backbone,
            fusion_layer,
            feature_sizes=config.model.feature_sizes,
            num_classes=train_dataset.num_classes,
            class_weights=class_weights,
        ).cuda()

    if not freeze:
        model.unfreeze_backbone()

    if checkpoint is not None and not os.path.exists(f"./checkpoints/{model_label}/stopper.json"):
        load_best_model(checkpoint, model)
        result = TrainingResult.auto_compute(model, test_data_loader)
        logger.info("Test result(best model):")
        result.print()

    result: TrainingResult = train_and_eval(
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        num_epochs=200,
        model_label=model_label,
    )
    best_epoch = find_best_model(checkpoint_dir)
    logger.info(f"Test result(best model in epoch {best_epoch}):")
    result.print()


@app.command()
def inference(
    checkpoint: Path,
    text: Optional[str] = None,
    audio_path: Optional[Path] = None,
    video_path: Optional[Path] = None,
):
    config = load_config(f"{checkpoint}/config.toml")
    init_logger(config.log_level)

    preprocessor = Preprocessor.from_pretrained(f"{checkpoint}/preprocessor")
    model_checkpoint = f"{checkpoint}/{find_best_model(checkpoint)}"

    backbone = MultimodalBackbone.from_checkpoint(f"{model_checkpoint}/backbones")
    if config.model.fusion is None:
        model = UnimodalModel.from_checkpoint(model_checkpoint, backbone=backbone).cuda()
    else:
        fusion_layer = gen_fusion_layer(config.model.fusion)
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
