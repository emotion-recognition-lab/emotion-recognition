from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
import typer
from loguru import logger
from pydantic import BaseModel, Field
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer, VivitImageProcessor

from recognize.dataset import DatasetSplit, MELDDataset, MELDDatasetLabelType, Preprocessor
from recognize.evaluate import TrainingResult
from recognize.model import (
    LazyMultimodalInput,
    LowRankFusionLayer,
    MultimodalBackbone,
    MultimodalModel,
)
from recognize.typing import LogLevel
from recognize.utils import find_best_model, load_best_model, train_and_eval


def init_logger(log_level: LogLevel):
    handler = RichHandler(highlighter=NullHighlighter(), markup=True)
    logger.remove()
    logger.add(handler, format="{message}", level=log_level)


def provide_meld_datasets(
    dataset_path: Path, preprocessor: Preprocessor, label_type=MELDDatasetLabelType.EMOTION
):
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


def clean_cache():
    for file in os.listdir("./cache"):
        file_path = os.path.join("./cache", file)
        os.unlink(file_path)


def generate_model_label(modal: ModalType, freeze: bool, label_type: MELDDatasetLabelType):
    model_label = modal.value
    if label_type == MELDDatasetLabelType.SENTIMENT:
        model_label += "--S"
    else:
        model_label += "--E"
    if freeze:
        model_label += "F"
    else:
        model_label += "T"
    return model_label


class ModalType(str, Enum):
    TEXT = "T"
    # AUDIO = "A"
    # VIDEO = "V"
    TEXT_AUDIO = "T+A"


class MultimodalModelConfig(BaseModel):
    modal: ModalType
    freeze: bool
    label_type: MELDDatasetLabelType
    fusion_method: str = "low-rank-fusion"
    backbones: list[str] = Field(
        ["sentence-transformers/all-mpnet-base-v2", "facebook/wav2vec2-base-960h"]
    )


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def train(
    dataset_path: Path,
    modal: ModalType = ModalType.TEXT,
    freeze: bool = True,
    checkpoint: Optional[Path] = None,
    label_type: MELDDatasetLabelType = MELDDatasetLabelType.EMOTION,
    log_level: LogLevel = "DEBUG",
) -> None:
    clean_cache()
    init_logger(log_level)
    model_label = generate_model_label(modal, freeze, label_type)
    batch_size = 64 if freeze else 2
    if checkpoint is not None and os.path.exists(f"./{checkpoint}/preprocessor"):
        preprocessor = Preprocessor.from_pretrained(f"./{checkpoint}/preprocessor")
        backbone = MultimodalBackbone.from_checkpoint(
            f"{checkpoint}/backbones",
        )
        logger.info("load preprocessor and backbone from checkpoint")
    else:
        preprocessor = Preprocessor(
            AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2"),
            AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h"),
            VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400"),
        )
        preprocessor.save_pretrained(f"./checkpoints/{model_label}/preprocessor")

        backbone = MultimodalBackbone(
            AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2"),
            AutoModel.from_pretrained("facebook/wav2vec2-base-960h"),
            AutoModel.from_pretrained("google/vit-base-patch16-224-in21k"),
        )

    train_dataset, dev_dataset, test_dataset = provide_meld_datasets(
        dataset_path, preprocessor, label_type=label_type
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

    text_feature_size, audio_feature_size, video_feature_size = 128, 16, 1
    fusion_layer = LowRankFusionLayer(
        [text_feature_size, audio_feature_size, video_feature_size], 16, 128
    )

    model = MultimodalModel(
        backbone,
        fusion_layer,
        text_feature_size=text_feature_size,
        audio_feature_size=audio_feature_size,
        video_feature_size=video_feature_size,
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
    logger.info(f"Test result(best model in epoch {result.best_epoch}):")
    result.print()


@app.command()
def inference(
    text: str | None = None,
    audio_path: Path | None = None,
    video_path: Path | None = None,
    checkpoint: Path = Path("."),
    *,
    log_level: LogLevel = "DEBUG",
):
    init_logger(log_level)
    preprocessor = Preprocessor.from_pretrained(f"{checkpoint}/preprocessor")
    model_checkpoint = f"{checkpoint}/{find_best_model(checkpoint)}"
    text_feature_size, audio_feature_size, video_feature_size = 128, 16, 1

    backbone = MultimodalBackbone.from_checkpoint(
        f"{model_checkpoint}/backbones",
        use_cache=False,
    )
    fusion_layer = LowRankFusionLayer(
        [text_feature_size, audio_feature_size, video_feature_size], 16, 128
    )
    model = MultimodalModel.from_checkpoint(
        model_checkpoint, backbone=backbone, fusion_layer=fusion_layer
    ).cuda()
    model.eval()
    inputs = LazyMultimodalInput(
        preprocessor=preprocessor,
        # texts=["I'm feeling really good today.", "I'm feeling really good today, I'm sad."],
        audio_paths=[
            "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly/audios/test/dia58_utt1.flac",
            "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly/audios/test/dia58_utt2.flac",
            "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly/audios/test/dia58_utt2.flac",
        ],
        video_paths=[
            "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly/videos/test/dia58_utt1.mp4",
            "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly/videos/test/dia58_utt1.mp4",
            "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly/videos/test/dia58_utt2.mp4",
        ],
    ).cuda()
    for _ in range(100):
        outputs = model(inputs)
        print(outputs.logits)


if __name__ == "__main__":
    app()
