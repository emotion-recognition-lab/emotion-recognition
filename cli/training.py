from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
import typer
from loguru import logger
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer

from recognize.dataset import DatasetSplit, MELDDataset, MELDDatasetLabelType, Preprocessor
from recognize.evaluate import TrainingResult
from recognize.model import LazyMultimodalInput, LowRankFusionLayer, MultimodalBackbone, MultimodalModel
from recognize.utils import init_logger, load_best_model, train_and_eval

app = typer.Typer(pretty_exceptions_show_locals=False)


def provide_meld_datasets(preprocessor, label_type=MELDDatasetLabelType.EMOTION):
    train_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        preprocessor,
        split=DatasetSplit.TRAIN,
        label_type=label_type,
        custom_unique_id="T",
    )
    dev_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        preprocessor,
        split=DatasetSplit.VALID,
        label_type=label_type,
        custom_unique_id="T",
    )
    test_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
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
        model_label += "--sentiment"
    if freeze:
        model_label += "--frozen"
    return model_label


class ModalType(str, Enum):
    TEXT = "T"
    # AUDIO = "A"
    # VIDEO = "V"
    TEXT_AUDIO = "T+A"


@app.command()
def train(
    modal: ModalType = ModalType.TEXT,
    freeze: bool = True,
    checkpoint: Optional[Path] = None,
    label_type: MELDDatasetLabelType = MELDDatasetLabelType.EMOTION,
    log_level: str = "DEBUG",
):
    clean_cache()
    init_logger(log_level)
    model_label = generate_model_label(modal, freeze, label_type)
    batch_size = 64 if freeze else 2
    if checkpoint is not None and os.path.exists(f"./{checkpoint}/preprocessor"):
        preprocessor = Preprocessor.load(f"./{checkpoint}/preprocessor")
        logger.info("load preprocessor from checkpoint")
    else:
        preprocessor = Preprocessor(
            AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2"),
            AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h"),
            # VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        )
        preprocessor.save(f"./checkpoints/{model_label}/preprocessor")

    train_dataset, dev_dataset, test_dataset = provide_meld_datasets(preprocessor, label_type=label_type)
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
    backbone = MultimodalBackbone(
        AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2"),
        AutoModel.from_pretrained("facebook/wav2vec2-base-960h"),
    )
    fusion_layer = LowRankFusionLayer([text_feature_size, audio_feature_size, video_feature_size], 16, 128)

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


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    app()
