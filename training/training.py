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

from recognize.dataset import DatasetSplit, MELDDataset, MELDDatasetLabelType
from recognize.evaluate import calculate_accuracy_and_f1_score
from recognize.model import LazyMultimodalInput, MultimodalModel
from recognize.utils import init_logger, load_best_model, train_and_eval

app = typer.Typer(pretty_exceptions_show_locals=False)


def provide_meld_datasets(
    tokenizer, feature_extracor=None, image_processor=None, label_type=MELDDatasetLabelType.EMOTION
):
    train_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        tokenizer,
        feature_extracor,
        image_processor,
        split=DatasetSplit.TRAIN,
        label_type=label_type,
        custom_unique_id="T",
    )
    dev_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        tokenizer,
        feature_extracor,
        image_processor,
        split=DatasetSplit.VALID,
        label_type=label_type,
        custom_unique_id="T",
    )
    test_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        tokenizer,
        feature_extracor,
        image_processor,
        split=DatasetSplit.TEST,
        label_type=label_type,
        custom_unique_id="T",
    )
    return train_dataset, dev_dataset, test_dataset


def clean_cache():
    for file in os.listdir("./cache"):
        file_path = os.path.join("./cache", file)
        os.unlink(file_path)


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
    batch_size = 64 if freeze else 2
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    feature_extracor = (
        AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h") if modal == ModalType.TEXT_AUDIO else None
    )
    # image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-8x2-kinetics400")
    train_dataset, dev_dataset, test_dataset = provide_meld_datasets(tokenizer, feature_extracor, label_type=label_type)
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

    model = MultimodalModel(
        AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2"),
        AutoModel.from_pretrained("facebook/wav2vec2-base-960h") if modal == ModalType.TEXT_AUDIO else None,
        num_classes=train_dataset.num_classes,
        class_weights=class_weights,
    ).cuda()
    model_label = modal.value

    if not freeze:
        model.unfreeze_backbone()
    else:
        model_label += "--frozen"
    if checkpoint is not None and not os.path.exists(f"./checkpoints/{model_label}/stopper.json"):
        load_best_model(checkpoint, model)
        test_accuracy, test_f1_score = calculate_accuracy_and_f1_score(model, test_data_loader)
        logger.info(f"Test result(best model): accuracy={test_accuracy}, f1_score={test_f1_score}")

    result = train_and_eval(
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
