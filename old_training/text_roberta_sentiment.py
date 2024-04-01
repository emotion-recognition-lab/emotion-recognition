from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from recognize.dataset import DatasetSplit, MELDDataset, MELDDatasetLabelType
from recognize.model import LazyMultimodalInput, TextModel
from recognize.utils import (
    train_and_eval,
)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    train_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        tokenizer,
        split=DatasetSplit.TRAIN,
        label_type=MELDDatasetLabelType.SENTIMENT,
    )
    dev_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        tokenizer,
        split=DatasetSplit.DEV,
        label_type=MELDDatasetLabelType.SENTIMENT,
    )
    test_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        tokenizer,
        split=DatasetSplit.TEST,
        label_type=MELDDatasetLabelType.SENTIMENT,
    )
    train_data_loader = DataLoader(
        train_dataset,
        num_workers=4,
        batch_size=64,
        shuffle=False,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
    )
    dev_data_loader = DataLoader(
        dev_dataset,
        num_workers=4,
        batch_size=64,
        shuffle=False,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
    )
    test_data_loader = DataLoader(
        test_dataset,
        num_workers=4,
        batch_size=64,
        shuffle=False,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
    )

    model = TextModel(
        AutoModel.from_pretrained("FacebookAI/roberta-base"),
        num_classes=train_dataset.num_classes,
        class_weights=torch.tensor(train_dataset.class_weights, dtype=torch.float32).cuda(),
    ).cuda()
    result = train_and_eval(
        model, train_data_loader, test_data_loader, num_epochs=100, model_label="text--roberta-base--sentiment"
    )

    result.print()
    # 97.63740114125538 67.24137931034483 97.34414747521095 66.37389599086532
