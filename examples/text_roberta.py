from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from recognize.dataset import MELDDataset, MELDDatasetLabelType, MELDDatasetSplit
from recognize.model import MultimodalInput, TextModel
from recognize.model.utils import (
    calculate_class_weights,
    train_and_eval,
)

# class ModelConfig(BaseModel):
#     model_config = ConfigDict(arbitrary_types_allowed=True)

# class TextModelConfig(ModelConfig):
#     num_classes: int

# class DatasetMeta(BaseModel):
#     emotions: list[str]

# class TextDatasetMeta(DatasetMeta):
#     text: list[str]


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    train_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        tokenizer,
        split=MELDDatasetSplit.TRAIN,
        label_type=MELDDatasetLabelType.EMOTION,
    )
    dev_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        tokenizer,
        split=MELDDatasetSplit.DEV,
        label_type=MELDDatasetLabelType.EMOTION,
    )
    test_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        tokenizer,
        split=MELDDatasetSplit.TEST,
        label_type=MELDDatasetLabelType.EMOTION,
    )
    train_data_loader = DataLoader(
        train_dataset,
        num_workers=4,
        batch_size=64,
        shuffle=False,
        collate_fn=MultimodalInput.collate_fn,
        pin_memory=True,
    )
    dev_data_loader = DataLoader(
        dev_dataset,
        num_workers=4,
        batch_size=32,
        shuffle=False,
        collate_fn=MultimodalInput.collate_fn,
        pin_memory=True,
    )
    test_data_loader = DataLoader(
        test_dataset,
        num_workers=4,
        batch_size=32,
        shuffle=False,
        collate_fn=MultimodalInput.collate_fn,
        pin_memory=True,
    )

    model = TextModel(
        AutoModel.from_pretrained("FacebookAI/roberta-base"),
        num_classes=7,
        class_weights=1 / torch.tensor(calculate_class_weights(train_data_loader, num_classes=7)).cuda(),
    ).cuda()
    train_and_eval(model, 100, train_data_loader, test_data_loader, checkpoint_label="text--roberta-base")
    # f1: 0.37
