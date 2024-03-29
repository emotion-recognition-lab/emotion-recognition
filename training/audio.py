from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer

from recognize.dataset import DatasetSplit, MELDDataset, MELDDatasetLabelType
from recognize.model import MultimodalInput, MultimodalModel
from recognize.utils import train_and_eval

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    feature_extracor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    train_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        tokenizer,
        feature_extracor,
        split=DatasetSplit.TRAIN,
        label_type=MELDDatasetLabelType.EMOTION,
    )
    dev_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        tokenizer,
        feature_extracor,
        split=DatasetSplit.DEV,
        label_type=MELDDatasetLabelType.EMOTION,
    )
    test_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        tokenizer,
        feature_extracor,
        split=DatasetSplit.TEST,
        label_type=MELDDatasetLabelType.EMOTION,
    )
    train_data_loader = DataLoader(
        train_dataset,
        num_workers=4,
        batch_size=2,
        shuffle=False,
        collate_fn=MultimodalInput.collate_fn,
        pin_memory=True,
    )
    dev_data_loader = DataLoader(
        dev_dataset,
        num_workers=4,
        batch_size=16,
        shuffle=False,
        collate_fn=MultimodalInput.collate_fn,
        pin_memory=True,
    )
    test_data_loader = DataLoader(
        test_dataset,
        num_workers=4,
        batch_size=16,
        shuffle=False,
        collate_fn=MultimodalInput.collate_fn,
        pin_memory=True,
    )
    # class_weights = torch.tensor([0.001, 3.0, 4.0, 3.0, 15.0, 15.0, 3.0]).cuda()
    class_weights = torch.tensor(train_dataset.class_weights, dtype=torch.float32)
    model = MultimodalModel(
        AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2"),
        AutoModel.from_pretrained("facebook/wav2vec2-base-960h"),
        num_classes=train_dataset.num_classes,
        class_weights=class_weights,
    ).cuda()
    model.freeze_backbone()

    train_and_eval(model, train_data_loader, dev_data_loader, num_epochs=100, model_label="multimodal")
