from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer, VivitImageProcessor

from recognize.dataset import DatasetSplit, MELDDataset, MELDDatasetLabelType
from recognize.model import MultimodalInput, MultimodalModel
from recognize.utils import train_and_eval

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    feature_extracor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
    train_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        tokenizer,
        feature_extracor,
        # image_processor,
        split=DatasetSplit.TRAIN,
        label_type=MELDDatasetLabelType.EMOTION,
    )
    dev_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        tokenizer,
        feature_extracor,
        # image_processor,
        split=DatasetSplit.DEV,
        label_type=MELDDatasetLabelType.EMOTION,
    )
    test_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        tokenizer,
        feature_extracor,
        # image_processor,
        split=DatasetSplit.TEST,
        label_type=MELDDatasetLabelType.EMOTION,
    )
    train_data_loader = DataLoader(
        train_dataset,
        num_workers=4,
        batch_size=16,
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
    class_weights = torch.tensor(train_dataset.class_weights, dtype=torch.float32).cuda()
    model = MultimodalModel(
        AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2"),
        AutoModel.from_pretrained("facebook/wav2vec2-base-960h"),
        AutoModel.from_pretrained("google/vivit-b-16x2-kinetics400"),
        num_classes=train_dataset.num_classes,
        class_weights=class_weights,
    ).cuda()

    result = train_and_eval(model, train_data_loader, test_data_loader, num_epochs=200, model_label="multimodal")

    result.print()
