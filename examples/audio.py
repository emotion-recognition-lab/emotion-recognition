from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer

from recognize.dataset import MELDDataset, MELDDatasetLabelType, MELDDatasetSplit
from recognize.model import MultimodalInput, MultimodalModel
from recognize.model.utils import calculate_accuracy, calculate_class_weights, calculate_f1_score, train_and_eval

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    feature_extracor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    train_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        tokenizer,
        feature_extracor,
        split=MELDDatasetSplit.TRAIN,
        label_type=MELDDatasetLabelType.EMOTION,
    )
    dev_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        tokenizer,
        feature_extracor,
        split=MELDDatasetSplit.DEV,
        label_type=MELDDatasetLabelType.EMOTION,
    )
    test_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        tokenizer,
        feature_extracor,
        split=MELDDatasetSplit.TEST,
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
    class_weights = (
        1 / torch.tensor(calculate_class_weights(train_data_loader, num_classes=train_dataset.num_classes)).cuda()
    )
    model = MultimodalModel(
        AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2"),
        AutoModel.from_pretrained("facebook/wav2vec2-base-960h"),
        num_classes=train_dataset.num_classes,
        class_weights=class_weights,
    ).cuda()
    model.freeze_backbone()

    train_and_eval(model, 100, train_data_loader, dev_data_loader, "multimodal")

    test_accuracy = calculate_accuracy(model, test_data_loader)
    test_f1_score = calculate_f1_score(model, test_data_loader)
    print(test_accuracy, test_f1_score)
