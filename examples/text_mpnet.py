from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from recognize.dataset import MELDDataset, MELDDatasetLabelType, MELDDatasetSplit
from recognize.model import MultimodalInput, MultimodalModel
from recognize.utils import train_and_eval

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    train_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        tokenizer,
        split=MELDDatasetSplit.TRAIN,
        label_type=MELDDatasetLabelType.EMOTION,
        custom_unique_id="T",
    )
    dev_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        tokenizer,
        split=MELDDatasetSplit.DEV,
        label_type=MELDDatasetLabelType.EMOTION,
        custom_unique_id="T",
    )
    test_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        tokenizer,
        split=MELDDatasetSplit.TEST,
        label_type=MELDDatasetLabelType.EMOTION,
        custom_unique_id="T",
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
        batch_size=64,
        shuffle=False,
        collate_fn=MultimodalInput.collate_fn,
        pin_memory=True,
    )
    test_data_loader = DataLoader(
        test_dataset,
        num_workers=4,
        batch_size=64,
        shuffle=False,
        collate_fn=MultimodalInput.collate_fn,
        pin_memory=True,
    )
    class_weights = torch.tensor(train_dataset.class_weights, dtype=torch.float32).cuda()
    model = MultimodalModel(
        AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2"),
        num_classes=train_dataset.num_classes,
        class_weights=class_weights,
    ).cuda()
    _, train_accuracy, test_accuracy, train_f1_score, test_f1_score = train_and_eval(
        model, train_data_loader, test_data_loader, num_epochs=200, model_label="text--all-mpnet-base-v2(freezed)"
    )

    print(train_accuracy, test_accuracy, train_f1_score, test_f1_score)
    # 96.69636600260286 59.46360153256705 94.61830442314921 57.29641334713633

    model.unfreeze_backbone()
    _, train_accuracy, test_accuracy, train_f1_score, test_f1_score = train_and_eval(
        model, train_data_loader, test_data_loader, num_epochs=200, model_label="text--all-mpnet-base-v2"
    )

    print(train_accuracy, test_accuracy, train_f1_score, test_f1_score)
