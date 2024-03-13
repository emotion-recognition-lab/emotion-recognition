from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer

from recognize.dataset import MELDDataset
from recognize.model import MultimodalInput, TextModel
from recognize.utils import (
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

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    feature_extracor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    train_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        tokenizer,
        feature_extracor,
        split="train",
        label_type="emotion",
    )
    dev_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        tokenizer,
        feature_extracor,
        split="dev",
        label_type="emotion",
    )
    test_dataset = MELDDataset(
        "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly",
        tokenizer,
        feature_extracor,
        split="test",
        label_type="emotion",
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
        AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2"),
        num_classes=7,
        class_weights=1 / torch.tensor(calculate_class_weights(train_data_loader, num_classes=7)).cuda(),
    ).cuda()
    train_and_eval(model, 30, train_data_loader, test_data_loader)
    # print(dataset[0])
    # for batch in data_loader:
    #     print(batch)
    #     print(model(batch))
    # break
