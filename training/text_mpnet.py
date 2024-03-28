from __future__ import annotations

import torch
import typer
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from recognize.dataset import MELDDataset, MELDDatasetLabelType, MELDDatasetSplit
from recognize.model import MultimodalInput, MultimodalModel
from recognize.utils import train_and_eval

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def train(freeze: bool = True):
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
        # AutoModel.from_pretrained("facebook/wav2vec2-base-960h"),
        num_classes=train_dataset.num_classes,
        class_weights=class_weights,
    ).cuda()

    if not freeze:
        model.unfreeze_backbone()
        model_label = "text--all-mpnet-base-v2"
    else:
        model_label = "text--all-mpnet-base-v2--frozen"

    result = train_and_eval(
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        num_epochs=200,
        model_label=model_label,
    )
    result.print()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    app()
