from __future__ import annotations

import torch
import typer
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from recognize.dataset import DatasetSplit, SIMSDataset
from recognize.model import LazyMultimodalInput, MultimodalModel
from recognize.utils import train_and_eval

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def train(freeze: bool = True):
    tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-chinese-extractive-qa")
    train_dataset = SIMSDataset(
        "/home/zrr/datasets/SIMS",
        tokenizer,
        split=DatasetSplit.TRAIN,
        custom_unique_id="SIMS--T",
    )
    valid_dataset = SIMSDataset(
        "/home/zrr/datasets/SIMS",
        tokenizer,
        split=DatasetSplit.VALID,
        custom_unique_id="SIMS--T",
    )
    test_dataset = SIMSDataset(
        "/home/zrr/datasets/SIMS",
        tokenizer,
        split=DatasetSplit.TEST,
        custom_unique_id="SIMS--T",
    )
    train_data_loader = DataLoader(
        train_dataset,
        num_workers=4,
        batch_size=64,
        shuffle=False,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
    )
    valid_data_loader = DataLoader(
        valid_dataset,
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

    model = MultimodalModel(
        AutoModel.from_pretrained("uer/roberta-base-chinese-extractive-qa"),
        # AutoModel.from_pretrained("facebook/wav2vec2-base-960h"),
        num_classes=train_dataset.num_classes,
    ).cuda()

    if not freeze:
        model.unfreeze_backbone()
        model_label = "SIMS--text--all-mpnet-base-v2"
    else:
        model_label = "SIMS--text--all-mpnet-base-v2--frozen"

    result = train_and_eval(
        model,
        train_data_loader,
        valid_data_loader,
        test_data_loader,
        num_epochs=200,
        model_label=model_label,
    )
    result.print()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    app()
