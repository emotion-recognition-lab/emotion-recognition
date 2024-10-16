from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import torch
from loguru import logger

from recognize.typing import DatasetLabelType

if TYPE_CHECKING:
    from recognize.model import MultimodalBackbone
    from recognize.typing import DatasetSplit

from .base import MultimodalDataset


class PilotDataset(MultimodalDataset):
    @staticmethod
    def label2int(item):
        sentiment = item["Sentiment"]
        return int(sentiment)

    def __init__(
        self,
        dataset_path: str,
        *,
        label_type: DatasetLabelType = "sentiment",
        split: DatasetSplit = "train",
    ):
        import pandas as pd

        if split == "valid" or split == "dev":
            logger.warning("Pilot dataset does not have a valid and dev split. Using test split instead.")
            split = "test"
        if label_type != "sentiment":
            logger.warning("Pilot dataset only has sentiment label. Ignoring label_type argument.")
        self.split = split
        self.num_classes = 4

        super().__init__(
            dataset_path,
            pd.read_csv(f"{dataset_path}/pilot_data.csv", sep=",", index_col=0, header=0),
            num_classes=self.num_classes,
            split=split,
        )

        self.data = self._generate_data(split)

    def special_process(self, backbone: MultimodalBackbone):
        assert self.preprocessor is not None, "Preprocessor is not set"
        tokenizer = self.preprocessor.tokenizer
        if tokenizer is None:
            return
        text_backbone = backbone.named_encoders["T"]
        tokenizer.add_special_tokens({"additional_special_tokens": ["<q>", "<a>"]})
        text_backbone.resize_token_embeddings(len(tokenizer))

    def _generate_data(self, split: DatasetSplit = "train"):
        import pandas as pd

        groups = self.meta.groupby("QuestionType")

        grouped_iterrows = [group.iterrows() for _, group in groups]

        data = []
        for rows in itertools.product(*grouped_iterrows):
            data.append(
                {
                    "text": "".join([f"<q>{row['Question']}<a>{row['Answer']}" for _, row in rows]),
                    "label": self.label2int(rows[0][1]),
                }
            )

        return pd.DataFrame(data, columns=["text", "label"])

    def __getitem__(self, index: int):
        from recognize.model import LazyMultimodalInput

        item = self.data.iloc[index]
        text = item["text"]
        label = item["label"]

        assert self.preprocessor is not None, "Preprocessor is not set"
        return LazyMultimodalInput(
            preprocessor=self.preprocessor,
            texts=[text],
            labels=torch.tensor([label], dtype=torch.int64),
        )

    def __len__(self):
        return len(self.meta)
