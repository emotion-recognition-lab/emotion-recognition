from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import torch
from loguru import logger

from recognize.typing import DatasetLabelType

if TYPE_CHECKING:
    from recognize.preprocessor import Preprocessor
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
        preprocessor: Preprocessor,
        *,
        label_type: DatasetLabelType = "sentiment",
        split: DatasetSplit = "train",
        custom_unique_id: str = "",
    ):
        import pandas as pd

        if split != "train":
            logger.warning("Pilot dataset only has train split. Ignoring split argument.")
            split = "train"
        if label_type != "sentiment":
            logger.warning("Pilot dataset only has sentiment label. Ignoring label_type argument.")
        self.split = split
        self.num_classes = 4

        super().__init__(
            dataset_path,
            pd.read_csv(f"{dataset_path}/pilot_data.csv", sep=",", index_col=0, header=0),
            preprocessor,
            num_classes=self.num_classes,
            split=split,
            custom_unique_id=custom_unique_id,
        )

    @cached_property
    def class_weights(self) -> list[float]:
        labels = self.meta.apply(self.label2int, axis=1)
        class_counts = [(labels == i).sum() for i in range(self.num_classes)]
        total_samples = sum(class_counts)
        class_weights = [class_count / total_samples for class_count in class_counts]
        return class_weights

    def __getitem__(self, index: int):
        from recognize.model import LazyMultimodalInput

        item = self.meta.iloc[index]
        label = self.label2int(item)
        return LazyMultimodalInput(
            preprocessor=self.preprocessor,
            unique_ids=[f"{self.custom_unique_id}--{self.split}_{index}"],
            texts=[item["Answer"]],
            labels=torch.tensor([label], dtype=torch.int64),
        )

    def __len__(self):
        return len(self.meta)
