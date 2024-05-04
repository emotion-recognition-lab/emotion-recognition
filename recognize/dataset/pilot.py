from __future__ import annotations

from functools import cached_property

import pandas as pd
import torch
from loguru import logger

from ..preprocessor import Preprocessor
from .base import DatasetSplit, MultimodalDataset


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
        split: DatasetSplit = DatasetSplit.TRAIN,
        custom_unique_id: str = "",
    ):
        if split != DatasetSplit.TRAIN:
            logger.warning("Pilot dataset only has train split. Ignoring split argument.")
            split = DatasetSplit.TRAIN
        self.split = split.value
        self.num_classes = 4

        super().__init__(
            dataset_path,
            pd.read_csv(f"{dataset_path}/pilot_data.csv", sep=",", index_col=0, header=0),
            preprocessor,
            num_classes=self.num_classes,
            split=split,
            custom_unique_id=custom_unique_id,
        )
        self.split = "dev" if split == DatasetSplit.VALID else split.value

    @cached_property
    def class_weights(self) -> list[float]:
        labels = self.meta.apply(self.label2int, axis=1)
        class_counts = [(labels == i).sum() for i in range(self.num_classes)]
        total_samples = sum(class_counts)
        class_weights = [class_count / total_samples for class_count in class_counts]
        return class_weights

    def __getitem__(self, index: int):
        from ..model import LazyMultimodalInput

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
