from __future__ import annotations

import os
from functools import cached_property
from typing import TYPE_CHECKING

import pandas as pd
import torch

if TYPE_CHECKING:
    from recognize.preprocessor import Preprocessor

from .base import DatasetSplit, MultimodalDataset


class SIMSDataset(MultimodalDataset):
    def __init__(
        self,
        dataset_path: str,
        preprocessor: Preprocessor,
        *,
        split: DatasetSplit = DatasetSplit.TRAIN,
        custom_unique_id: str = "",
    ):
        meta = pd.read_csv(os.path.join(dataset_path, "label.csv"))
        meta = meta[meta["mode"] == split.value]
        super().__init__(
            dataset_path,
            meta,
            preprocessor,
            num_classes=1,
            split=split,
            custom_unique_id=custom_unique_id,
        )

    @cached_property
    def class_weights(self) -> list[float]:
        labels = self.meta.apply(self.label2int, axis=1)
        positive_counts = (labels > 0).sum() / len(labels)
        return [positive_counts, 1 - positive_counts]

    @staticmethod
    def label2int(item):
        sentiment = item["label"]
        return float(sentiment)

    def __getitem__(self, index: int):
        from recognize.model import LazyMultimodalInput

        item = self.meta.iloc[index]
        label = self.label2int(item)

        video_id = item["video_id"]
        audio_id = video_id.replace("video", "audio")
        clip_id = item["clip_id"]
        audio_path = f"{self.dataset_path}/Raw/{audio_id}/{clip_id}.flac"
        video_path = f"{self.dataset_path}/Raw/{video_id}/{clip_id}.mp4"

        return LazyMultimodalInput(
            preprocessor=self.preprocessor,
            unique_ids=[f"{self.custom_unique_id}--{self.split}_{index}"],
            texts=[item["text"]],
            audio_paths=[audio_path],
            video_paths=[video_path],
            labels=torch.tensor(label, dtype=torch.int64),
        )
