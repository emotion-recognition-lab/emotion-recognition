from __future__ import annotations

import os
from functools import cached_property

import torch

from recognize.typing import DatasetSplit

from .base import MultimodalDataset


class SIMSDataset(MultimodalDataset):
    def __init__(
        self,
        dataset_path: str,
        *,
        split: DatasetSplit = "train",
    ):
        import pandas as pd

        meta = pd.read_csv(os.path.join(dataset_path, "label.csv"))
        meta = meta[meta["mode"] == split]
        super().__init__(
            dataset_path,
            meta,
            num_classes=1,
            split=split,
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
        audio_id = video_id.replace("V", "A")
        clip_id = item["clip_id"]
        audio_path = f"{self.dataset_path}/Raw/{audio_id}/{clip_id}.flac"
        video_path = f"{self.dataset_path}/Raw/{video_id}/{clip_id}.mp4"

        assert self.preprocessor is not None, "Preprocessor is not set"
        return LazyMultimodalInput(
            preprocessor=self.preprocessor,
            texts=[item["T"]],
            audio_paths=[audio_path],
            video_paths=[video_path],
            labels=torch.tensor(label, dtype=torch.int64),
        )
