from __future__ import annotations

import os

import torch
from loguru import logger

from recognize.typing import DatasetLabelType, DatasetSplit

from .base import MultimodalDataset


class SIMSDataset(MultimodalDataset):
    def __init__(
        self,
        dataset_path: str,
        *,
        split: DatasetSplit = "train",
        label_type: DatasetLabelType = "sentiment",
    ):
        if label_type != "sentiment":
            logger.warning("SIMS only supports sentiment label type")
        import pandas as pd

        meta = pd.read_csv(os.path.join(dataset_path, "label.csv"))
        meta = meta[meta["mode"] == split]
        super().__init__(
            dataset_path,
            meta,
            num_classes=3,
            split=split,
        )

    @staticmethod
    def label2int(item):
        sentiment = item["annotation"]
        str2int = {"Neutral": 0, "Positive": 1, "Negative": 2}
        return str2int[sentiment]

    def __getitem__(self, index: int):
        from recognize.model import LazyMultimodalInput

        item = self.meta.iloc[index]
        label = self.label2int(item)

        video_id = item["video_id"]
        audio_id = video_id.replace("video", "audio")
        clip_id = item["clip_id"]
        audio_path = f"{self.dataset_path}/Raw/{audio_id}/{clip_id}.flac"
        video_path = f"{self.dataset_path}/Raw/{video_id}/{clip_id}.mp4"

        assert self.preprocessor is not None, "Preprocessor is not set"
        return LazyMultimodalInput(
            preprocessor=self.preprocessor,
            texts=[item["text"]],
            audio_paths=[audio_path],
            video_paths=[video_path],
            labels=torch.tensor([label], dtype=torch.int64),
        )
