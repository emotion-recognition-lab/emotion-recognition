from __future__ import annotations

from enum import Enum
from functools import cached_property

import pandas as pd
import torch
from loguru import logger

from .base import DatasetSplit, MultimodalDataset
from .preprocessor import Preprocessor


class MELDDatasetLabelType(Enum):
    EMOTION = "emotion"
    SENTIMENT = "sentiment"


class MELDDataset(MultimodalDataset):
    @staticmethod
    def emotion2int(item):
        emotion = item["Emotion"]
        str2int = {"neutral": 0, "joy": 1, "sadness": 2, "anger": 3, "fear": 4, "disgust": 5, "surprise": 6}
        return str2int[emotion]

    @staticmethod
    def sentiment2int(item):
        sentiment = item["Sentiment"]
        str2int = {"neutral": 0, "positive": 1, "negative": 2}
        return str2int[sentiment]

    def __init__(
        self,
        dataset_path: str,
        preprocessor: Preprocessor,
        *,
        split: DatasetSplit = DatasetSplit.TRAIN,
        label_type: MELDDatasetLabelType = MELDDatasetLabelType.EMOTION,
        custom_unique_id: str = "",
    ):
        if split == DatasetSplit.DEV:
            logger.warning("DEV split is deprecated. Using VALID split instead.")
            split = DatasetSplit.VALID
        self.split = "dev" if split == DatasetSplit.VALID else split.value
        self.label_type = label_type.value

        if label_type == MELDDatasetLabelType.EMOTION:
            self.label2int = self.emotion2int
            self.num_classes = 7
        elif label_type == MELDDatasetLabelType.SENTIMENT:
            self.label2int = self.sentiment2int
            self.num_classes = 3
        else:
            raise ValueError(f"Unsupported label type {label_type}")

        super().__init__(
            dataset_path,
            pd.read_csv(f"{dataset_path}/{self.split}_sent_emo.csv", sep=",", index_col=0, header=0),
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

        utt_id = item["Utterance_ID"]
        dia_id = item["Dialogue_ID"]
        audio_path = f"{self.dataset_path}/audios/{self.split}/dia{dia_id}_utt{utt_id}.flac"
        video_path = f"{self.dataset_path}/videos/{self.split}/dia{dia_id}_utt{utt_id}.mp4"
        return LazyMultimodalInput(
            preprocessor=self.preprocessor,
            unique_ids=[f"{self.custom_unique_id}--{self.split}_{index}"],
            texts=[item["Utterance"]],
            audio_paths=[audio_path],
            video_paths=[video_path],
            labels=torch.tensor([label], dtype=torch.int64),
        )

    def __len__(self):
        return len(self.meta)
