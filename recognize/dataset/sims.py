from __future__ import annotations

import os
from functools import cached_property

import pandas as pd
import torch

from ..model import MultimodalInput
from .base import DatasetSplit, MultimodalDataset


class SIMSDataset(MultimodalDataset):
    def __init__(
        self,
        dataset_path,
        tokenizer,
        feature_extractor=None,
        image_processor=None,
        *,
        split: DatasetSplit = DatasetSplit.TRAIN,
        custom_unique_id: str = "",
    ):
        meta = pd.read_csv(os.path.join(dataset_path, "label.csv"))
        meta = meta[meta["mode"] == split.value]
        super().__init__(
            dataset_path,
            meta,
            tokenizer,
            feature_extractor,
            image_processor,
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

    def __getitem__(self, index: int) -> MultimodalInput:
        item = self.meta.iloc[index]
        label = self.label2int(item)

        video_id = item["video_id"]
        audio_id = video_id.replace("video", "audio")
        clip_id = item["clip_id"]
        audio_path = f"{self.dataset_path}/Raw/{audio_id}/{clip_id}.flac"
        video_path = f"{self.dataset_path}/Raw/{video_id}/{clip_id}.mp4"

        text_input_ids, text_attention_mask = self.load_text(item["text"])
        audio_input_values, audio_attention_mask = self.load_audio(audio_path)
        video_pixel_values = self.load_video(video_path)
        assert text_input_ids is not None, "Now, text must be provided"

        return MultimodalInput(
            unique_id=f"{self.custom_unique_id}--{self.split}_{index}",
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            audio_input_values=audio_input_values,
            audio_attention_mask=audio_attention_mask,
            video_pixel_values=video_pixel_values,
            labels=torch.tensor(label, dtype=torch.int64),
        )
