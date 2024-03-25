from __future__ import annotations

import os
from enum import Enum
from functools import cached_property

import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from ..model import MultimodalInput
from .utils import read_videos


class MELDDatasetSplit(Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class MELDDatasetLabelType(Enum):
    EMOTION = "emotion"
    SENTIMENT = "sentiment"


class MELDDataset(Dataset):
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
        dataset_path,
        tokenizer,
        feature_extractor=None,
        image_processor=None,
        *,
        split: MELDDatasetSplit,
        label_type: MELDDatasetLabelType,
    ):
        self.split = split.value
        self.label_type = label_type.value

        self.dataset_path = dataset_path
        self.meta = pd.read_csv(f"{dataset_path}/{self.split}_sent_emo.csv", sep=",", index_col=0, header=0)

        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.image_processor = image_processor

        if label_type == MELDDatasetLabelType.EMOTION:
            self.label2int = self.emotion2int
            self.num_classes = 7
        elif label_type == MELDDatasetLabelType.SENTIMENT:
            self.label2int = self.sentiment2int
            self.num_classes = 3
        else:
            raise ValueError(f"Unsupported label type {label_type}")

    def load_audio(self, audio_path):
        if self.feature_extractor is not None and os.path.exists(audio_path):
            raw_speech, sampling_rate = sf.read(audio_path)
            audio_inputs = self.feature_extractor(
                raw_speech.mean(1), sampling_rate=sampling_rate, return_attention_mask=True, return_tensors="pt"
            )
            audio_input_values = audio_inputs["input_values"][0]
            audio_attention_mask = audio_inputs["attention_mask"][0]

            if audio_input_values.shape[0] < 3280:
                audio_input_values = torch.cat(
                    [
                        audio_input_values,
                        torch.zeros(3280 - audio_input_values.shape[0]),
                    ],
                    dim=0,
                )
        else:
            audio_input_values = None
            audio_attention_mask = None
        return audio_input_values, audio_attention_mask

    def load_video(self, video_path):
        if self.image_processor is not None and os.path.exists(video_path):
            videos = read_videos(video_path)
            video_inputs = self.image_processor(list(videos), return_tensors="pt")
            video_pixel_values = video_inputs["pixel_values"][0]
        else:
            video_pixel_values = None
        return video_pixel_values

    @cached_property
    def class_weights(self) -> list[float]:
        labels = self.meta.apply(self.label2int, axis=1)
        class_counts = [(labels == i).sum() for i in range(self.num_classes)]
        total_samples = sum(class_counts)
        class_weights = [class_count / total_samples for class_count in class_counts]
        return class_weights

    def __getitem__(self, index: int) -> MultimodalInput:
        item = self.meta.iloc[index]
        label = self.label2int(item)

        text_inputs = self.tokenizer(item["Utterance"], return_attention_mask=True, return_tensors="pt")
        text_input_ids = text_inputs["input_ids"][0]
        text_attention_mask = text_inputs["attention_mask"][0]

        utt_id = item["Utterance_ID"]
        dia_id = item["Dialogue_ID"]
        audio_path = f"{self.dataset_path}/audios/{self.split}/dia{dia_id}_utt{utt_id}.flac"
        video_path = f"{self.dataset_path}/videos/{self.split}/dia{dia_id}_utt{utt_id}.mp4"

        audio_input_values, audio_attention_mask = self.load_audio(audio_path)
        video_pixel_values = self.load_video(video_path)

        return MultimodalInput(
            unique_hash=hash(f"{self.split}_{index}"),
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            audio_input_values=audio_input_values,
            audio_attention_mask=audio_attention_mask,
            video_pixel_values=video_pixel_values,
            labels=torch.tensor(label, dtype=torch.int64),
        )

    def __len__(self):
        return len(self.meta)
