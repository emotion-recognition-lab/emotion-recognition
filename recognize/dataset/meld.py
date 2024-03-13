from __future__ import annotations

import os

import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from ..model import MultimodalInput


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

    def __init__(self, meta_path, tokenizer, feature_extractor, *, split="train", label_type="emotion"):
        self.meta_path = meta_path
        self.meta = pd.read_csv(f"{meta_path}/{split}_sent_emo.csv", sep=",", index_col=0, header=0)
        self.split = split
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        if label_type == "emotion":
            self.label2int = self.emotion2int
            self.num_classes = 7
        elif label_type == "sentiment":
            self.label2int = self.sentiment2int
            self.num_classes = 3
        else:
            raise ValueError(f"Unsupported label type {label_type}")

    def __getitem__(self, index: int) -> MultimodalInput:
        item = self.meta.iloc[index]
        label = self.label2int(item)

        text_inputs = self.tokenizer(item["Utterance"], return_attention_mask=True, return_tensors="pt")
        text_input_ids = text_inputs["input_ids"][0]
        text_attention_mask = text_inputs["attention_mask"][0]

        utt_id = item["Utterance_ID"]
        dia_id = item["Dialogue_ID"]
        audio_path = f"{self.meta_path}/{self.split}/dia{dia_id}_utt{utt_id}.flac"
        if os.path.exists(audio_path):
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

        return MultimodalInput(
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            audio_input_values=audio_input_values,
            audio_attention_mask=audio_attention_mask,
            labels=torch.tensor(label, dtype=torch.int64),
        )

    def __len__(self):
        return len(self.meta)
