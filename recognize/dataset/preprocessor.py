from __future__ import annotations

import os

import soundfile as sf
import torch

from .utils import read_videos


class Preprocessor:
    def __init__(self, tokenizer=None, feature_extractor=None, image_processor=None):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.image_processor = image_processor

    def load_text(self, text: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.tokenizer is not None:
            text_inputs = self.tokenizer(text, return_attention_mask=True, return_tensors="pt")
            text_input_ids = text_inputs["input_ids"][0]
            text_attention_mask = text_inputs["attention_mask"][0]
        else:
            text_input_ids = None
            text_attention_mask = None
        return text_input_ids, text_attention_mask

    def load_audio(self, audio_path: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
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
            # TODO: better way to reduce langth
            audio_input_values = audio_input_values[:200000]
            audio_attention_mask = audio_attention_mask[:200000]
        else:
            audio_input_values = None
            audio_attention_mask = None
        return audio_input_values, audio_attention_mask

    def load_video(self, video_path: str) -> torch.Tensor | None:
        if self.image_processor is not None and os.path.exists(video_path):
            videos = read_videos(video_path)
            video_inputs = self.image_processor(list(videos), return_tensors="pt")
            video_pixel_values = video_inputs["pixel_values"][0]
        else:
            video_pixel_values = None
        return video_pixel_values
