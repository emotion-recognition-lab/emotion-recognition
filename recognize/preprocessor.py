from __future__ import annotations

import os
from pathlib import Path

import soundfile as sf
import torch
from faster_whisper import WhisperModel
from loguru import logger
from torch.nn.utils.rnn import pad_sequence

from .dataset.utils import read_videos


class Preprocessor:
    def __init__(self, tokenizer=None, feature_extractor=None, image_processor=None):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.image_processor = image_processor
        self.whisper_model: WhisperModel | None = None

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer):
        if tokenizer is not None:
            tokenizer.truncation_side = "left"
            tokenizer.padding_side = "left"
        self._tokenizer = tokenizer

    def load_text(self, text: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.tokenizer is not None:
            tokenized = self.tokenizer.tokenize(text)[-self.tokenizer.model_max_length :]
            text_input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokenized))
            text_attention_mask = torch.ones(len(text_input_ids))
        else:
            text_input_ids = None
            text_attention_mask = None
        return text_input_ids, text_attention_mask

    def load_audio(self, audio_path: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.feature_extractor is not None and os.path.exists(audio_path):
            raw_speech, sampling_rate = sf.read(audio_path)
            audio_inputs = self.feature_extractor(
                raw_speech.mean(1),
                sampling_rate=sampling_rate,
                return_attention_mask=True,
                return_tensors="pt",
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
            video = read_videos(video_path)
            video_inputs = self.image_processor(video, return_tensors="pt")
            video_pixel_values = video_inputs["pixel_values"][0]
        else:
            video_pixel_values = None
        return video_pixel_values

    def load_texts(self, texts: list[str]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.tokenizer is not None:
            text_inputs = self.tokenizer(
                texts,
                return_attention_mask=True,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )
            text_input_ids = text_inputs["input_ids"]
            text_attention_mask = text_inputs["attention_mask"]
        else:
            text_input_ids = None
            text_attention_mask = None
        return text_input_ids, text_attention_mask

    def load_audios(self, audio_paths: list[str]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.feature_extractor is None:
            return None, None
        audio_input_values_list, audio_attention_mask_list = [], []
        for audio_path in audio_paths:
            if not os.path.exists(audio_path):
                return None, None

            raw_speech, sampling_rate = sf.read(audio_path)
            audio_inputs = self.feature_extractor(
                raw_speech.mean(1),
                sampling_rate=sampling_rate,
                return_attention_mask=True,
                return_tensors="pt",
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

            audio_input_values = audio_input_values[:200000]
            audio_attention_mask = audio_attention_mask[:200000]
            audio_input_values_list.append(audio_input_values)
            audio_attention_mask_list.append(audio_attention_mask)
        return pad_sequence(audio_input_values_list, batch_first=True), pad_sequence(
            audio_attention_mask_list, batch_first=True
        )

    def load_videos(self, video_paths: list[str]) -> torch.Tensor | None:
        if self.image_processor is None:
            return None
        video_pixel_values_list = []
        for video_path in video_paths:
            if not os.path.exists(video_path):
                return None
            video = read_videos(video_path)
            video_inputs = self.image_processor(video, return_tensors="pt")
            video_pixel_values = video_inputs["pixel_values"][0]
            video_pixel_values_list.append(video_pixel_values)
        return pad_sequence(video_pixel_values_list, batch_first=True)

    def recoginize_audio(self, audio_path: str) -> str:
        if self.whisper_model is None:
            # TODO: use self.device
            self.whisper_model = WhisperModel(
                "medium",
                device="cuda",
                compute_type="float16",
                download_root=os.environ.get("WHISPER_DOWNLOAD_ROOT", None),
            )
        segments, _ = self.whisper_model.transcribe(audio_path, language="zh", initial_prompt="简体")
        text = "。".join(seg.text for seg in segments)
        logger.debug(f"Recognized text: {text}")
        return text

    def save_pretrained(self, path: str | Path):
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(f"{path}/tokenizer")
        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(f"{path}/feature_extractor")
        if self.image_processor is not None:
            self.image_processor.save_pretrained(f"{path}/image_processor")

    @classmethod
    def from_pretrained(cls, path: str | Path):
        from transformers import AutoFeatureExtractor, AutoTokenizer, VivitImageProcessor

        tokenizer = None
        feature_extractor = None
        image_processor = None
        if os.path.exists(f"{path}/tokenizer"):
            tokenizer = AutoTokenizer.from_pretrained(f"{path}/tokenizer")
        if os.path.exists(f"{path}/feature_extractor"):
            feature_extractor = AutoFeatureExtractor.from_pretrained(f"{path}/feature_extractor")
        if os.path.exists(f"{path}/image_processor"):
            image_processor = VivitImageProcessor.from_pretrained(f"{path}/image_processor")
        return cls(tokenizer, feature_extractor, image_processor)
