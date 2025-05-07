from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, ClassVar

import torch
from loguru import logger

from recognize.typing import DatasetLabelType

from .base import MultimodalDataset

if TYPE_CHECKING:
    from recognize.model import MultimodalBackbone
    from recognize.typing import DatasetSplit


class IEMOCAPDataset(MultimodalDataset):
    emotion_class_names_mapping: ClassVar[dict[str, int]] = {
        "neu": 0,
        "ang": 1,
        "exc": 2,
        "fru": 3,
        "hap": 4,
        "sad": 5,
    }

    @staticmethod
    def label2int(item):
        emotion = item["Emotion"]
        return IEMOCAPDataset.emotion_class_names_mapping[emotion]

    def __init__(
        self,
        dataset_path: str,
        *,
        split: DatasetSplit = "train",
        label_type: DatasetLabelType = "emotion",
    ):
        import pandas as pd

        if label_type != "emotion":
            logger.warning("IEMOCAP only supports emotion label type")

        if split == "dev":
            logger.warning("DEV split is deprecated. Using VALID split instead.")
        elif split == "valid":
            split = "dev"
        self.split = split
        self.label_type = "emotion"

        super().__init__(
            dataset_path,
            pd.read_csv(f"{dataset_path}/IEMOCAP_{self.split}.csv", sep=",", header=0),
            num_classes=6,
            split=split,
        )

        self._generate_session()

    def _generate_session(self):
        unique_speakers = self.meta["Speaker"].unique().tolist()
        renamed_speakers = [f"<s{i}>" for i in range(len(unique_speakers))]
        speaker_map = dict(zip(unique_speakers, renamed_speakers, strict=True))
        self.meta["Speaker"] = self.meta["Speaker"].map(speaker_map)

        self.meta["Session"] = ""
        for _, group in self.meta.groupby("Dialogue_ID"):
            session_parts = []
            for idx, row in group.iterrows():
                session_parts.append(f"{row['Speaker']} {row['Utterance']}")
                self.meta.at[idx, "Session"] = " ".join(session_parts)

    def special_process(self, backbone: MultimodalBackbone):
        assert self.preprocessor is not None, "Preprocessor is not set"
        tokenizer = self.preprocessor.tokenizer
        if tokenizer is None:
            return
        text_backbone = backbone.named_encoders["T"]
        tokenizer.add_special_tokens({"additional_special_tokens": self.speakers})
        text_backbone.resize_token_embeddings(len(tokenizer))

    @cached_property
    def speakers(self) -> list[str]:
        return self.meta["Speaker"].unique().tolist()

    def __getitem__(self, index: int):
        from recognize.model import LazyMultimodalInput

        item = self.meta.iloc[index]
        label = self.label2int(item)

        speaker = item["Speaker"]
        session = item["Session"]
        text = f"{session} </s> Now {speaker} feels"
        audio_path = item["Wav_Path"]
        video_path = item["Video_Path"]

        assert self.preprocessor is not None, "Preprocessor is not set"
        return LazyMultimodalInput(
            preprocessor=self.preprocessor,
            texts=[text],
            audio_paths=[f"{self.dataset_path}/{audio_path}"],
            video_paths=[f"{self.dataset_path}/{video_path}"],
            labels=torch.tensor([label], dtype=torch.int64),
        )

    def __len__(self):
        return len(self.meta)
