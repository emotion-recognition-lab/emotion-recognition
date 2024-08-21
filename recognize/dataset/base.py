from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

from torch.utils.data import Dataset

if TYPE_CHECKING:
    from pandas import DataFrame

    from recognize.model import MultimodalInput
    from recognize.preprocessor import Preprocessor


class DatasetSplit(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    DEV = "dev"  # meld dataset


class MultimodalDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        meta: DataFrame,
        preprocessor: Preprocessor,
        *,
        num_classes: int = 2,
        split: DatasetSplit = DatasetSplit.TRAIN,
        custom_unique_id: str = "",
        cache_mode: bool = False,
    ):
        self.dataset_path = dataset_path
        self.meta = meta

        self.preprocessor = preprocessor
        self.load_text = self.preprocessor.load_text
        self.load_audio = self.preprocessor.load_audio
        self.load_video = self.preprocessor.load_video

        self.num_classes = num_classes
        self.split = split.value

        self.custom_unique_id = custom_unique_id
        self.cache_mode = cache_mode

    @abstractmethod
    def __getitem__(self, index: int) -> MultimodalInput: ...

    def __len__(self):
        return len(self.meta)
