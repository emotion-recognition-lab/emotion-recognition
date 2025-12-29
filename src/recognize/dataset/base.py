from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar

from torch.utils.data import Dataset

if TYPE_CHECKING:
    from pandas import DataFrame

    from recognize.model import MultimodalBackbone, MultimodalInput
    from recognize.preprocessor import Preprocessor
    from recognize.typing import DatasetSplit


class MultimodalDataset(Dataset):
    dataset_path: str
    meta: DataFrame
    num_classes: int
    split: DatasetSplit
    preprocessor: Preprocessor | None

    emotion_class_names_mapping: ClassVar[dict[str, int]]
    sentiment_class_names_mapping: ClassVar[dict[str, int]]

    def __init__(
        self,
        dataset_path: str,
        meta: DataFrame,
        *,
        num_classes: int = 2,
        split: DatasetSplit = "train",
    ):
        self.dataset_path = dataset_path
        self.meta = meta

        self.num_classes = num_classes
        self.split = split

        self.preprocessor = None

    def set_preprocessor(self, preprocessor: Preprocessor):
        self.preprocessor = preprocessor

    @abstractmethod
    def special_process(self, backbone: MultimodalBackbone): ...

    @staticmethod
    @abstractmethod
    def label2int(item) -> int: ...

    @cached_property
    def class_weights(self) -> list[float]:
        labels = self.meta.apply(self.label2int, axis=1)
        class_counts = [(labels == i).sum() for i in range(self.num_classes)]
        total_samples = sum(class_counts)
        class_weights = [
            (total_samples / (self.num_classes * class_count)) if class_count > 0 else 0.0
            for class_count in class_counts
        ]
        weight_sum = sum(class_weights)
        if weight_sum > 0:
            class_weights = [weight * self.num_classes / weight_sum for weight in class_weights]
        return class_weights

    @abstractmethod
    def __getitem__(self, index: int) -> MultimodalInput: ...

    def __len__(self):
        return len(self.meta)
