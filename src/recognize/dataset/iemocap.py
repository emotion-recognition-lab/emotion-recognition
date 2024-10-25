from __future__ import annotations

from typing import TYPE_CHECKING

from recognize.typing import DatasetLabelType

from .base import MultimodalDataset

if TYPE_CHECKING:
    from recognize.typing import DatasetSplit


class IEMOCAPDataset(MultimodalDataset):
    def __init__(
        self,
        dataset_path: str,
        *,
        split: DatasetSplit = "train",
        label_type: DatasetLabelType = "emotion",
    ):
        raise NotImplementedError
