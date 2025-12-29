from __future__ import annotations

import hashlib
from abc import abstractmethod
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger
from pydantic import BaseModel
from torch import nn

from .typing import DatasetClass, DatasetLabelType, LogLevel, ModalType

if TYPE_CHECKING:
    from recognize.module import (
        AdaptivePrototypeContrastiveLoss,
        CrossModalContrastiveLoss,
        ReconstructionLoss,
        SelfContrastiveLoss,
        SupervisedProtoContrastiveLoss,
    )


def hash_string(string: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(string.encode())
    return hasher.hexdigest()[:8]


def recursive_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class ModelEncoderConfig(BaseModel):
    model: str
    feature_size: int
    checkpoint: Path | None = None

    @cached_property
    def hash(self) -> str:
        return hash_string(f"{self.model}-{self.feature_size}-{self.checkpoint}")


class ModelFusionConfig(BaseModel):
    base: str
    args: list[str]
    kwargs: dict[str, str]

    def __str__(self) -> str:
        args = ", ".join(self.args)
        kwargs = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        return f"{self.base}({args}, {kwargs})"

    @property
    def hash(self) -> str:
        kwargs = sorted(self.kwargs.items())
        return hash_string(f"{self.base}-{self.args}-{kwargs}")


class ModelConfig(BaseModel):
    encoder: dict[ModalType, ModelEncoderConfig]
    fusion: ModelFusionConfig | None = None
    num_experts: int = 1

    @cached_property
    def label(self) -> str:
        modalities = "+".join(sorted(self.encoder.keys()))
        # TODO: fusion need more explicit label
        model_labels = [
            f"{self.num_experts}xE",
            f"{modalities}",
        ]
        return "--".join(model_labels)

    @cached_property
    def hash(self) -> str:
        encoder_items = sorted(self.encoder.items())
        encoder_hash = "".join(f"{modal}:{encoder.hash}" for modal, encoder in encoder_items)
        fusion_hash = self.fusion.hash if self.fusion is not None else ""
        return hash_string(f"{encoder_hash}-{fusion_hash}")


class DatasetConfig(BaseModel):
    path: Path
    dataset_class: DatasetClass
    label_type: DatasetLabelType = "sentiment"

    @cached_property
    def label(self) -> str:
        dataset_class_mapping = {
            "MELDDataset": "MELD",
            "PilotDataset": "Pilot",
            "SIMSDataset": "SIMS",
            "IEMOCAPDataset": "IEMOCAP",
        }
        label_type_mapping = {"sentiment": "S", "emotion": "E"}

        dataset_class = dataset_class_mapping[self.dataset_class]
        label_type = label_type_mapping[self.label_type]
        model_labels = [
            dataset_class,
            label_type,
        ]
        return "--".join(model_labels)


class LossConfigItem[T](BaseModel):
    @abstractmethod
    def to_loss_object(self, num_classes: int, original_dims: Mapping[str, int], hidden_dim: int) -> T: ...


class PrototypeContrastiveConfig(LossConfigItem):
    temperature: float = 0.08


class SupervisedPrototypeContrastiveConfig(PrototypeContrastiveConfig):
    pool_size: int = 512
    support_set_size: int = 64

    @cached_property
    def label(self) -> str:
        return f"spcl{self.temperature}-{self.pool_size}-{self.support_set_size}"

    def to_loss_object(
        self, num_classes: int, original_dims: Mapping[str, int], hidden_dim: int
    ) -> SupervisedProtoContrastiveLoss:
        from recognize.module import (
            SupervisedProtoContrastiveLoss,
        )

        return SupervisedProtoContrastiveLoss(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            temp=self.temperature,
            pool_size=self.pool_size,
            support_set_size=self.support_set_size,
        )


class AdaptivePrototypeContrastiveConfig(PrototypeContrastiveConfig):
    alpha: float = 0.5
    gamma: float = 0.9

    @cached_property
    def label(self) -> str:
        return f"apcl{self.temperature}-{self.gamma}-{self.alpha}"

    def to_loss_object(
        self, num_classes: int, original_dims: Mapping[str, int], hidden_dim: int
    ) -> AdaptivePrototypeContrastiveLoss:
        from recognize.module import (
            AdaptivePrototypeContrastiveLoss,
        )

        return AdaptivePrototypeContrastiveLoss(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            temp=self.temperature,
            alpha=self.alpha,
            gamma=self.gamma,
        )


class SelfContrastiveConfig(LossConfigItem):
    hidden_dim: int = 512

    @cached_property
    def label(self) -> str:
        return f"scl{self.hidden_dim}"

    def to_loss_object(
        self, num_classes: int, original_dims: Mapping[str, int], hidden_dim: int
    ) -> SelfContrastiveLoss:
        raise NotImplementedError("SelfContrastiveLoss is not implemented")


class CrossModalContrastiveConfig(LossConfigItem):
    main_modal: ModalType

    @cached_property
    def label(self) -> str:
        return f"cmcl-{self.main_modal}"

    def to_loss_object(
        self, num_classes: int, original_dims: Mapping[str, int], hidden_dim: int
    ) -> CrossModalContrastiveLoss:
        from recognize.module import (
            CrossModalContrastiveLoss,
        )

        return CrossModalContrastiveLoss(original_dims, self.main_modal)


class ReconstructionLossConfig(LossConfigItem):
    alpha: float = 0.5

    @cached_property
    def label(self) -> str:
        return f"rec{self.alpha}"

    def to_loss_object(self, num_classes: int, original_dims: Mapping[str, int], hidden_dim: int) -> ReconstructionLoss:
        from recognize.module import (
            ReconstructionLoss,
        )

        return ReconstructionLoss(original_dims, hidden_dim, alpha=self.alpha)


class LossConfig(BaseModel):
    classification: Literal["weight", "focal"] | None = None
    sample_contrastive: SupervisedPrototypeContrastiveConfig | AdaptivePrototypeContrastiveConfig | None = None
    modal_contrastive: SelfContrastiveConfig | CrossModalContrastiveConfig | None = None
    reconstruction: ReconstructionLossConfig | None = None

    def get_loss_objects(self, num_classes: int, original_dims: Mapping[str, int], hidden_dim: int) -> list[nn.Module]:
        loss_objects = []
        if self.sample_contrastive is not None:
            loss_objects.append(self.sample_contrastive.to_loss_object(num_classes, original_dims, hidden_dim))
        if len(original_dims) == 1:
            # NOTE: some loss functions only support multiple modalities
            return loss_objects
        if self.modal_contrastive is not None:
            loss_objects.append(self.modal_contrastive.to_loss_object(num_classes, original_dims, hidden_dim))
        if self.reconstruction is not None:
            loss_objects.append(self.reconstruction.to_loss_object(num_classes, original_dims, hidden_dim))
        return loss_objects

    @cached_property
    def label(self) -> str:
        labels = []
        if self.classification is not None:
            labels.append(self.classification)
        if self.sample_contrastive is not None:
            labels.append(self.sample_contrastive.label)
        if self.modal_contrastive is not None:
            labels.append(self.modal_contrastive.label)
        if self.reconstruction is not None:
            labels.append(self.reconstruction.label)
        return "--".join(labels)


class TrainingConfig(BaseModel):
    log_level: LogLevel = "INFO"  # TODO: remove after typer supports Literal
    batch_size: int = 2
    training_mode: Literal["trainable", "lora", "frozen"] = "trainable"
    dropout_prob: float | None = None
    seed: int | None = None

    model: ModelConfig
    dataset: DatasetConfig
    loss: LossConfig | None = None

    @cached_property
    def label(self) -> str:
        training_label = f"{self.training_mode}--{self.batch_size}"
        if self.loss is not None and self.loss.label != "":
            training_label += f"--{self.loss.label}"
        if self.dropout_prob is not None:
            training_label += f"--d{self.dropout_prob}"
        return f"{self.dataset.label}/{training_label}/{self.model.label}"


class InferenceConfig(BaseModel):
    num_classes: int
    model: ModelConfig


def load_dict_from_path(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file {path} not found")
    path = path.expanduser()
    with open(path) as f:
        if path.suffix == ".json":
            import json

            config = json.load(f)
        elif path.suffix == ".toml":
            import rtoml

            config = rtoml.load(f)
        elif path.suffix == ".yaml":
            import yaml
            from yaml import FullLoader

            config = yaml.load(f, Loader=FullLoader)
        else:
            raise NotImplementedError(f"Unsupportted suffix {path.suffix}")

    return config


def save_dict_to_file(config: dict[str, Any], path: Path) -> None:
    if path.exists():
        logger.warning(f"{path} already exists, will be overwritten")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    path = path.expanduser()
    with open(path.absolute(), "w") as f:
        if path.suffix == ".json":
            import json

            json.dump(config, f)
        elif path.suffix == ".toml":
            import rtoml

            rtoml.dump(config, f, none_value=None)
        elif path.suffix == ".yaml":
            import yaml

            yaml.dump(config, f)
        else:
            raise NotImplementedError(f"Unsupportted suffix {path.suffix}")


def load_training_config(*paths: Path, batch_size: int | None = None, seed: int | None = None) -> TrainingConfig:
    config_dict = {}
    for path in paths:
        recursive_update(config_dict, load_dict_from_path(path))
    config = TrainingConfig(**config_dict)
    if batch_size is not None:
        config.batch_size = batch_size
    if seed is not None:
        config.seed = seed
    return config


def load_inference_config(*paths: Path) -> InferenceConfig:
    config_dict = {}
    for path in paths:
        recursive_update(config_dict, load_dict_from_path(path))
    return InferenceConfig(**config_dict)


def save_config(config: BaseModel, path: Path) -> None:
    config_dict = config.model_dump(mode="json")
    save_dict_to_file(config_dict, path)
