from __future__ import annotations

import hashlib
from abc import abstractmethod
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger
from pydantic import BaseModel

from .typing import DatasetClass, DatasetLabelType, LogLevel, ModalType

if TYPE_CHECKING:
    from recognize.module import (
        AdaptivePrototypeContrastiveLoss,
        CrossModalContrastiveLoss,
        PrototypeContrastiveLoss,
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


class ModelConfig(BaseModel):
    encoder: dict[ModalType, ModelEncoderConfig]
    fusion: str | None = None
    num_experts: int = 1

    @cached_property
    def label(self) -> str:
        modals = "+".join(self.encoder.keys())
        # TODO: fusion need more explicit label
        model_labels = [
            f"{self.num_experts}xE",
            f"{modals}",
        ]
        return "--".join(model_labels)

    @cached_property
    def hash(self) -> str:
        encoder_hash = "".join(encoder.hash for encoder in self.encoder.values())
        return hash_string(f"{encoder_hash}-{self.fusion}")


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


class PrototypeContrastiveConfig(BaseModel):
    temperature: float = 0.08

    @abstractmethod
    def to_loss_object(self, num_classes: int, hidden_dim: int) -> PrototypeContrastiveLoss: ...


class SupervisedPrototypeContrastiveConfig(PrototypeContrastiveConfig):
    pool_size: int = 512
    support_set_size: int = 64

    @cached_property
    def label(self) -> str:
        return f"spcl{self.temperature}-{self.pool_size}-{self.support_set_size}"

    def to_loss_object(self, num_classes: int, hidden_dim: int) -> SupervisedProtoContrastiveLoss:
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
    beta: float = 0.1
    gamma: float = 0.1

    @cached_property
    def label(self) -> str:
        return f"apcl{self.temperature}-{self.beta}-{self.gamma}"

    def to_loss_object(self, num_classes: int, hidden_dim: int) -> AdaptivePrototypeContrastiveLoss:
        from recognize.module import (
            AdaptivePrototypeContrastiveLoss,
        )

        return AdaptivePrototypeContrastiveLoss(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            temp=self.temperature,
            beta=self.beta,
            gamma=self.gamma,
        )


class SelfContrastiveConfig(BaseModel):
    hidden_dim: int = 512

    @cached_property
    def label(self) -> str:
        return f"scl{self.hidden_dim}"

    def to_loss_object(self, feature_sizes_dict: Mapping[str, int]) -> SelfContrastiveLoss:
        raise NotImplementedError


class CrossModalContrastiveConfig(BaseModel):
    main_modal: ModalType

    @cached_property
    def label(self) -> str:
        return f"cmcl-{self.main_modal}"

    def to_loss_object(self, feature_sizes_dict: Mapping[str, int]) -> CrossModalContrastiveLoss:
        from recognize.module import (
            CrossModalContrastiveLoss,
        )

        return CrossModalContrastiveLoss(feature_sizes_dict, self.main_modal)


class LossConfig(BaseModel):
    # reweight_loss: bool = True

    sample_contrastive: SupervisedPrototypeContrastiveConfig | AdaptivePrototypeContrastiveConfig | None = None
    modal_contrastive: SelfContrastiveConfig | CrossModalContrastiveConfig | None = None

    @cached_property
    def label(self) -> str:
        labels = []
        if self.sample_contrastive is not None:
            labels.append(self.sample_contrastive.label)
        if self.modal_contrastive is not None:
            labels.append(self.modal_contrastive.label)
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
