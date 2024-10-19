from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Self

from loguru import logger
from pydantic import BaseModel, Field, model_validator

from .typing import LogLevel, ModalType


def recursive_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class ModelConfig(BaseModel):
    modals: list[ModalType]
    feature_sizes: list[int]
    encoders: list[str]
    training_mode: Literal["trainable", "lora", "frozen"] = "trainable"
    fusion: str | None = None
    num_experts: int = 1

    @model_validator(mode="after")
    def verify_model_config(self) -> Self:
        assert (
            len(self.modals) == len(self.feature_sizes) == len(self.encoders)
        ), "Number of modals, feature_sizes and encoders should be the same"
        return self


class DatasetConfig(BaseModel):
    path: Path
    dataset_class: Literal["MELDDataset", "PilotDataset", "SIMSDataset"]
    label_type: Literal["sentiment", "emotion"] = "sentiment"


class TrainingConfig(BaseModel):
    log_level: LogLevel = "DEBUG"  # TODO: remove after typer supports Literal
    batch_size: int = 32
    custom_label: str | None = Field(default=None, description="To mark the model, e.g. MoE")

    model: ModelConfig
    dataset: DatasetConfig

    @cached_property
    def model_label(self) -> str:
        training_mode_mapping = {"trainable": "T", "lora": "L", "frozen": "F"}

        modals = "+".join(self.model.modals)
        training_mode = training_mode_mapping[self.model.training_mode]
        model_labels = [
            modals,
            training_mode,
        ]
        if self.custom_label:
            model_labels.insert(0, self.custom_label)
        return "--".join(model_labels)

    @cached_property
    def dataset_label(self) -> str:
        dataset_class_mapping = {"MELDDataset": "MELD", "PilotDataset": "Pilot", "SIMSDataset": "SIMS"}
        label_type_mapping = {"sentiment": "S", "emotion": "E"}

        dataset_class = dataset_class_mapping[self.dataset.dataset_class]
        label_type = label_type_mapping[self.dataset.label_type]
        model_labels = [
            dataset_class,
            label_type,
        ]
        return "--".join(model_labels)


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


def load_training_config(*paths: Path) -> TrainingConfig:
    config_dict = {}
    for path in paths:
        recursive_update(config_dict, load_dict_from_path(path))
    return TrainingConfig(**config_dict)


def load_inference_config(*paths: Path) -> InferenceConfig:
    config_dict = {}
    for path in paths:
        recursive_update(config_dict, load_dict_from_path(path))
    return InferenceConfig(**config_dict)


def save_config(config: BaseModel, path: Path) -> None:
    config_dict = config.model_dump(mode="json")
    save_dict_to_file(config_dict, path)
