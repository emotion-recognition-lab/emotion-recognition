from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

from loguru import logger
from pydantic import BaseModel, model_validator
from typing_extensions import deprecated

from .typing import LogLevel, ModalType


class ModelConfig(BaseModel):
    modals: list[ModalType]
    feature_sizes: list[int]
    encoders: list[str]
    freeze_backbone: bool = True
    fusion: str | None = None

    @model_validator(mode="after")
    def verify_model_config(self) -> Self:
        assert (
            len(self.modals) == len(self.feature_sizes) == len(self.encoders)
        ), "Number of modals, feature_sizes and encoders should be the same"
        return self

    @property
    @deprecated("Use `encoders` instead")
    def backbones(self) -> list[str]:
        return self.encoders


class DatasetConfig(BaseModel):
    path: Path
    dataset_class: str  # TODO: use Literal
    label_type: Literal["sentiment", "emotion"] = "sentiment"


class TrainingConfig(BaseModel):
    log_level: LogLevel = "INFO"
    batch_size: int = 32

    model: ModelConfig
    dataset: DatasetConfig

    def generate_model_label(self):
        model_label = "+".join(self.model.modals)
        if self.dataset.label_type == "sentiment":
            model_label += "--S"
        else:
            model_label += "--E"
        if self.model.freeze_backbone:
            model_label += "F"
        else:
            model_label += "T"
        return model_label


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
            import toml

            config = toml.load(f)
        elif path.suffix == ".yaml":
            import yaml
            from yaml import FullLoader

            config = yaml.load(f, Loader=FullLoader)
        else:
            raise NotImplementedError(f"Unsupportted suffix {path.suffix}")

    return config


def save_dict_to_path(config: dict[str, Any], path: Path) -> None:
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
            import toml

            toml.dump(config, f)
        elif path.suffix == ".yaml":
            import yaml

            yaml.dump(config, f)
        else:
            raise NotImplementedError(f"Unsupportted suffix {path.suffix}")


def load_training_config(path: str | Path) -> TrainingConfig:
    config_dict = load_dict_from_path(Path(path))
    return TrainingConfig(**config_dict)


def load_inference_config(path: str | Path) -> InferenceConfig:
    config_dict = load_dict_from_path(Path(path))
    return InferenceConfig(**config_dict)


def save_config(config: BaseModel, path: str | Path) -> None:
    config_dict = config.model_dump(mode="json")
    save_dict_to_path(config_dict, Path(path))
