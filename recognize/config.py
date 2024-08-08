from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import deprecated

from .typing import LogLevel, ModalType


class ModelConfig(BaseModel):
    modals: list[ModalType]
    feature_sizes: list[int]
    encoders: list[str]
    freeze_backbone: bool = True
    fusion: str | None = None

    @property
    @deprecated("Use `encoders` instead")
    def backbones(self) -> list[str]:
        return self.encoders


class DatasetConfig(BaseModel):
    path: Path = Path("datasets/MELD")
    label_type: Literal["sentiment", "emotion"] = "sentiment"


class Config(BaseModel):
    log_level: LogLevel = "INFO"
    model: ModelConfig
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)

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


def load_config(path: str | Path) -> Config:
    config_dict = load_dict_from_path(Path(path))
    return Config(**config_dict)
