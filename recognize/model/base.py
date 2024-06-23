from __future__ import annotations

import hashlib
import json
import os
from functools import cached_property
from pathlib import Path
from typing import Callable, Literal, TypeVar, overload

import torch
import whisperx
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft.peft_model import PeftModel
from pydantic import BaseModel, ConfigDict
from safetensors.torch import load_file, save, save_file
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel
from whisperx.asr import FasterWhisperPipeline

ModelInputT = TypeVar("ModelInputT", bound="ModelInput")
ClassifierModelT = TypeVar("ClassifierModelT", bound="ClassifierModel")


class ModelOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ClassifierOutput(ModelOutput):
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class ModelInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    whisper_model: FasterWhisperPipeline | None = None

    def cuda(self):
        for field in self.model_fields.keys():
            field_value = getattr(self, field)
            if isinstance(field_value, torch.Tensor):
                setattr(self, field, field_value.cuda())
        return self

    def pin_memory(self):
        for field in self.model_fields.keys():
            field_value = getattr(self, field)
            if isinstance(field_value, torch.Tensor):
                setattr(self, field, field_value.pin_memory().cuda())
        return self

    @staticmethod
    def merge(batch: list[ModelInputT], attr_name: str):
        attr: list[torch.Tensor] = []
        for item in batch:
            if getattr(item, attr_name) is not None:
                attr.append(getattr(item, attr_name))
            else:
                return None
        return attr

    @property
    def device(self) -> torch.device:
        raise NotImplementedError("device property must be implemented in subclass")

    def recoginize_audio(self, audio_path: str) -> str:
        if self.whisper_model is None:
            # use self.device
            self.whisper_model = whisperx.load_model("medium", device="cuda")
        result = self.whisper_model.transcribe(audio_path)
        text = "。".join(seg["text"] for seg in result["segments"])
        return text


class Pooler(nn.Module):
    def __init__(self, in_features: int, out_features: int | None = None, bias: bool = True):
        super().__init__()
        if out_features is None:
            out_features = in_features
        self.pool = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(in_features, out_features, bias=bias), nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled_output = self.pool(x)
        return pooled_output


class Backbone(nn.Module):
    def __init__(
        self,
        *backbones: nn.Module | None,
        use_cache: bool = True,
        is_frozen: bool = True,
        use_peft: bool = False,
        backbones_dir: str | Path = "./checkpoints/backbones",
    ):
        # Hidden size of text, audio and video backbones must be the same
        output_size: int | None = None
        for backbone in backbones:
            if backbone is not None:
                if output_size is not None and output_size != backbone.config.hidden_size:
                    raise ValueError(
                        "Hidden size of text, audio and video backbones must be the same"
                    )
                output_size = backbone.config.hidden_size

        if output_size is None:
            raise ValueError(
                "output_size must be provided if text_backbone, audio_backbone and video_backbone are None"
            )

        super().__init__()
        self.backbones = backbones
        self.output_size = output_size
        self.use_cache = use_cache
        self.use_peft = use_peft
        self.is_frozen = is_frozen
        self.backbones_dir = Path(backbones_dir)

    def freeze(self):
        self.is_frozen = True

    def unfreeze(self):
        self.is_frozen = False

    @overload
    def pretrained_module(self, module: nn.Module) -> nn.Module: ...

    @overload
    def pretrained_module(self, module: None) -> None: ...

    def pretrained_module(self, module: nn.Module | None) -> nn.Module | None:
        if module is None:
            return None
        if self.use_peft:
            module = self.apply_peft(module)
        module_forward = module.forward

        def forward(*args, **kwargs):
            if self.is_frozen:
                with torch.no_grad():
                    return module_forward(*args, **kwargs)
            else:
                return module_forward(*args, **kwargs)

        module.forward = forward
        return module

    @staticmethod
    def apply_peft(
        model: nn.Module,
        *,
        rank: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.1,
        bias: Literal["none", "all", "lora_only"] = "all",
    ):
        model = get_peft_model(
            model,  # type: ignore  # noqa: PGH003
            LoraConfig(
                target_modules=[
                    n
                    for n, m in model.named_modules()
                    if type(m) in [nn.Linear, nn.Embedding, nn.Conv2d]
                ],
                inference_mode=False,
                r=rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=bias,
            ),
        )
        model.print_trainable_parameters()
        return model

    def set_state_dicts(self, state_dicts: dict[str, dict[str, torch.Tensor]]):
        for name, state_dict in state_dicts.items():
            module: nn.Module = getattr(self, name)
            if isinstance(module, PeftModel):
                set_peft_model_state_dict(module, state_dict)
            else:
                module.load_state_dict(state_dict)

    def get_state_dicts(self):
        state_dicts: dict[str, dict[str, torch.Tensor]] = {}
        peft_state_dicts: dict[str, dict[str, torch.Tensor]] = {}
        for name, module in self.named_modules():
            if not hasattr(self, name):
                continue
            if isinstance(module, PeftModel):
                peft_state_dicts[name] = get_peft_model_state_dict(
                    module,
                    save_embedding_layers=False,  # type: ignore  # noqa: PGH003
                )
                state_dicts[name] = module.get_base_model().state_dict()
            else:
                state_dicts[name] = module.state_dict()
        return state_dicts, peft_state_dicts

    def save(self):
        hash_dict: dict[str, Path] = {}
        self.backbones_dir.mkdir(parents=True, exist_ok=True)
        state_dicts, peft_state_dicts = self.get_state_dicts()
        for name, state_dict in state_dicts.items():
            state_bytes = save(state_dict)
            hasher = hashlib.md5()
            hasher.update(state_bytes)
            model_hash = hasher.hexdigest()
            model_path = Path(f"{self.backbones_dir}/{model_hash}.safetensors")
            hash_dict[name] = model_path.absolute()
            if model_path.exists():
                continue
            with open(model_path, "wb") as f:
                f.write(state_bytes)
        # TODO: peft
        # for name, state_dict in peft_state_dicts.items():
        #     state_bytes = save(state_dict)
        #     path = Path(f"{self.backbones_dir}/{model_hash}.safetensors")
        #     module.base_model.config.save_pretrained("name")

        #     hash_dict[name] = path.absolute()
        #     if path.exists():
        #         continue
        #     with open(path, "wb") as f:
        #         f.write(state_bytes)

        return hash_dict

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        use_cache: bool = True,
        is_frozen: bool = True,
        use_peft: bool = False,
        backbones_dir: str | Path = "./checkpoints/backbones",
    ):
        checkpoint_path = Path(checkpoint_path)
        backbones: list[nn.Module] = []
        for name in os.listdir(checkpoint_path):
            with open(checkpoint_path / name / "config.json", "r") as f:
                config = json.load(f)
            model = AutoModel.from_config(config)
            model.load_state_dict(load_file(checkpoint_path / f"{name}.safetensors"))
            backbones.append(model)

        return cls(
            *backbones,
            use_cache=use_cache,
            is_frozen=is_frozen,
            use_peft=use_peft,
            backbones_dir=backbones_dir,
        )


class ClassifierModel(nn.Module):
    __call__: Callable[..., ClassifierOutput]

    def __init__(
        self,
        backbone: Backbone,
        feature_size: int,
        num_classes: int,
        *,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.feature_size = feature_size
        self.hidden_size = feature_size  # for compatibility with old code
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_size, num_classes),
        )

    def freeze_backbone(self):
        self.backbone.freeze()

    def unfreeze_backbone(self):
        self.backbone.unfreeze()

    @cached_property
    def sample_weights(self):
        return 1 / self.class_weights if self.class_weights is not None else None

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 1:
            loss_fct = nn.MSELoss()
            return loss_fct(logits.view(-1), labels.float())
        else:
            loss_fct = CrossEntropyLoss(weight=self.sample_weights)
            return loss_fct(logits, labels)

    def classify(self, features: torch.Tensor, labels: torch.Tensor | None) -> ClassifierOutput:
        logits = self.classifier(features)
        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels)
        return ClassifierOutput(logits=logits.detach(), loss=loss)

    def get_hyperparameter(self):
        return {
            "num_classes": self.num_classes,
            "feature_size": self.feature_size,
        }

    @cached_property
    def hyperparameter(self):
        return self.get_hyperparameter()

    def save_checkpoint(self, checkpoint_path: str | Path):
        checkpoint_path = Path(checkpoint_path)
        model_state_dict = {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("backbone.")
        }
        save_file(model_state_dict, checkpoint_path / "model.safetensors")
        with open(checkpoint_path / "config.json", "w") as f:
            json.dump(self.hyperparameter, f)
