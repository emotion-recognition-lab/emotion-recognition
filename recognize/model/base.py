from __future__ import annotations

import hashlib
import itertools
import json
import os
import pickle
import zlib
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import Generic, Literal, Self, overload

import torch
from faster_whisper import WhisperModel
from loguru import logger
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft.peft_model import PeftModel
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import load_file, save, save_file
from torch import nn
from torch.nn import CrossEntropyLoss

from recognize.cache import load_cached_tensors, save_cached_tensors
from recognize.module.basic import Pooler
from recognize.typing import BackboneT, ModelInputT, StateDicts


class ModelOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ClassifierOutput(ModelOutput):
    logits: torch.Tensor
    pooler_output: tuple[torch.Tensor | None, ...] = Field(default_factory=tuple)
    loss: torch.Tensor | None = None


class ModelInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    whisper_model: WhisperModel | None = None
    unique_ids: list[str] | None = None  # TODO: maybe better name

    def cuda(self) -> Self:
        for field in self.model_fields.keys():
            field_value = getattr(self, field)
            if isinstance(field_value, torch.Tensor):
                setattr(self, field, field_value.cuda())
        return self

    def pin_memory(self) -> Self:
        for field in self.model_fields.keys():
            field_value = getattr(self, field)
            if isinstance(field_value, torch.Tensor):
                setattr(self, field, field_value.pin_memory().cuda())
        return self

    def __getitem__(self, index: int | list[int] | slice) -> Self:
        raise NotImplementedError("__getitem__ method must be implemented in subclass")

    @staticmethod
    def merge(batch: Sequence[ModelInputT], attr_name: str):
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
            self.whisper_model = WhisperModel(
                "medium",
                device="cuda",
                compute_type="float16",
                download_root=os.environ.get("WHISPER_DOWNLOAD_ROOT", None),
            )
        segments, _ = self.whisper_model.transcribe(audio_path, language="zh", initial_prompt="简体")
        text = "。".join(seg.text for seg in segments)
        logger.info(f"Recognized text: {text}")
        return text


class Backbone(nn.Module, Generic[ModelInputT]):
    def __init__(
        self,
        # TODO: change to Mapping[str, tuple[nn.Module, int]]
        encoders: Mapping[str, tuple[nn.Module | None, int]],
        use_cache: bool = True,
        is_frozen: bool = True,
        use_peft: bool = False,
        encoder_dir: str | Path = "./checkpoints/training/backbones",
    ):
        super().__init__()
        self.encoder_dir = Path(encoder_dir)
        self.feature_sizes = {name: feature_size for name, (_, feature_size) in encoders.items()}
        self.use_cache = use_cache
        self.use_peft = use_peft
        self.is_frozen = is_frozen
        self.encoders = nn.ModuleDict(
            {name: self.pretrained_module(module) for name, (module, _) in encoders.items() if module is not None}
        )
        self.poolers = nn.ModuleDict(
            {
                name: Pooler(module.config.hidden_size, feature_size)
                for name, (module, feature_size) in encoders.items()
                if module is not None
            }
        )

    @property
    def hyperparameter(self):
        return {
            "feature_sizes": self.feature_sizes,
        }

    def freeze(self):
        self.is_frozen = True

    def unfreeze(self):
        self.is_frozen = False

    def compute_embs(self, inputs: ModelInputT) -> dict[str, torch.Tensor | None]:
        raise NotImplementedError("compute_embs method must be implemented in subclass")

    def pool_embs(
        self,
        embs_dict: dict[str, torch.Tensor | None],
    ) -> dict[str, torch.Tensor]:
        return {name: self.poolers[name](embs) for name in self.poolers if (embs := embs_dict.get(name)) is not None}

    def load_cache(self, inputs: ModelInputT) -> tuple[list[dict[str, torch.Tensor]], list[int]]:
        assert inputs.unique_ids is not None, "unique_ids must be a list"
        cached_list, _, no_cached_index_list = load_cached_tensors(inputs.unique_ids, cache_dir=f"./cache/{self.hash}")
        return cached_list, no_cached_index_list

    def save_cache(self, no_cached_inputs: ModelInputT) -> None:
        assert no_cached_inputs.unique_ids is not None
        with torch.no_grad():
            pooler_output = self.direct_forward(no_cached_inputs)
        save_cached_tensors(
            no_cached_inputs.unique_ids,
            dict(pooler_output.items()),
            cache_dir=f"./cache/{self.hash}",
        )

    def merge_cache(self, cached_list: list[dict[str, torch.Tensor]], device: torch.device) -> dict[str, torch.Tensor]:
        embs_list_dict: dict[str, list[torch.Tensor]] = {}
        for cache in cached_list:
            for k, v in cache.items():
                if k not in embs_list_dict:
                    embs_list_dict[k] = []
                embs_list_dict[k].append(v)
        embs_dict: dict[str, torch.Tensor] = {k: torch.stack(v).to(device) for k, v in embs_list_dict.items()}
        return embs_dict

    def direct_forward(self, inputs: ModelInputT) -> dict[str, torch.Tensor]:
        embs_tuple = self.compute_embs(inputs)
        pooler_output = self.pool_embs(embs_tuple)
        return pooler_output

    def cached_forward(self, inputs: ModelInputT) -> dict[str, torch.Tensor]:
        cached_list, no_cached_index_list = self.load_cache(inputs)
        if len(no_cached_index_list) != 0:
            no_cached_inputs = inputs[no_cached_index_list]
            self.save_cache(no_cached_inputs)
            cached_list, no_cached_index_list = self.load_cache(inputs)
        assert len(no_cached_index_list) == 0, "All tensors should be cached"

        return self.merge_cache(cached_list, inputs.device)

    def forward(self, inputs: ModelInputT):
        if self.is_frozen and self.use_cache and inputs.unique_ids is not None:
            return self.cached_forward(inputs)
        else:
            return self.direct_forward(inputs)

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
    ) -> nn.Module:
        model = get_peft_model(
            model,  # type: ignore
            LoraConfig(
                target_modules=[n for n, m in model.named_modules() if type(m) in [nn.Linear, nn.Embedding, nn.Conv2d]],
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
            module = self.encoders[name]
            if isinstance(module, PeftModel):
                set_peft_model_state_dict(module, state_dict)
            else:
                module.load_state_dict(state_dict)

    def get_state_dicts(self) -> tuple[StateDicts, StateDicts]:
        state_dicts: dict[str, dict[str, torch.Tensor]] = {}
        peft_state_dicts: dict[str, dict[str, torch.Tensor]] = {}
        for name, module in self.encoders.items():
            if isinstance(module, PeftModel):
                peft_state_dicts[name] = get_peft_model_state_dict(
                    module,
                    save_embedding_layers=False,  # type: ignore
                )
                state_dicts[name] = module.get_base_model().state_dict()
            else:
                state_dicts[name] = module.state_dict()
        return state_dicts, peft_state_dicts

    @cached_property
    def hash(self):
        if not self.is_frozen:
            self.is_frozen = True
            logger.warning("Model is not frozen, freezing model for hashing")
        bytes_dict: dict[str, bytes] = {}
        state_dicts, peft_state_dicts = self.get_state_dicts()
        for name, state_dict in itertools.chain(state_dicts.items(), peft_state_dicts.items()):
            bytes_dict[name] = save(state_dict)
        serialized_dict = pickle.dumps(bytes_dict)
        return f"{zlib.crc32(serialized_dict):08x}"

    def save(self, original_encoder_dir: Path) -> dict[str, Path]:
        state_path_dict: dict[str, Path] = {}
        self.encoder_dir.mkdir(parents=True, exist_ok=True)
        state_dicts, peft_state_dicts = self.get_state_dicts()
        for name, state_dict in state_dicts.items():
            self.encoders[name].config.save_pretrained(original_encoder_dir / name)
            state_bytes = save(state_dict)
            hasher = hashlib.md5()
            hasher.update(state_bytes)
            model_hash = hasher.hexdigest()
            model_path = Path(f"{self.encoder_dir}/{model_hash}.safetensors")
            state_path_dict[name] = model_path.absolute()
            if model_path.exists():
                continue
            with open(model_path, "wb") as f:
                f.write(state_bytes)
        for name, state_dict in peft_state_dicts.items():
            state_bytes = save(state_dict)
            hasher = hashlib.md5()
            hasher.update(state_bytes)
            model_hash = hasher.hexdigest()
            model_path = Path(f"{self.encoder_dir}/peft_{model_hash}.safetensors")
            state_path_dict[f"peft_{name}"] = model_path.absolute()
            if model_path.exists():
                continue
            with open(model_path, "wb") as f:
                f.write(state_bytes)

        with open(original_encoder_dir / "config.json", "w") as f:
            json.dump(self.hyperparameter, f)
        save_file(self.poolers.state_dict(), original_encoder_dir / "pooler.safetensors")
        return state_path_dict

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        use_cache: bool = True,
        is_frozen: bool = True,
        use_peft: bool = False,
        encoder_dir: str | Path = "./checkpoints/training/backbones",
    ) -> Self:
        from transformers import AutoConfig, AutoModel

        checkpoint_path = Path(checkpoint_path)
        backbones: dict[str, tuple[nn.Module, int]] = {}
        state_dicts: dict[str, dict[str, torch.Tensor]] = {}

        with open(checkpoint_path / "config.json") as f:
            backbone_config = json.load(f)

        for name, feature_size in backbone_config["feature_sizes"].items():
            if not (checkpoint_path / name).exists():
                logger.warning(f"{name} not found in {checkpoint_path}")
                continue
            config = AutoConfig.from_pretrained(checkpoint_path / name / "config.json")
            model = AutoModel.from_config(config)
            state_dicts[name] = load_file(checkpoint_path / f"{name}.safetensors")
            backbones[name] = (model, feature_size)
        self = cls(
            backbones,
            use_cache=use_cache,
            is_frozen=is_frozen,
            use_peft=use_peft,
            encoder_dir=encoder_dir,
        )
        self.set_state_dicts(state_dicts)
        self.poolers.load_state_dict(load_file(checkpoint_path / "pooler.safetensors"))
        return self


class ClassifierModel(nn.Module, Generic[BackboneT]):
    __call__: Callable[..., ClassifierOutput]

    def __init__(
        self,
        backbone: BackboneT,
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
            loss_fn = nn.MSELoss()
            return loss_fn(logits.view(-1), labels.float())
        else:
            loss_fn = CrossEntropyLoss(weight=self.sample_weights)
            return loss_fn(logits, labels)

    def classify(self, features: torch.Tensor, labels: torch.Tensor | None) -> ClassifierOutput:
        logits = self.classifier(features)
        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels)
        return ClassifierOutput(logits=logits, loss=loss)

    def get_hyperparameter(self):
        return {
            "num_classes": self.num_classes,
            "feature_size": self.feature_size,
        }

    @property
    def hyperparameter(self):
        return self.get_hyperparameter()

    def save_checkpoint(self, checkpoint_path: str | Path):
        checkpoint_path = Path(checkpoint_path)
        model_state_dict = {key: value for key, value in self.state_dict().items() if not key.startswith("backbone.")}
        save_file(model_state_dict, checkpoint_path / "model.safetensors")
        with open(checkpoint_path / "config.json", "w") as f:
            json.dump(self.hyperparameter, f)
