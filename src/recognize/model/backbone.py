from __future__ import annotations

import itertools
import pickle
from abc import abstractmethod
from collections.abc import Callable, Mapping
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

import torch
from loguru import logger
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft.peft_model import PeftModel
from safetensors.torch import load_file, save, save_file
from torch import nn

from recognize.cache import CacheManager, hash_bytes, load_cached_tensors, save_cached_tensors
from recognize.config import load_dict_from_path, save_dict_to_file
from recognize.module import Projector
from recognize.typing import StateDicts

from .inputs import ModelInput, MultimodalInput


class Backbone[T: ModelInput](nn.Module):
    __call__: Callable[[T], dict[str, torch.Tensor]]

    @abstractmethod
    def forward(self, inputs: T) -> dict[str, torch.Tensor]: ...


class MultimodalBackbone(Backbone[MultimodalInput]):
    def __init__(
        self,
        encoders: Mapping[str, tuple[nn.Module, int]],
        *,
        use_cache: bool = True,
        frozen_encoders: bool = True,
        use_peft: bool = False,
        encoder_dir: Path = Path("./checkpoints/encoders"),
        init_hook: Callable[[Self], None] | None = None,
    ):
        super().__init__()
        self.encoder_dir = encoder_dir
        self.feature_sizes = {name: feature_size for name, (_, feature_size) in encoders.items()}
        self.use_cache = use_cache
        self.use_peft = use_peft
        self.frozen_encoders = frozen_encoders
        self.named_encoders = nn.ModuleDict({name: module for name, (module, _) in encoders.items()})
        self.named_poolers = nn.ModuleDict(
            {
                name: Projector(module.config.hidden_size, feature_size)
                for name, (module, feature_size) in encoders.items()
            }
        )
        self.cache_manager = CacheManager((8 * 2**30, 4 * 2**30), cache_dir=Path(f"./cache/{self.encoder_hash}"))

        if init_hook is not None:
            init_hook(self)
        if self.use_peft:
            self.named_encoders = nn.ModuleDict(
                {name: self.apply_peft(module) for name, (module, _) in encoders.items()}
            )

    def get_meta_info(self):
        return {
            "feature_sizes": self.feature_sizes,
        }

    def freeze_modal(self, modal: str):
        if modal not in self.named_encoders:
            logger.warning(f"Modal {modal} not found in {self.named_encoders.keys()}")
            return
        module = self.named_encoders[modal]
        module.requires_grad_(False)

    def freeze(self):
        self.frozen_encoders = True
        for module in self.named_encoders.values():
            module.requires_grad_(False)

    def unfreeze(self):
        self.frozen_encoders = False
        for module in self.named_encoders.values():
            module.requires_grad_(True)

    def compute_embs(self, inputs: MultimodalInput) -> dict[str, torch.Tensor]:
        embs_dict: dict[str, torch.Tensor] = {}
        if "T" in self.named_encoders and inputs.text_input_ids is not None:
            text_outputs = self.named_encoders["T"](inputs.text_input_ids, attention_mask=inputs.text_attention_mask)
            # -1 corresponds to mask_token_id
            text_embs = text_outputs.last_hidden_state[:, -1]
            embs_dict["T"] = text_embs

        if "A" in self.named_encoders and inputs.audio_input_values is not None:
            audio_outputs = self.named_encoders["A"](
                inputs.audio_input_values, attention_mask=inputs.audio_attention_mask
            )
            audio_embs = audio_outputs.last_hidden_state[:, 0]
            embs_dict["A"] = audio_embs

        if "V" in self.named_encoders and inputs.video_pixel_values is not None:
            video_outputs = self.named_encoders["V"](inputs.video_pixel_values)
            video_embs = video_outputs.last_hidden_state[:, 0]
            embs_dict["V"] = video_embs

        return embs_dict

    def cached_compute_embs(self, inputs: MultimodalInput) -> dict[str, torch.Tensor]:
        unique_keys = inputs.get_unique_keys()
        cached_list = load_cached_tensors(unique_keys, cache_manager=self.cache_manager)
        if cached_list is None:
            logger.debug("cache missed")
            with torch.no_grad():
                embs_dict = self.compute_embs(inputs)
            save_cached_tensors(unique_keys, embs_dict, cache_manager=self.cache_manager)
            return embs_dict
        return {modal: torch.cat(cache) for modal, cache in cached_list.items()}

    def pool_embs(
        self,
        embs_dict: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {name: self.named_poolers[name](embs_dict[name]) for name in self.named_poolers if name in embs_dict}

    def direct_forward(self, inputs: MultimodalInput) -> dict[str, torch.Tensor]:
        embs_dict = self.compute_embs(inputs)
        pooler_output = self.pool_embs(embs_dict)
        return pooler_output

    def cached_forward(self, inputs: MultimodalInput) -> dict[str, torch.Tensor]:
        embs_dict = self.cached_compute_embs(inputs)
        pooler_output = self.pool_embs(embs_dict)
        return pooler_output

    def forward(self, inputs: MultimodalInput) -> dict[str, torch.Tensor]:
        if self.frozen_encoders and self.use_cache:
            try:
                return self.cached_forward(inputs)
            except Exception as e:
                logger.warning(f"use direct_forward because of error in cached_forward: {e}")
        return self.direct_forward(inputs)

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
            module = self.named_encoders[name]
            if isinstance(module, PeftModel):
                set_peft_model_state_dict(module, state_dict)
            else:
                module.load_state_dict(state_dict)

    def get_state_dicts(self) -> tuple[StateDicts, StateDicts]:
        state_dicts: dict[str, dict[str, torch.Tensor]] = {}
        peft_state_dicts: dict[str, dict[str, torch.Tensor]] = {}
        for name, module in self.named_encoders.items():
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
    def encoder_hash(self):
        if not self.frozen_encoders:
            self.frozen_encoders = True
            logger.warning("Encoders are not frozen, freezing encoders for hashing")
        bytes_dict: dict[str, bytes] = {}
        state_dicts, peft_state_dicts = self.get_state_dicts()
        for name, state_dict in itertools.chain(state_dicts.items(), peft_state_dicts.items()):
            bytes_dict[name] = save(state_dict)
        serialized_dict = pickle.dumps(bytes_dict)
        return hash_bytes(serialized_dict)

    def save(self, original_encoder_dir: Path) -> dict[str, Path]:
        state_path_dict: dict[str, Path] = {}
        self.encoder_dir.mkdir(parents=True, exist_ok=True)
        state_dicts, peft_state_dicts = self.get_state_dicts()
        for name, state_dict in state_dicts.items():
            self.named_encoders[name].config.save_pretrained(original_encoder_dir / name)
            state_bytes = save(state_dict)
            model_hash = hash_bytes(state_bytes)
            model_path = Path(f"{self.encoder_dir}/{model_hash}.safetensors")
            state_path_dict[name] = model_path.absolute()
            if model_path.exists():
                continue
            with open(model_path, "wb") as f:
                f.write(state_bytes)
        for name, state_dict in peft_state_dicts.items():
            state_bytes = save(state_dict)
            model_hash = hash_bytes(state_bytes)
            model_path = Path(f"{self.encoder_dir}/peft_{model_hash}.safetensors")
            state_path_dict[f"peft_{name}"] = model_path.absolute()
            if model_path.exists():
                continue
            with open(model_path, "wb") as f:
                f.write(state_bytes)

        save_dict_to_file(self.get_meta_info(), original_encoder_dir / "meta.toml")
        save_file(self.named_poolers.state_dict(), original_encoder_dir / "poolers.safetensors")
        return state_path_dict

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        *,
        use_cache: bool = True,
        frozen_encoders: bool = True,
        use_peft: bool = False,
        encoder_dir: Path = Path("./checkpoints/encoders"),
    ) -> Self:
        from transformers import AutoConfig, AutoModel

        encoders: dict[str, tuple[nn.Module, int]] = {}
        backbone_config = load_dict_from_path(checkpoint_path / "meta.toml")

        for name, feature_size in backbone_config["feature_sizes"].items():
            if not (checkpoint_path / name).exists():
                logger.warning(f"{name} not found in {checkpoint_path}")
                continue
            config = AutoConfig.from_pretrained(checkpoint_path / name / "config.json")
            model = AutoModel.from_config(config)
            model.load_state_dict(load_file(checkpoint_path / f"{name}.safetensors"), strict=False)
            if (checkpoint_path / f"peft_{name}.safetensors").exists():
                model.load_state_dict(load_file(checkpoint_path / f"peft_{name}.safetensors"), strict=False)
            encoders[name] = (model, feature_size)
        self = cls(
            encoders,
            use_cache=use_cache,
            frozen_encoders=frozen_encoders,
            use_peft=use_peft,
            encoder_dir=encoder_dir,
        )
        self.named_poolers.load_state_dict(load_file(checkpoint_path / "poolers.safetensors"))
        self.cache_manager.cache_dir = Path(f"./cache/{self.encoder_hash}")
        return self

    def load_checkpoint(self, checkpoint_path: Path) -> Self:
        for name, encoder in self.named_encoders.items():
            if not (checkpoint_path / name).exists():
                logger.warning(f"{name} not found in {checkpoint_path}")
                continue
            encoder.load_state_dict(load_file(checkpoint_path / f"{name}.safetensors"), strict=False)
            if (checkpoint_path / f"peft_{name}.safetensors").exists():
                encoder.load_state_dict(load_file(checkpoint_path / f"peft_{name}.safetensors"), strict=False)
        self.named_poolers.load_state_dict(load_file(checkpoint_path / "poolers.safetensors"))
        # TODO: encoder hash maybe changed but not updated
        self.cache_manager.cache_dir = Path(f"./cache/{self.encoder_hash}")
        return self
