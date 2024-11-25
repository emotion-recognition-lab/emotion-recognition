from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import torch
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import load_file, save_file

from .typing import StateDict


def hash_bytes(bytes_data: bytes) -> str:
    hasher = hashlib.sha256()
    hasher.update(bytes_data)
    return hasher.hexdigest()[:16]


def get_tensor_bytes(tensors: Iterable[torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


class Cache(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    maxsize: int
    cache: dict[str, StateDict] = Field(default_factory=dict)
    current_size: int = 0

    def get(self, key: str) -> StateDict | None:
        return self.cache.get(key)

    def set(self, key: str, value: StateDict):
        value_size = get_tensor_bytes(value.values())
        if self.current_size + value_size < self.maxsize:
            self.cache[key] = value
            self.current_size += value_size
        else:
            logger.info(f"skip {key} for cache full")


class CacheManager:
    cache: Cache
    deivce_cache: Cache | None
    maxsize: tuple[int, int]

    def __init__(
        self,
        maxsize: int | tuple[int, int],
        device: torch.device = torch.device("cuda"),
        *,
        cache_dir: Path = Path("./cache"),
    ):
        self.maxsize = maxsize if isinstance(maxsize, tuple) else (maxsize, maxsize)
        self.cache = Cache(maxsize=self.maxsize[0])
        self.device = device
        if device.type == "cpu":
            self.deivce_cache = None
        else:
            self.deivce_cache = Cache(maxsize=self.maxsize[1])
        self.cache_dir = cache_dir

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, cache_dir: Path):
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_dir = cache_dir

    def load_disk_cache(self, key: str) -> StateDict | None:
        cache_file = self.cache_dir / key
        if cache_file.exists():
            value = load_file(cache_file)
            self.cache.set(key, value)
            return value
        else:
            return None

    def save_disk_cache(self, key: str, value: StateDict):
        cache_file = self.cache_dir / key
        save_file(value, cache_file)

    def get(self, key: str) -> StateDict | None:
        if self.deivce_cache is None:
            if value := self.cache.get(key):
                return value
            if value := self.load_disk_cache(key):
                self.cache.set(key, value)
                return value
        else:
            if value := self.deivce_cache.get(key):
                return value
            if value := self.cache.get(key):
                value = {k: v.to(self.device) for k, v in value.items()}
                self.deivce_cache.set(key, value)
                return value
            if value := self.load_disk_cache(key):
                self.cache.set(key, value)
                value = {k: v.to(self.device) for k, v in value.items()}
                self.deivce_cache.set(key, value)
                return value
        return None

    def set(self, key: str, value: StateDict):
        self.cache.set(key, {k: v.to("cpu") for k, v in value.items()})
        if self.deivce_cache is not None:
            value = {k: v.to(self.device, copy=True) for k, v in value.items()}
            self.deivce_cache.set(key, value)
        self.save_disk_cache(key, value)


def load_cached_tensors(
    keys_mapping: Mapping[str, Sequence[str]],
    *,
    cache_manager: CacheManager,
) -> dict[str, list[torch.Tensor]] | None:
    lengths = [len(v) for v in keys_mapping.values()]
    if len(set(lengths)) != 1:
        raise ValueError("All sequences must have the same length")
    length = lengths[0]
    modal_names = list(keys_mapping.keys())

    values_mapping: dict[str, list[torch.Tensor]] = {modal_name: [] for modal_name in modal_names}
    for modal_name in modal_names:
        for i in range(length):
            cache = cache_manager.get(keys_mapping[modal_name][i])
            if cache is None:
                logger.trace(f"{modal_name} not found")
                del values_mapping[modal_name]
                break
            values_mapping[modal_name].append(cache[modal_name])

    if not values_mapping:
        return None
    return values_mapping


def save_cached_tensors(
    keys_mapping: Mapping[str, Sequence[str]],
    values_mapping: StateDict,
    *,
    cache_manager: CacheManager,
):
    for modal_name, keys in keys_mapping.items():
        if (values := values_mapping.get(modal_name)) is None:
            continue
        for k, v in zip(keys, values, strict=True):
            tensor_dict = {modal_name: v.detach().unsqueeze(0)}
            cache_manager.set(k, tensor_dict)
