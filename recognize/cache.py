from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import torch
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import load_file, save_file

from .typing import StateDict


class Cache(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    maxsize: int
    cache: dict[str, StateDict] = Field(default_factory=dict)
    current_size: int = 0

    def get(self, key: str) -> StateDict | None:
        return self.cache.get(key)

    def set(self, key: str, value: StateDict):
        value_size = sys.getsizeof(value)
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
        cache_file = self.cache_dir / f"{key}.safetensors"
        if cache_file.exists():
            value = load_file(cache_file)
            self.cache.set(key, value)
            return value
        else:
            return None

    def save_disk_cache(self, key: str, value: StateDict):
        cache_file = self.cache_dir / f"{key}.safetensors"
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
        self.cache.set(key, value)
        if self.deivce_cache is not None:
            value = {k: v.to(self.device) for k, v in value.items()}
            self.deivce_cache.set(key, value)
        self.save_disk_cache(key, value)


def load_cached_tensors(
    keys: list[str],
    *,
    cache_manager: CacheManager,
) -> tuple[list[StateDict], list[int], list[int]]:
    no_cache_index_list: list[int] = []
    cache_index_list: list[int] = []
    cache_list: list[dict[str, torch.Tensor]] = []
    for i, n in enumerate(keys):
        cache = cache_manager.get(n)
        if cache is not None:
            cache_index_list.append(i)
            cache_list.append(cache)
        else:
            no_cache_index_list.append(i)
    return cache_list, cache_index_list, no_cache_index_list


def save_cached_tensors(
    keys: Sequence[str],
    value: StateDict,
    *,
    cache_manager: CacheManager,
):
    for i, n in enumerate(keys):
        tensor_dict = {k: v[i] for k, v in value.items()}
        cache_manager.set(n, tensor_dict)
