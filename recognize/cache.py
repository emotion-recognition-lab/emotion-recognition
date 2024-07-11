from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .typing import StateDict


def load_cached_tensors(
    names: list[str], cache_dir: Path = Path("cache")
) -> tuple[list[StateDict], list[int], list[int]]:
    no_cache_index_list: list[int] = []
    cache_index_list: list[int] = []
    cache_list: list[dict[str, torch.Tensor]] = []
    for i, n in enumerate(names):
        cache_file = cache_dir / f"{n}.safetensors"
        if cache_file.exists():
            cache_index_list.append(i)
            cache_list.append(load_file(cache_file))
        else:
            no_cache_index_list.append(i)
    return cache_list, cache_index_list, no_cache_index_list


def save_cached_tensors(
    names: list[str],
    tensors_dict: dict[str, torch.Tensor | None],
    *,
    cache_dir: Path = Path("cache"),
):
    for i, n in enumerate(names):
        cache_file = cache_dir / f"{n}.safetensors"
        cache_file.mkdir(parents=True, exist_ok=True)
        tensor_dict = {k: v[i] for k, v in tensors_dict.items() if v is not None}
        save_file(tensor_dict, cache_file)
