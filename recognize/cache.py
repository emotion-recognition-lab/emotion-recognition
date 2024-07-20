from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .typing import StateDict


def load_cache(cache_file: Path | str) -> StateDict | None:
    cache_file = Path(cache_file)
    if cache_file.exists():
        return load_file(cache_file)
    else:
        return None


def load_cached_tensors(
    names: list[str], *, cache_dir: Path | str = Path("./cache")
) -> tuple[list[StateDict], list[int], list[int]]:
    no_cache_index_list: list[int] = []
    cache_index_list: list[int] = []
    cache_list: list[dict[str, torch.Tensor]] = []
    for i, n in enumerate(names):
        cache_file = Path(cache_dir) / f"{n}.safetensors"
        cache = load_cache(cache_file)
        if cache is not None:
            cache_index_list.append(i)
            cache_list.append(cache)
        else:
            no_cache_index_list.append(i)
    return cache_list, cache_index_list, no_cache_index_list


def save_cached_tensors(
    names: list[str],
    tensors_dict: dict[str, torch.Tensor | None],
    *,
    cache_dir: Path | str = Path("./cache"),
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(names):
        tensor_dict = {k: v[i] for k, v in tensors_dict.items() if v is not None}
        cache_file = Path(cache_dir) / f"{n}.safetensors"
        save_file(tensor_dict, cache_file)
