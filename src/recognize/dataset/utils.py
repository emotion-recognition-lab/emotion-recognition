from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


def sample_frame_indices(length: int, target_length: int):
    if length <= target_length:
        indices = np.arange(length)
        indices = np.pad(indices, (0, target_length - length), "constant", constant_values=indices[-1])
        return indices

    step = length / target_length
    indices = np.arange(0, length, step=step)
    indices = np.floor(indices).astype(int)
    return indices


def read_videos(video_path: str):
    if os.path.exists(f"{video_path}.cache.pkl"):
        with open(f"{video_path}.cache.pkl", "rb") as f:
            return pickle.load(f)
    from collections import Counter

    import cv2
    from cv2.typing import MatLike

    video = cv2.VideoCapture(video_path)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[MatLike] = []
    count = 0
    indices = sample_frame_indices(target_length=32, length=length)
    indices = list(indices)
    indices_counter = Counter(indices)

    while video.isOpened():
        ret, image = video.read()
        if not ret:
            break
        if count in indices_counter:
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for _j in range(indices_counter[count]):
                frames.append(frame)
        count += 1

    video.release()
    with open(f"{video_path}.cache.pkl", "wb") as f:
        pickle.dump(frames, f)
    return frames
