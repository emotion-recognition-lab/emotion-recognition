from __future__ import annotations

import numpy as np


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
    import cv2

    video = cv2.VideoCapture(video_path)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    count = 0
    indices = sample_frame_indices(target_length=32, length=length)
    indices = list(indices)

    while video.isOpened():
        ret, image = video.read()
        if not ret:
            break

        if count in list(indices):
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        count += 1

    video.release()

    return np.array(frames)
