from __future__ import annotations

from collections import Counter

import numpy as np


def timeit(func, count: int = 100):
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"first run {func.__name__} cost time: {end - start:.2f}s")

        costs = []
        for _ in range(count):
            start = time.time()
            func(*args, **kwargs)
            end = time.time()
            costs.append(end - start)
        print(f"{func.__name__} mean cost time: {sum(costs)/len(costs):.2f}s")
        return result

    return wrapper


def sample_frame_indices(length: int, target_length: int):
    if length <= target_length:
        indices = np.arange(length)
        indices = np.pad(indices, (0, target_length - length), "constant", constant_values=indices[-1])
        return indices

    step = length / target_length
    indices = np.arange(0, length, step=step)
    indices = np.floor(indices).astype(int)
    return indices


@timeit
def read_video_opencv(video_path: str):
    import cv2

    video = cv2.VideoCapture(video_path)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
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

    return frames


@timeit
def read_video_pyav(video_path: str):
    import av

    container = av.open(video_path)
    indices = sample_frame_indices(target_length=32, length=container.streams.video[0].frames)

    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    indices_counter = Counter(indices)

    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices_counter:
            for _ in range(indices_counter[i]):
                frames.append(frame.to_image())

    return frames


np.testing.assert_allclose(
    read_video_opencv("benchmark/samples/video.mp4"),
    read_video_pyav("benchmark/samples/video.mp4"),
)
