from __future__ import annotations

from collections import Counter

import numpy as np


def read_video_pyav(container, indices):  # 解码视频
    """
    Decode the video with PyAV decoder.
    Args:
    container (av.container.input.InputContainer): PyAV container.
    indices (List[int]): List of frame indices to decode.
    Returns:
    result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """

    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    indices_counter = Counter(indices)
    # print(indices_counter)
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices_counter:
            for _j in range(indices_counter[i]):
                frames.append(frame)
    # print(len(frames))

    return np.stack([x.to_ndarray(format="rgb24") for x in frames])  # 返回解码后的帧序列


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):  # 从视频中采样一定数量的帧索引
    """
    Sample a given number of frame indices from the video.
    Args:
    clip_len (int): Total number of frames to sample.  #采样的总帧数
    frame_sample_rate (int): Sample every n-th frame.  #采样的帧间隔

    seg_len (int): Maximum allowed index of sample's last frame.  #采样的最后一帧的最大索引
    Returns:
    indices (List[int]): List of sampled frame indices
    """

    converted_len = int(clip_len * frame_sample_rate)  # 需要被使用的总帧数
    total_frames = seg_len
    # print(total_frames)
    if total_frames <= converted_len:
        increase_factor = converted_len // total_frames
        repeated_frames = np.repeat(np.arange(total_frames), increase_factor + 1)
        # indices = np.random.choice(repeated_frames, size=clip_len, replace=True).astype(np.int64)
        end_idx = np.random.randint(converted_len, len(repeated_frames))
        # print(end_idx)
        start_idx = end_idx - converted_len
        # print(repeated_frames)
        # indices = repeated_frames[start_idx:end_idx:frame_sample_rate-1].astype(np.int64)
        indices = np.random.choice(repeated_frames, size=clip_len, replace=False)
        indices = np.sort(indices)
        # print(indices,indices[-1])
        indices = indices.astype(np.int64)
    else:
        end_idx = np.random.randint(converted_len, seg_len)  # 随机选择一帧作为采样的最后一帧
        start_idx = end_idx - converted_len  # 采样序列的开始帧
        indices = np.linspace(start_idx, end_idx, num=clip_len)  # 生成采样帧的索引
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)  # 确保采样的最后一帧索引不超过 end_idx -1
        # print(indices)
    return indices  # 返回采样后的帧 索引列表
    # end_idx = np.random.randint(converted_len, seg_len)  #随机选择一帧作为采样的最后一帧
    # start_idx = end_idx - converted_len  #采样序列的开始帧
    # indices = np.linspace(start_idx, end_idx, num=clip_len)  #生成采样帧的索引
    # indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)  #确保采样的最后一帧索引不超过 end_idx -1
    # return indices  #返回采样后的帧 索引列表


def read_videos(video_path):
    import av

    container = av.open(video_path)
    indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
    videos = read_video_pyav(container=container, indices=indices)
    return list(videos)
