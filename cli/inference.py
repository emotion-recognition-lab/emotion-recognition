from __future__ import annotations

from pathlib import Path

import torch
import typer

from recognize.dataset import Preprocessor
from recognize.model import (
    LazyMultimodalInput,
    LowRankFusionLayer,
    MultimodalBackbone,
    MultimodalModel,
)
from recognize.utils import find_best_model, init_logger

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def inference(checkpoint: Path = Path("."), *, log_level: str = "DEBUG"):
    init_logger(log_level)
    preprocessor = Preprocessor.from_pretrained(f"{checkpoint}/preprocessor")
    model_checkpoint = f"{checkpoint}/{find_best_model(checkpoint)}"
    text_feature_size, audio_feature_size, video_feature_size = 128, 16, 1

    backbone = MultimodalBackbone.from_checkpoint(
        f"{model_checkpoint}/backbones",
        use_cache=False,
    )
    fusion_layer = LowRankFusionLayer(
        [text_feature_size, audio_feature_size, video_feature_size], 16, 128
    )
    model = MultimodalModel.from_checkpoint(
        model_checkpoint, backbone=backbone, fusion_layer=fusion_layer
    ).cuda()
    model.eval()
    inputs = LazyMultimodalInput(
        preprocessor=preprocessor,
        # texts=["I'm feeling really good today.", "I'm feeling really good today, I'm sad."],
        audio_paths=[
            "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly/audios/test/dia58_utt1.flac",
            "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly/audios/test/dia58_utt2.flac",
            "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly/audios/test/dia58_utt2.flac",
        ],
        video_paths=[
            "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly/videos/test/dia58_utt1.mp4",
            "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly/videos/test/dia58_utt1.mp4",
            "/home/zrr/datasets/OpenDataLab___MELD/raw/MELD/MELD.AudioOnly/videos/test/dia58_utt2.mp4",
        ],
    ).cuda()
    for _ in range(100):
        outputs = model(inputs)
        print(outputs.logits)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    app()
