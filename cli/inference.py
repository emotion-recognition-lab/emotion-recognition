from __future__ import annotations

from pathlib import Path

import torch
import typer
from transformers import AutoModel

from recognize.dataset import Preprocessor
from recognize.model import LazyMultimodalInput, LowRankFusionLayer, MultimodalBackbone, MultimodalModel
from recognize.utils import init_logger, load_best_model

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def inference(checkpoint: Path = Path("."), *, log_level: str = "DEBUG"):
    init_logger(log_level)
    preprocessor = Preprocessor.from_pretrained(f"./{checkpoint}/preprocessor")
    text_feature_size, audio_feature_size, video_feature_size = 128, 16, 1
    backbone = MultimodalBackbone(
        AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2"),
        AutoModel.from_pretrained("facebook/wav2vec2-base-960h"),
        use_cache=False,
    )
    fusion_layer = LowRankFusionLayer([text_feature_size, audio_feature_size, video_feature_size], 16, 128)
    model = MultimodalModel(
        backbone,
        fusion_layer,
        text_feature_size=text_feature_size,
        audio_feature_size=audio_feature_size,
        video_feature_size=video_feature_size,
        num_classes=3,
    ).cuda()
    model.eval()
    load_best_model(checkpoint, model)
    inputs = LazyMultimodalInput(
        preprocessor=preprocessor,
        texts=["I'm feeling really good today.", "I'm feeling really good today, I'm sad."],
    ).cuda()
    for _ in range(100):
        outputs = model(inputs)
        print(outputs.logits)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    app()
