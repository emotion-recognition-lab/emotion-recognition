from __future__ import annotations

import os
import random
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import typer
from loguru import logger
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler
from safetensors.torch import load_file

from recognize.config import InferenceConfig, ModelEncoderConfig, load_training_config, save_config
from recognize.dataset import IEMOCAPDataset, MELDDataset, MultimodalDataset, PilotDataset, SIMSDataset
from recognize.evaluate import TrainingResult
from recognize.model import (
    LazyMultimodalInput,
    MultimodalBackbone,
    MultimodalModel,
    UnimodalModel,
)
from recognize.module import gen_fusion_layer, get_feature_sizes_dict
from recognize.preprocessor import Preprocessor
from recognize.typing import DatasetClass, DatasetLabelType, LogLevel, ModalType
from recognize.utils import (
    find_best_model,
    load_best_model,
    load_model,
    train_and_eval,
)


def init_logger(log_level: LogLevel, label: str):
    handler = RichHandler(highlighter=NullHighlighter(), markup=True)
    logger.remove()
    logger.add(f"./logs/{label}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt")
    logger.add(handler, format="{message}", level=log_level)


def init_torch():
    import torch

    torch.set_float32_matmul_precision("high")


def seed_everything(seed: int | None = None):
    import torch

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    logger.info(f"Set seed to {seed}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def provide_datasets(
    dataset_path: Path,
    label_type: DatasetLabelType = "emotion",
    dataset_class_str: DatasetClass = "MELDDataset",
) -> tuple[MultimodalDataset, MultimodalDataset, MultimodalDataset]:
    dataset_class: type[MELDDataset | PilotDataset | SIMSDataset | IEMOCAPDataset] = {
        "MELDDataset": MELDDataset,
        "PilotDataset": PilotDataset,
        "SIMSDataset": SIMSDataset,
        "IEMOCAPDataset": IEMOCAPDataset,
    }[dataset_class_str]
    train_dataset = dataset_class(
        dataset_path.as_posix(),
        split="train",
        label_type=label_type,
    )
    dev_dataset = dataset_class(
        dataset_path.as_posix(),
        split="valid",
        label_type=label_type,
    )
    test_dataset = dataset_class(
        dataset_path.as_posix(),
        split="test",
        label_type=label_type,
    )
    return train_dataset, dev_dataset, test_dataset


def symlink(src: Path | str, dst: Path | str, target_is_directory: bool = False):
    dst = Path(dst)
    src = Path(src)
    dst.unlink(missing_ok=True)
    logger.info(f"Create symlink: {dst} -> {src}")
    dst.symlink_to(src.absolute(), target_is_directory=target_is_directory)


def generate_preprocessor_and_backbone(
    config_model_encoder: dict[ModalType, ModelEncoderConfig],
    datasets: list[MultimodalDataset],
    checkpoints: Sequence[Path] = (),
):
    from transformers import (
        AutoConfig,
        AutoFeatureExtractor,
        AutoImageProcessor,
        AutoModel,
        AutoTokenizer,
        VivitImageProcessor,
    )

    for checkpoint in checkpoints:
        preprocessor_path = checkpoint / "preprocessor"
        if preprocessor_path.exists():
            preprocessor = Preprocessor.from_pretrained(preprocessor_path)
            logger.info(f"Load preprocessor from [blue]{checkpoint}[/]")
            break
    else:
        preprocessor = Preprocessor()

    for modal, config in config_model_encoder.items():
        encoder_name = config.model
        if modal == "T":
            preprocessor.tokenizer = preprocessor.tokenizer or AutoTokenizer.from_pretrained(encoder_name)
        elif modal == "A":
            preprocessor.feature_extractor = preprocessor.feature_extractor or AutoFeatureExtractor.from_pretrained(
                encoder_name
            )
        elif modal == "V":
            preprocessor.image_processor = preprocessor.image_processor or VivitImageProcessor.from_pretrained(
                encoder_name
            )
        else:
            preprocessor.image_processor = preprocessor.image_processor or AutoImageProcessor.from_pretrained(
                encoder_name
            )
    for dataset in datasets:
        dataset.set_preprocessor(preprocessor)
    for checkpoint in checkpoints:
        backbone_path = checkpoint / "backbone"
        if backbone_path.exists():
            backbone = MultimodalBackbone.from_checkpoint(backbone_path)
            logger.info(f"Load backbone from [blue]{checkpoint}[/]")
            break
    else:
        backbone_encoders = {}
        for modal, config in config_model_encoder.items():
            encoder_name = config.model
            encoder_checkpoint = config.checkpoint
            encoder_feature_size = config.feature_size
            if encoder_checkpoint is not None:
                logger.info(f"Load {modal} encoder from [blue]{encoder_checkpoint}[/]")
                checkpoint_path = Path(encoder_checkpoint)
                config = AutoConfig.from_pretrained(checkpoint_path / f"{modal}/config.json")
                backbone_encoder = AutoModel.from_config(config)
                backbone_encoder.load_state_dict(load_file(checkpoint_path / f"{modal}.safetensors"))
            else:
                backbone_encoder = AutoModel.from_pretrained(encoder_name)
            backbone_encoders[modal] = (backbone_encoder, encoder_feature_size)
        backbone = MultimodalBackbone(
            backbone_encoders,
            init_hook=datasets[0].special_process if len(datasets) > 0 else None,
        )
    return preprocessor, backbone


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def distill(
    config_path: list[Path],
    teacher_checkpoint: Path = typer.Option(..., help="The checkpoint of the teacher model"),
    checkpoint: Path | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
) -> None:
    """
    When using knowledge distillation,
    it is essential to ensure that the preprocessor for both the teacher model and the student model is the same.
    Generally, you can use the one corresponding to the student model,
    as the student model typically has more modalities.
    """
    import torch
    from torch.utils.data import DataLoader

    config = load_training_config(*config_path, batch_size=batch_size, seed=seed)
    config_training_mode = config.training_mode
    config_batch_size = config.batch_size
    config_model_encoder = config.model.encoder
    config_fusion = config.model.fusion
    config_dataset = config.dataset

    init_logger(config.log_level, config.label)
    seed = seed_everything(seed)
    init_torch()

    model_label = config.model.label
    model_hash = config.model.hash

    assert config_training_mode != "lora", "Lora is not supported in distillation"
    assert config_fusion is not None
    assert (teacher_checkpoint / "preprocessor").exists()
    assert (teacher_checkpoint / "backbone").exists()

    if checkpoint is not None:
        checkpoint_dir = checkpoint
    else:
        checkpoint_dir = Path(f"./checkpoints/distillation/{config.label}/{model_hash}--{seed}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, dev_dataset, test_dataset = provide_datasets(
        config_dataset.path,
        label_type=config_dataset.label_type,
        dataset_class_str=config_dataset.dataset_class,
    )
    preprocessor, backbone = generate_preprocessor_and_backbone(
        config_model_encoder,
        datasets=[train_dataset, dev_dataset, test_dataset],
        checkpoints=[checkpoint_dir],
    )

    teacher_backbone = MultimodalBackbone.from_checkpoint(teacher_checkpoint / "backbone")
    teacher_preprocessor = Preprocessor.from_pretrained(teacher_checkpoint / "preprocessor")
    # TODO: maybe those will be covered by load_best_model
    backbone.named_encoders.load_state_dict(teacher_backbone.named_encoders.state_dict(), strict=False)
    backbone.named_poolers.load_state_dict(teacher_backbone.named_poolers.state_dict(), strict=False)

    preprocessor_dir = checkpoint_dir / "preprocessor"
    if not preprocessor_dir.exists():
        preprocessor.save_pretrained(preprocessor_dir)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config_batch_size,
        shuffle=True,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
        num_workers=0,
    )
    dev_data_loader = DataLoader(
        dev_dataset,
        batch_size=config_batch_size,
        shuffle=True,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
        num_workers=0,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=config_batch_size,
        shuffle=True,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
        num_workers=0,
    )
    class_weights = torch.tensor(train_dataset.class_weights, dtype=torch.float32).cuda()

    inference_config = InferenceConfig(num_classes=train_dataset.num_classes, model=config.model)
    save_config(config, checkpoint_dir / "training.toml")
    save_config(inference_config, checkpoint_dir / "inference.toml")

    feature_size = next(iter(config_model_encoder.values())).feature_size
    teacher_model = UnimodalModel(
        teacher_backbone,
        feature_size=feature_size,
        num_classes=train_dataset.num_classes,
        class_weights=class_weights,
    ).cuda()

    if (teacher_checkpoint / "stopper.yaml").exists():
        load_best_model(teacher_checkpoint, teacher_model)
    else:
        load_model(teacher_checkpoint, teacher_model)
    test_dataset.set_preprocessor(teacher_preprocessor)
    result = TrainingResult.auto_compute(teacher_model, test_data_loader)
    logger.info("Test result in [green]best teacher model[/]:")
    result.print()
    test_dataset.set_preprocessor(preprocessor)

    feature_sizes_dict = get_feature_sizes_dict(config_model_encoder)
    fusion_layer = gen_fusion_layer(config_fusion, feature_sizes_dict)
    model = MultimodalModel(
        backbone,
        fusion_layer,
        num_classes=train_dataset.num_classes,
        class_weights=class_weights,
    ).cuda()

    if config_training_mode == "trainable":
        model.backbone.use_cache = False
        model.backbone.unfreeze()

    if (checkpoint_dir / "stopper.yaml").exists():
        load_best_model(checkpoint_dir, model)
        result = TrainingResult.auto_compute(model, test_data_loader)
        result.print()

    result: TrainingResult = train_and_eval(
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        teacher_model=teacher_model,
        checkpoint_dir=checkpoint_dir,
        num_epochs=200,
        model_label=model_label,
        use_valid=False,
    )

    logger.info(f"Test result in best model({model_label}):")
    result.print()


@app.command()
def train(
    config_path: list[Path],
    checkpoint: Path | None = None,
    from_checkpoint: Path | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
) -> None:
    import torch
    from torch.utils.data import DataLoader

    config = load_training_config(*config_path, batch_size=batch_size, seed=seed)
    config_training_mode = config.training_mode
    config_batch_size = config.batch_size
    config_model_encoder = config.model.encoder
    config_dataset = config.dataset

    init_logger(config.log_level, config.label)
    seed = seed_everything(seed)
    init_torch()

    model_label = config.model.label
    model_hash = config.model.hash

    assert config_training_mode != "lora", "Lora is not supported in training"

    if checkpoint is not None:
        checkpoint_dir = checkpoint
    else:
        checkpoint_dir = Path(f"./checkpoints/training/{config.label}/{model_hash}--{seed}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, dev_dataset, test_dataset = provide_datasets(
        config_dataset.path,
        label_type=config_dataset.label_type,
        dataset_class_str=config_dataset.dataset_class,
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config_batch_size,
        shuffle=True,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
        num_workers=0,
    )
    dev_data_loader = DataLoader(
        dev_dataset,
        batch_size=config_batch_size,
        shuffle=True,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
        num_workers=0,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=config_batch_size,
        shuffle=True,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
        num_workers=0,
    )
    candidate_checkpoints = []
    if from_checkpoint is not None:
        from_checkpoint_best = find_best_model(from_checkpoint)
        candidate_checkpoints.append(from_checkpoint / f"{from_checkpoint_best}")
    candidate_checkpoints.append(checkpoint_dir)
    preprocessor, backbone = generate_preprocessor_and_backbone(
        config_model_encoder,
        datasets=[train_dataset, dev_dataset, test_dataset],
        checkpoints=candidate_checkpoints,
    )
    if not (checkpoint_dir / "preprocessor").exists():
        preprocessor.save_pretrained(checkpoint_dir / "preprocessor")

    inference_config = InferenceConfig(num_classes=train_dataset.num_classes, model=config.model)
    save_config(config, checkpoint_dir / "training.toml")
    save_config(inference_config, checkpoint_dir / "inference.toml")

    class_weights = torch.tensor(train_dataset.class_weights, dtype=torch.float32).cuda()

    if config.model.fusion is None:
        assert len(config_model_encoder) == 1, "Multiple modals must give a fusion layer"
        feature_size = next(iter(config_model_encoder.values())).feature_size
        model = UnimodalModel(
            backbone,
            feature_size=feature_size,
            num_classes=train_dataset.num_classes,
            class_weights=class_weights,
        ).cuda()
    else:
        feature_sizes_dict = get_feature_sizes_dict(config_model_encoder)
        fusion_layer = gen_fusion_layer(config.model.fusion, feature_sizes_dict)
        model = MultimodalModel(
            backbone,
            fusion_layer,
            num_classes=train_dataset.num_classes,
            class_weights=class_weights,
        ).cuda()
    if config_training_mode == "trainable":
        model.backbone.use_cache = False
        model.backbone.unfreeze()

    if (checkpoint_dir / "stopper.yaml").exists():
        # TODO: stopper should be loaded in the training process
        load_best_model(checkpoint_dir, model)
        result = TrainingResult.auto_compute(model, test_data_loader)
        result.print()

    result: TrainingResult = train_and_eval(
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        checkpoint_dir=checkpoint_dir,
        num_epochs=200,
        model_label=model_label,
        use_valid=False,
    )
    logger.info(f"Test result in best model({model_label}):")
    result.print()


@app.command()
def evaluate(checkpoint: Path) -> None:
    init_torch()
    from torch.utils.data import DataLoader

    config = load_training_config(checkpoint / "training.toml")
    init_logger(config.log_level, config.label)
    config_model_encoder = config.model.encoder
    config_dataset = config.dataset

    config_batch_size = config.batch_size

    assert (checkpoint / "preprocessor").exists(), "Preprocessor not found, the checkpoint is not valid"

    _, _, test_dataset = provide_datasets(
        config_dataset.path,
        label_type=config_dataset.label_type,
        dataset_class_str=config_dataset.dataset_class,
    )

    _, backbone = generate_preprocessor_and_backbone(
        config_model_encoder,
        datasets=[test_dataset],
        checkpoints=[checkpoint],
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=config_batch_size,
        shuffle=True,
        collate_fn=LazyMultimodalInput.collate_fn,
        pin_memory=True,
    )
    if config.model.fusion is None:
        assert len(config_model_encoder) == 1, "Multiple modals must give a fusion layer"
        feature_size = next(iter(config_model_encoder.values())).feature_size
        model = UnimodalModel(
            backbone,
            feature_size=feature_size,
            num_classes=test_dataset.num_classes,
        ).cuda()
    else:
        feature_sizes_dict = get_feature_sizes_dict(config_model_encoder)
        fusion_layer = gen_fusion_layer(config.model.fusion, feature_sizes_dict)
        model = MultimodalModel(
            backbone,
            fusion_layer,
            num_classes=test_dataset.num_classes,
        ).cuda()

    model.backbone.freeze()
    load_best_model(checkpoint, model)
    result = TrainingResult.auto_compute(model, test_data_loader)
    result.print()


@app.command()
def inference(
    checkpoint: Path,
    text: str | None = None,
    audio_path: Path | None = None,
    video_path: Path | None = None,
):
    import torch

    config = load_training_config(checkpoint / "training.toml")
    config_model_encoder = config.model.encoder
    config_fusion = config.model.fusion
    init_logger(config.log_level, config.label)

    preprocessor = Preprocessor.from_pretrained(f"{checkpoint}/preprocessor")
    model_checkpoint = checkpoint / str(find_best_model(checkpoint))

    backbone = MultimodalBackbone.from_checkpoint(Path(f"{model_checkpoint}/backbone"))
    if config_fusion is None:
        model = UnimodalModel.from_checkpoint(model_checkpoint, backbone=backbone).cuda()
    else:
        feature_sizes_dict = get_feature_sizes_dict(config_model_encoder)
        fusion_layer = gen_fusion_layer(config_fusion, feature_sizes_dict)
        model = MultimodalModel.from_checkpoint(model_checkpoint, backbone=backbone, fusion_layer=fusion_layer).cuda()
    model.eval()
    inputs = LazyMultimodalInput(
        preprocessor=preprocessor,
        texts=[text] if text is not None else None,
        audio_paths=[audio_path.as_posix()] if audio_path is not None else None,
        video_paths=[video_path.as_posix()] if video_path is not None else None,
    ).cuda()
    outputs = model(inputs)
    print("Predicted logits:", outputs.logits)
    print("Predicted labels:", torch.argmax(outputs.logits, dim=-1).item())


if __name__ == "__main__":
    app()
