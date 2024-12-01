from __future__ import annotations

import os
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import typer
from loguru import logger
from safetensors.torch import load_file
from utils import init_logger

from recognize.config import (
    CrossModalContrastiveConfig,
    InferenceConfig,
    ModelEncoderConfig,
    load_training_config,
    save_config,
)
from recognize.dataset import IEMOCAPDataset, MELDDataset, MultimodalDataset, PilotDataset, SIMSDataset
from recognize.evaluate import TrainingResult
from recognize.model import (
    LazyMultimodalInput,
    MultimodalBackbone,
    MultimodalModel,
    UnimodalModel,
)
from recognize.module import DistillationLoss, gen_fusion_layer, get_feature_sizes_dict
from recognize.preprocessor import Preprocessor
from recognize.typing import DatasetClass, DatasetLabelType, ModalType
from recognize.utils import (
    find_best_model,
    load_best_model,
    load_model,
    train_and_eval,
)


def init_torch():
    torch.set_float32_matmul_precision("high")


def seed_everything(seed: int | None = None):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    logger.info(f"Set seed to {seed}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
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


def generate_preprocessor_and_backbone(
    config_model_encoder: dict[ModalType, ModelEncoderConfig],
    datasets: list[MultimodalDataset],
    checkpoints: Sequence[Path] = (),
    *,
    frozen_encoders: bool = False,
    use_cache: bool = True,
):
    from transformers import (
        AutoConfig,
        AutoFeatureExtractor,
        AutoImageProcessor,
        AutoModel,
        AutoTokenizer,
        VideoMAEImageProcessor,
        VivitImageProcessor,
    )

    assert len(datasets) > 0, "No dataset provided"

    for checkpoint in checkpoints:
        preprocessor_path = checkpoint / "preprocessor"
        if preprocessor_path.exists():
            preprocessor = Preprocessor.from_pretrained(preprocessor_path)
            logger.info(f"Load preprocessor from [blue]{checkpoint}[/]")
            break
    else:
        preprocessor = Preprocessor()
    for dataset in datasets:
        dataset.set_preprocessor(preprocessor)

    for modal, config in config_model_encoder.items():
        encoder_name = config.model
        if modal == "T":
            preprocessor.tokenizer = preprocessor.tokenizer or AutoTokenizer.from_pretrained(encoder_name)
        elif modal == "A":
            preprocessor.feature_extractor = preprocessor.feature_extractor or AutoFeatureExtractor.from_pretrained(
                encoder_name
            )
        elif modal == "V":
            if "timesformer" in encoder_name or "vivit" in encoder_name:
                preprocessor.image_processor = preprocessor.image_processor or VivitImageProcessor.from_pretrained(
                    encoder_name
                )
            else:
                preprocessor.image_processor = preprocessor.image_processor or VideoMAEImageProcessor.from_pretrained(
                    encoder_name
                )
        else:
            preprocessor.image_processor = preprocessor.image_processor or AutoImageProcessor.from_pretrained(
                encoder_name
            )
    for checkpoint in checkpoints:
        backbone_path = checkpoint / "backbone"
        if backbone_path.exists():
            backbone = MultimodalBackbone.from_checkpoint(
                backbone_path, frozen_encoders=frozen_encoders, use_cache=use_cache
            )
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
            init_hook=datasets[0].special_process,
            frozen_encoders=frozen_encoders,
            use_cache=use_cache,
        )
    return preprocessor, backbone


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command(deprecated=True)
def distill(
    config_path: list[Path],
    teacher_checkpoint: Path = typer.Option(..., help="The checkpoint of the teacher model"),
    checkpoint: Path | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
) -> None:
    logger.warning("Distill command is deprecated, please use train instead")


@app.command()
def train(
    config_path: list[Path],
    batch_size: int | None = None,
    seed: int | None = None,
    checkpoint: Path | None = None,
    from_checkpoint: Path | None = None,
    teacher_checkpoint: Path | None = typer.Option(
        None, help="The checkpoint of the teacher model to be used in distillation"
    ),
) -> None:
    from torch.utils.data import DataLoader

    config = load_training_config(*config_path, batch_size=batch_size, seed=seed)
    config_training_mode = config.training_mode
    config_batch_size = config.batch_size
    config_model_encoder = config.model.encoder
    config_model_fusion = config.model.fusion
    config_dataset = config.dataset
    config_loss = config.loss

    init_logger(config.log_level, Path(f"./logs/{config.label}"))
    seed = seed_everything(seed)
    init_torch()

    model_label = config.model.label
    model_hash = config.model.hash

    assert config_training_mode != "lora", "Lora is not supported in training"
    if teacher_checkpoint is not None:
        assert config_model_fusion is not None
        assert (teacher_checkpoint / "preprocessor").exists()
        assert (teacher_checkpoint / "backbone").exists()

    if checkpoint is not None:
        checkpoint_dir = checkpoint
    elif teacher_checkpoint is not None:
        checkpoint_dir = Path(f"./checkpoints/distillation/{config.label}/{model_hash}--{seed}")
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
        frozen_encoders=config_training_mode == "frozen",
        use_cache=config_training_mode == "frozen",
    )

    if teacher_checkpoint is not None:
        teacher_backbone = MultimodalBackbone.from_checkpoint(teacher_checkpoint / "backbone")
        teacher_preprocessor = Preprocessor.from_pretrained(teacher_checkpoint / "preprocessor")
        # TODO: maybe those will be covered by load_best_model
        backbone.named_encoders.load_state_dict(teacher_backbone.named_encoders.state_dict(), strict=False)
        backbone.named_poolers.load_state_dict(teacher_backbone.named_poolers.state_dict(), strict=False)

        teacher_config = load_training_config(teacher_checkpoint / "training.toml")
        feature_size = next(iter(teacher_config.model.encoder.values())).feature_size
        teacher_model = UnimodalModel(
            teacher_backbone,
            feature_size=feature_size,
            num_classes=train_dataset.num_classes,
        ).cuda()
        if (teacher_checkpoint / "stopper.yaml").exists():
            load_best_model(teacher_checkpoint, teacher_model)
        else:
            load_model(teacher_checkpoint, teacher_model)

        # evaluate teacher model
        test_dataset.set_preprocessor(teacher_preprocessor)
        result = TrainingResult.auto_compute(teacher_model, test_data_loader)
        logger.info("Test result in [green]best teacher model[/]:")
        result.print()
        test_dataset.set_preprocessor(preprocessor)
    else:
        teacher_model = None

    if not (checkpoint_dir / "preprocessor").exists():
        preprocessor.save_pretrained(checkpoint_dir / "preprocessor")

    inference_config = InferenceConfig(num_classes=train_dataset.num_classes, model=config.model)
    save_config(config, checkpoint_dir / "training.toml")
    save_config(inference_config, checkpoint_dir / "inference.toml")

    class_weights = torch.tensor(train_dataset.class_weights, dtype=torch.float32).cuda()
    feature_sizes_dict = get_feature_sizes_dict(config_model_encoder)

    if config_model_fusion is None:
        assert len(config_model_encoder) == 1, "Multiple modals must give a fusion layer"
        feature_size = next(iter(feature_sizes_dict.values()))
        model = UnimodalModel(
            backbone,
            feature_size=feature_size,
            num_classes=train_dataset.num_classes,
            class_weights=class_weights,
        ).cuda()
        if config_loss is not None and (config_loss_contrastive := config_loss.sample_contrastive) is not None:
            model.add_extra_loss_fn(config_loss_contrastive.to_loss_object(train_dataset.num_classes, feature_size))
    else:
        fusion_layer = gen_fusion_layer(config_model_fusion, feature_sizes_dict)
        model = MultimodalModel(
            backbone,
            fusion_layer,
            num_classes=train_dataset.num_classes,
            class_weights=class_weights,
        ).cuda()
        if config_loss is not None:
            if (config_sample_contrastive := config_loss.sample_contrastive) is not None:
                model.add_extra_loss_fn(
                    config_sample_contrastive.to_loss_object(train_dataset.num_classes, fusion_layer.output_size),
                )
            if (config_modal_contrastive := config_loss.modal_contrastive) is not None:
                assert isinstance(config_modal_contrastive, CrossModalContrastiveConfig)
                model.add_extra_loss_fn(config_modal_contrastive.to_loss_object(feature_sizes_dict))

    if teacher_model is not None:
        model.add_extra_loss_fn(DistillationLoss(teacher_model))
        # if config_training_mode == "trainable":
        #     backbone.freeze_modal("T")

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
        dropout_prob=config.dropout_prob,
    )
    logger.info(f"Test result in best model({model_label}):")
    result.print()


@app.command()
def evaluate(
    checkpoint: Path,
    seed: int | None = None,
) -> None:
    init_torch()
    from torch.utils.data import DataLoader

    config = load_training_config(checkpoint / "training.toml", seed=seed)
    init_logger(config.log_level, Path(f"./logs/{config.label}"))
    seed_everything(config.seed)
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
        frozen_encoders=True,
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=config_batch_size * 2,
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

    load_best_model(checkpoint, model)
    result = TrainingResult.auto_compute(model, test_data_loader)
    result.print()


@app.command()
def inference(
    checkpoint: Path,
    text: str | None = None,
    audio: Path | None = None,
    video: Path | None = None,
):
    from recognize.estimator import EmotionEstimator

    init_logger("INFO")

    model_checkpoint = checkpoint / str(find_best_model(checkpoint))
    estimator = EmotionEstimator(model_checkpoint)

    print("Predicted logits:", estimator.compute_logits(text=text, audio_path=audio, video_path=video).tolist())
    print("Predicted cls:", estimator.classify(text=text, audio_path=audio, video_path=video))


if __name__ == "__main__":
    app()
