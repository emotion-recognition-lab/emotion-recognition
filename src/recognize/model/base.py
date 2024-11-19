from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, overload

import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn import CrossEntropyLoss

from recognize.config import load_inference_config
from recognize.module import MoE, SparseMoE
from recognize.typing import FusionLayerLike

from .backbone import Backbone, MultimodalBackbone
from .inputs import ModelInput, MultimodalInput
from .outputs import ClassifierOutput


class ClassifierModel[T: ModelInput](nn.Module):
    __call__: Callable[[T], ClassifierOutput]

    @abstractmethod
    def forward(self, inputs: T) -> ClassifierOutput: ...

    def __init__(
        self,
        backbone: Backbone[T],
        feature_size: int,
        num_classes: int,
        *,
        num_experts: int = 1,
        act_expert_num: int | None = None,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.sample_weights = 1 / self.class_weights if self.class_weights is not None else None
        self.feature_size = feature_size
        if num_experts == 1:
            self.classifier = nn.Linear(feature_size, num_classes)
        elif act_expert_num is None or act_expert_num == num_experts:
            self.classifier = MoE(
                feature_size,
                [nn.Linear(feature_size, num_classes) for _ in range(num_experts)],
            )
        else:
            self.classifier = SparseMoE(
                feature_size,
                num_classes,
                [nn.Linear(feature_size, num_classes) for _ in range(num_experts)],
                act_expert_num=act_expert_num,
            )

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss_fn = CrossEntropyLoss(weight=self.sample_weights)
        return loss_fn(logits, labels)

    def classify(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> ClassifierOutput:
        logits = self.classifier(features)
        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels)
        return ClassifierOutput(logits=logits, features=features, loss=loss)

    def save_checkpoint(self, checkpoint_path: Path):
        model_state_dict = {key: value for key, value in self.state_dict().items() if not key.startswith("backbone.")}
        save_file(model_state_dict, checkpoint_path / "model.safetensors")


class MultimodalModel(ClassifierModel[MultimodalInput]):
    __call__: Callable[[MultimodalInput], ClassifierOutput]

    def __init__(
        self,
        backbone: Backbone[MultimodalInput],
        fusion_layer: FusionLayerLike,
        *,
        num_classes: int = 2,
        num_experts: int = 1,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__(
            backbone,
            fusion_layer.output_size,
            num_classes=num_classes,
            num_experts=num_experts,
            class_weights=class_weights,
        )
        self.fusion_layer = fusion_layer

    def forward(self, inputs: MultimodalInput) -> ClassifierOutput:
        embs_dict = self.backbone(inputs)
        if inputs.labels is None:
            fusion_features = self.fusion_layer(embs_dict)
            output = self.classify(fusion_features)
        else:
            fusion_features, fusion_loss = self.fusion_layer.forward_with_loss(embs_dict, inputs.labels)
            output = self.classify(fusion_features, inputs.labels)
            output.loss += fusion_loss
        output.embs_dict = embs_dict
        return output

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        backbone: MultimodalBackbone,
        fusion_layer: FusionLayerLike,
        *,
        class_weights: torch.Tensor | None = None,
    ):
        from recognize.utils import load_model

        checkpoint_path = Path(checkpoint_path)
        config = load_inference_config(checkpoint_path / "inference.toml")
        model = cls(
            backbone,
            fusion_layer,
            num_classes=config.num_classes,
            num_experts=config.model.num_experts,
            class_weights=class_weights,
        )
        load_model(checkpoint_path, model)

        return model


class UnimodalModel(ClassifierModel[MultimodalInput]):
    __call__: Callable[[MultimodalInput], ClassifierOutput]

    def __init__(
        self,
        backbone: Backbone[MultimodalInput],
        *,
        feature_size: int = 128,
        num_classes: int = 2,
        num_experts: int = 1,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__(
            backbone,
            feature_size,
            num_classes=num_classes,
            num_experts=num_experts,
            class_weights=class_weights,
        )

    def forward(self, inputs: MultimodalInput) -> ClassifierOutput:
        pooled_embs: dict[str, torch.Tensor] = self.backbone(inputs)
        if len(pooled_embs) != 1:
            # NOTE: dataset maybe not have some modalities
            assert inputs.labels is not None, "labels is required"
            return ClassifierOutput(
                logits=torch.zeros((inputs.labels.shape[0], self.num_classes), device=inputs.device),
                features=torch.zeros((inputs.labels.shape[0], self.feature_size), device=inputs.device),
            )
        features = next(iter(pooled_embs.values()))
        return self.classify(features, inputs.labels)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        backbone: MultimodalBackbone,
        *,
        class_weights: torch.Tensor | None = None,
    ):
        from recognize.utils import load_model

        checkpoint_path = Path(checkpoint_path)
        config = load_inference_config(checkpoint_path / "inference.toml")
        feature_size = next(iter(config.model.encoder.values())).feature_size
        model = cls(
            backbone,
            feature_size=feature_size,
            num_classes=config.num_classes,
            num_experts=config.model.num_experts,
            class_weights=class_weights,
        )
        load_model(checkpoint_path, model)
        return model


@overload
def add_extra_loss_fn[T: ClassifierModel[Any]](
    cls: type[T], *, loss_fn: Callable[[ClassifierOutput, torch.Tensor], torch.Tensor]
) -> type[T]: ...
@overload
def add_extra_loss_fn[T: ClassifierModel[Any]](
    cls: None = None, *, loss_fn: Callable[[ClassifierOutput, torch.Tensor], torch.Tensor]
) -> Callable[[type[T]], type[T]]: ...


def add_extra_loss_fn[T: ClassifierModel[Any]](
    cls: type[T] | None = None,
    *,
    loss_fn: Callable[[ClassifierOutput, torch.Tensor], torch.Tensor],
) -> type[T] | Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        class ClassifierModelWithExtraLoss(cls):
            def forward(self, inputs: ModelInput) -> ClassifierOutput:
                output = super().forward(inputs)  # type: ignore
                if output.loss is not None and inputs.labels is not None:
                    output.loss += loss_fn(output, inputs.labels)
                return output

        return ClassifierModelWithExtraLoss  # type: ignore

    if cls is None:
        return decorator
    else:
        return decorator(cls)
