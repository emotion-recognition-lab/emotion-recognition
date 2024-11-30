from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

from recognize.config import load_inference_config
from recognize.module import MoE, SparseMoE

# from recognize.module.loss import MultiLoss
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
        extra_loss_fns: Sequence[Callable[[T, ClassifierOutput], torch.Tensor]] | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.sample_weights = 1 / self.class_weights if self.class_weights is not None else None
        self.feature_size = feature_size
        self.extra_loss_fns = list(extra_loss_fns) if extra_loss_fns is not None else []
        # self.multi_loss_fn = MultiLoss(len(self.extra_loss_fns) + 1)
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

    def add_extra_loss_fn(self, loss_fn: Callable[[T, ClassifierOutput], torch.Tensor]) -> None:
        self.extra_loss_fns.append(loss_fn)
        # self.multi_loss_fn = MultiLoss(len(self.extra_loss_fns) + 1)

    def clear_extra_loss_fns(self) -> None:
        self.extra_loss_fns.clear()
        # self.multi_loss_fn = MultiLoss(1)

    def compute_loss(self, inputs: T, output: ClassifierOutput) -> torch.Tensor:
        logits = output.logits
        labels = inputs.labels
        assert labels is not None
        loss = F.cross_entropy(logits, labels, weight=self.sample_weights)
        if self.extra_loss_fns:
            loss += sum(fn(inputs, output) for fn in self.extra_loss_fns)
        output.loss = loss
        return loss

    def classify(self, features: torch.Tensor) -> ClassifierOutput:
        logits = self.classifier(features)
        return ClassifierOutput(logits=logits, features=features)

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
            output = self.classify(fusion_features)
            output.embs_dict = embs_dict
            self.compute_loss(inputs, output)
            output.loss += fusion_loss

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
        output = self.classify(features)
        self.compute_loss(inputs, output)
        return output

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
