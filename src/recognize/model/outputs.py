from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict


class ModelOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ClassifierOutput(ModelOutput):
    logits: torch.Tensor
    features: torch.Tensor
    embs_dict: dict[str, torch.Tensor] | None = None
    loss: torch.Tensor | None = None
