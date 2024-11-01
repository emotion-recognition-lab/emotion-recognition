from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict, Field


class ModelOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ClassifierOutput(ModelOutput):
    logits: torch.Tensor
    pooler_output: tuple[torch.Tensor | None, ...] = Field(default_factory=tuple)
    loss: torch.Tensor | None = None
