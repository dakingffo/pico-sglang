from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from picosgl.layers import BaseOP, LinearColParallelMerged, LinearRowParallel
from picosgl.utils import nvtx_annotate

if TYPE_CHECKING:
    from picosgl.models.config import ModelConfig


class Qwen3_5MLP(BaseOP):
    """gate/up/down with independent projections (Qwen3.5 checkpoint is not merged)."""

    def __init__(self, config: ModelConfig):
        self.gate_proj = LinearColParallelMerged(
            config.hidden_size, [config.intermediate_size], has_bias=False
        )
        self.up_proj = LinearColParallelMerged(
            config.hidden_size, [config.intermediate_size], has_bias=False
        )
        self.down_proj = LinearRowParallel(
            config.intermediate_size, config.hidden_size, has_bias=False
        )

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(F.silu(self.gate_proj.forward(x)) * self.up_proj.forward(x))
