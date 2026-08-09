from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from picosgl.layers import BaseOP
from picosgl.utils import nvtx_annotate

from .attention import Qwen3_5Attention
from .gated_delta_net import Qwen3_5GatedDeltaNet
from .mlp import Qwen3_5MLP
from .norm import Qwen3_5RMSNorm

if TYPE_CHECKING:
    from picosgl.models.config import ModelConfig


class Qwen3_5DecoderLayer(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        block_type: str | None = None,
        full_attn_idx: int = 0,
        linear_attn_idx: int = 0,
        paged: bool = True,
    ):
        self.block_type = block_type or config.layer_types[layer_id]
        self._layer_id = layer_id
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, linear_attn_idx)
        elif self.block_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, full_attn_idx, paged=paged)
        else:
            raise ValueError(f"Invalid layer type {self.block_type}")
        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.input_layernorm.forward(x)
        if self.block_type == "linear_attention":
            h = self.linear_attn.forward(h)
        else:
            h = self.self_attn.forward(h, positions)
        h = residual + h

        residual = h
        h = self.post_attention_layernorm.forward(h)
        h = self.mlp.forward(h)
        return residual + h
