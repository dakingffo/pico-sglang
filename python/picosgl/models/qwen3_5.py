from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from picosgl.core import get_global_ctx
from picosgl.layers import (
    BaseOP,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from picosgl.layers.qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm
from picosgl.utils import nvtx_annotate

from .base import BaseLLMModel

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig, paged: bool = True):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        layers = []
        full_idx = 0
        linear_idx = 0
        for i, layer_type in enumerate(config.layer_types):
            if layer_type == "full_attention":
                layers.append(
                    Qwen3_5DecoderLayer(config, i, full_attn_idx=full_idx, paged=paged)
                )
                full_idx += 1
            else:
                layers.append(
                    Qwen3_5DecoderLayer(config, i, linear_attn_idx=linear_idx)
                )
                linear_idx += 1
        self.layers = OPList(layers)
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @nvtx_annotate("Model")
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        positions = get_global_ctx().batch.positions
        for layer in self.layers.op_list:
            x = layer.forward(x, positions)
        return self.norm.forward(x)


class Qwen3_5ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig, paged: bool = True):
        self.model = Qwen3_5Model(config, paged=paged)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits

    def forward_verify(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward returning (full_hidden, logits).

        Used for verify batches (full per-position logits, since is_prefill=False) and, in
        MTP mode, for prefill batches too (last-token logits via the LMHead gather) so the
        prefill->verify handoff can capture the full hidden states.
        """
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return output, logits


__all__ = ["Qwen3_5ForCausalLM"]
