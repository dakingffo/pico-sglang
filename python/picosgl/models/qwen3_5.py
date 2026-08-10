from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from picosgl.core import get_global_ctx
from picosgl.layers import (
    BaseOP,
    LinearRowParallel,
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


class Qwen3_5MultiTokenPredictor(BaseOP):
    """MTP head. Reuses the main model's embed_tokens and lm_head. Not wired into the
    speculative-decoding scheduler: it only needs to load weights and forward standalone."""

    def __init__(self, config: ModelConfig, embed_tokens, lm_head):
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = LinearRowParallel(config.hidden_size * 2, config.hidden_size, has_bias=False)
        self.layers = OPList(
            [
                Qwen3_5DecoderLayer(
                    config, 0, block_type="full_attention", paged=False
                )
            ]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._embed_tokens = embed_tokens
        self._lm_head = lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        emb = self._embed_tokens.forward(input_ids)
        emb = self.pre_fc_norm_embedding.forward(emb)
        h = self.pre_fc_norm_hidden.forward(hidden_states)
        h = torch.cat([emb, h], dim=-1)
        h = self.fc.forward(h)
        h = self.layers.op_list[0].forward(h, positions)
        return self.norm.forward(h)

    def get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project MTP output to vocab logits via the shared lm_head."""
        module = self._lm_head.tied_embedding or self._lm_head
        return F.linear(hidden_states, module.weight, self._lm_head.bias)

    def draft(self, input_ids, positions, hidden_states, past_kv=None):
        """Single MTP draft step, returns (next_token, logits, hidden, kv).

        Step-0 (past_kv=None): feed the carry window (last W accepted tokens + their target
        hidden); the last row's logits predict the next token -> draft_0, and the window's
        KV is returned for carry. Step-j (j>=1): feed [draft_{j-1}] + this predictor's own
        hidden from step j-1, attending to the carried KV -> draft_j.

        next_token is greedy (argmax); p_draft is recovered from ``logits`` by the caller.
        """
        emb = self._embed_tokens.forward(input_ids)
        emb = self.pre_fc_norm_embedding.forward(emb)
        h = self.pre_fc_norm_hidden.forward(hidden_states)
        h = torch.cat([emb, h], dim=-1)
        h = self.fc.forward(h)

        layer = self.layers.op_list[0]
        residual = h
        h = layer.input_layernorm.forward(h)
        h, kv = layer.self_attn.forward_with_kv(h, positions, past_kv)
        h = residual + h
        residual = h
        h = layer.post_attention_layernorm.forward(h)
        h = layer.mlp.forward(h)
        h = residual + h
        h = self.norm.forward(h)

        logits = self.get_logits(h)
        return logits.argmax(dim=-1), logits, h, kv


class Qwen3_5ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig, paged: bool = True):
        self.model = Qwen3_5Model(config, paged=paged)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        if config.mtp_num_hidden_layers > 0:
            self.mtp = Qwen3_5MultiTokenPredictor(
                config, self.model.embed_tokens, self.lm_head
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
