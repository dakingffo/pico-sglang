from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from picosgl.core import get_global_ctx
from picosgl.layers import (
    BaseOP,
    GatedDeltaNet,
    GatedMLP,
    GatedRotaryAttention,
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
)
from picosgl.utils import nvtx_annotate

from .base import BaseLLMModel

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen3_5DecoderLayer(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        block_type     : str | None = None,
        full_attn_idx  : int = 0,
        linear_attn_idx: int = 0,
    ):
        self.block_type = block_type or config.layer_types[layer_id]
        self._layer_id = layer_id
        if self.block_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_key_heads=config.linear_num_key_heads,
                num_value_heads=config.linear_num_value_heads,
                head_k_dim=config.linear_key_head_dim,
                head_v_dim=config.linear_value_head_dim,
                conv_kernel_size=config.linear_conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_idx=linear_attn_idx,
            )
        elif self.block_type == "full_attention":
            rotary_config = config.rotary_config
            self.self_attn = GatedRotaryAttention(
                hidden_size=config.hidden_size,
                head_dim=config.head_dim,
                num_qo_heads=config.num_qo_heads,
                num_kv_heads=config.num_kv_heads,
                layer_id=full_attn_idx,
                rotary_dim=rotary_config.rotary_dim,
                max_position=rotary_config.max_position,
                rope_base=rotary_config.base,
                rope_scaling=(
                    tuple(rotary_config.scaling.items())
                    if rotary_config.scaling else None
                ),
                qk_norm_eps=config.rms_norm_eps,
                zero_centered_norm=True,
            )
        else:
            raise ValueError(f"Invalid layer type {self.block_type}")
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, zero_centered=True
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, zero_centered=True
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


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        layers = []
        full_idx = 0
        linear_idx = 0
        for i, layer_type in enumerate(config.layer_types):
            if layer_type == "full_attention":
                layers.append(
                    Qwen3_5DecoderLayer(config, i, full_attn_idx=full_idx)
                )
                full_idx += 1
            else:
                layers.append(
                    Qwen3_5DecoderLayer(config, i, linear_attn_idx=linear_idx)
                )
                linear_idx += 1
        self.layers = OPList(layers)
        self.norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, zero_centered=True
        )

    @nvtx_annotate("Model")
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        positions = get_global_ctx().batch.positions
        for layer in self.layers.op_list:
            x = layer.forward(x, positions)
        return self.norm.forward(x)


class Qwen3_5ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
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


__all__ = ["Qwen3_5DecoderLayer", "Qwen3_5ForCausalLM"]
