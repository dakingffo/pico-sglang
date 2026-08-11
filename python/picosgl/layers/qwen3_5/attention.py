from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from picosgl.core import get_global_ctx
from picosgl.layers import BaseOP, LinearColParallelMerged, LinearRowParallel
from picosgl.utils import nvtx_annotate

from .norm import Qwen3_5RMSNorm
from .rotary import Qwen3_5RotaryEmbedding

if TYPE_CHECKING:
    from picosgl.models.config import ModelConfig


class Qwen3_5Attention(BaseOP):
    """Full attention with attn_output_gate: q_proj emits 2*head_dim, gate = sigmoid on the 2nd half.

    ``paged=True`` routes through the flashinfer paged backend (KV pool, ``layer_id`` is the
    full-attention local index). ``paged=False`` uses eager torch attention (MTP verification).
    """

    def __init__(self, config: ModelConfig, layer_id: int, paged: bool = True):
        self.head_dim = config.head_dim
        self.num_qo_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self._layer_id = layer_id
        self.paged = paged
        self.output_gate_type = config.output_gate_type

        self.q_proj = LinearColParallelMerged(
            config.hidden_size, [self.num_qo_heads * self.head_dim * 2], has_bias=False
        )
        self.k_proj = LinearColParallelMerged(
            config.hidden_size, [self.num_kv_heads * self.head_dim], has_bias=False
        )
        self.v_proj = LinearColParallelMerged(
            config.hidden_size, [self.num_kv_heads * self.head_dim], has_bias=False
        )
        self.o_proj = LinearRowParallel(
            self.num_qo_heads * self.head_dim, config.hidden_size, has_bias=False
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary = Qwen3_5RotaryEmbedding(
            self.head_dim,
            config.rotary_config.rotary_dim,
            config.rotary_config.max_position,
            config.rotary_config.base,
        )

    @nvtx_annotate("MHA", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        q_gate = self.q_proj.forward(x).view(-1, self.num_qo_heads, self.head_dim * 2)
        query, gate = q_gate.chunk(2, dim=-1)
        key = self.k_proj.forward(x).view(-1, self.num_kv_heads, self.head_dim)
        value = self.v_proj.forward(x).view(-1, self.num_kv_heads, self.head_dim)

        query = self.q_norm.forward(query)
        key = self.k_norm.forward(key)
        query, key = self.rotary.forward(positions, query, key)

        if self.paged:
            # flashinfer backend stores KV via a 2D kernel: (T, num_kv_heads * head_dim)
            o = get_global_ctx().attn_backend.forward(
                query,
                key.reshape(-1, self.num_kv_heads * self.head_dim),
                value.reshape(-1, self.num_kv_heads * self.head_dim),
                self._layer_id,
                get_global_ctx().batch,
            )
        else:
            o, _, _ = self._eager_attention(query, key, value)
        o = o.reshape(-1, self.num_qo_heads, self.head_dim)
        # Qwen3.5 gates with sigmoid; Qwen3.6 (output_gate_type="swish") uses gate*sigmoid.
        # The sigmoid branch is bit-identical to the old hardcoded path.
        o = o * (F.silu(gate) if self.output_gate_type == "swish" else torch.sigmoid(gate))
        return self.o_proj.forward(o.reshape(-1, self.num_qo_heads * self.head_dim))

    def forward_with_kv(
        self, x: torch.Tensor, positions: torch.Tensor, past_kv=None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Same as ``forward`` but returns (output, (k, v)) for the MTP draft carry.

        paged=False only (the MTP head's single decoder layer). ``past_kv`` is the
        accumulated window KV from prior draft steps in (heads, L, head_dim) layout.
        """
        assert not self.paged
        q_gate = self.q_proj.forward(x).view(-1, self.num_qo_heads, self.head_dim * 2)
        query, gate = q_gate.chunk(2, dim=-1)
        key = self.k_proj.forward(x).view(-1, self.num_kv_heads, self.head_dim)
        value = self.v_proj.forward(x).view(-1, self.num_kv_heads, self.head_dim)

        query = self.q_norm.forward(query)
        key = self.k_norm.forward(key)
        query, key = self.rotary.forward(positions, query, key)

        past_k = past_v = None
        if past_kv is not None:
            past_k, past_v = past_kv
        o, k, v = self._eager_attention(query, key, value, past_k, past_v)
        o = o.reshape(-1, self.num_qo_heads, self.head_dim)
        # Qwen3.5 gates with sigmoid; Qwen3.6 (output_gate_type="swish") uses gate*sigmoid.
        # The sigmoid branch is bit-identical to the old hardcoded path.
        o = o * (F.silu(gate) if self.output_gate_type == "swish" else torch.sigmoid(gate))
        o = self.o_proj.forward(o.reshape(-1, self.num_qo_heads * self.head_dim))
        return o, (k, v)

    def _eager_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        past_k: torch.Tensor | None = None,
        past_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eager causal attention. Returns (output, k, v) with k/v in (heads, L, head_dim)
        layout (the accumulated sequence incl. ``past_k/v``, for MTP draft carry)."""
        # eager layout is (T, heads, head_dim): repeat kv heads along dim=1
        n_rep = self.num_qo_heads // self.num_kv_heads
        if n_rep > 1:
            key = key.repeat_interleave(n_rep, dim=1)
            value = value.repeat_interleave(n_rep, dim=1)
        seq_len = query.shape[0]
        q = query.transpose(0, 1)  # (heads, T, hd)
        k = key.transpose(0, 1) if past_k is None else torch.cat([past_k, key.transpose(0, 1)], dim=1)
        v = value.transpose(0, 1) if past_v is None else torch.cat([past_v, value.transpose(0, 1)], dim=1)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scaling  # (heads, T, L)
        # query i attends to positions <= past_len + i; diagonal = L - T + 1 (== 1 when no past)
        mask = torch.triu(
            torch.ones(1, seq_len, k.shape[1], dtype=torch.bool, device=query.device),
            diagonal=k.shape[1] - seq_len + 1,
        )
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
        o = torch.matmul(attn, v).transpose(0, 1)  # back to (T, heads, hd)
        return o, k, v
