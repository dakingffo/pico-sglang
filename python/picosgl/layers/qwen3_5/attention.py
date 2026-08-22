from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from picosgl.core import get_global_ctx
from picosgl.distributed import DistributedInfo, try_get_tp_info
from picosgl.layers import BaseOP, LinearColParallelMerged, LinearRowParallel
from picosgl.utils import div_even, nvtx_annotate

from .norm import Qwen3_5RMSNorm
from .rotary import Qwen3_5RotaryEmbedding

if TYPE_CHECKING:
    from picosgl.models.config import ModelConfig

# Fallback when TP info is unset (unit tests build layers without an engine).
_TP_DEFAULT = DistributedInfo(rank=0, size=1)


class Qwen3_5Attention(BaseOP):
    """Full attention with attn_output_gate: q_proj emits 2*head_dim, gate = sigmoid on the 2nd half.

    ``paged=True`` routes through the flashinfer paged backend (KV pool, ``layer_id`` is the
    full-attention local index). ``paged=False`` uses eager torch attention (MTP verification).
    """

    def __init__(self, config: ModelConfig, layer_id: int, paged: bool = True):
        self.head_dim = config.head_dim
        # Projections are column/row-parallel, so per-rank head counts are the sharded
        # ones (forward reshapes use these) while projections get the FULL sizes and
        # shard internally (LinearColParallelMerged/LinearRowParallel).
        tp_size = (try_get_tp_info() or _TP_DEFAULT).size
        self.num_qo_heads = div_even(config.num_qo_heads, tp_size)
        self.num_kv_heads = div_even(config.num_kv_heads, tp_size, allow_replicate=True)
        self.scaling = self.head_dim**-0.5
        self._layer_id = layer_id
        self.paged = paged

        self.q_proj = LinearColParallelMerged(
            config.hidden_size, [config.num_qo_heads * self.head_dim * 2], has_bias=False
        )
        self.k_proj = LinearColParallelMerged(
            config.hidden_size, [config.num_kv_heads * self.head_dim], has_bias=False
        )
        self.v_proj = LinearColParallelMerged(
            config.hidden_size, [config.num_kv_heads * self.head_dim], has_bias=False
        )
        self.o_proj = LinearRowParallel(
            config.num_qo_heads * self.head_dim, config.hidden_size, has_bias=False
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
        o = o * torch.sigmoid(gate)
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
        o = o * torch.sigmoid(gate)
        o = self.o_proj.forward(o.reshape(-1, self.num_qo_heads * self.head_dim))
        return o, (k, v)

    def forward_with_kv_batch(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        valid_mask: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_valid_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        """Dense batched MTP attention.

        ``x`` is ``(B, T, hidden)`` and KV is stored as ``(B, heads, L, head_dim)``.
        ``valid_mask`` separates left-padded carry windows and inactive rows from real
        tokens. This path intentionally remains independent of the global paged-attention
        context used by the target model.
        """
        assert not self.paged
        assert x.ndim == 3 and positions.shape == valid_mask.shape == x.shape[:2]
        batch_size, seq_len = x.shape[:2]

        q_gate = self.q_proj.forward(x).view(
            batch_size, seq_len, self.num_qo_heads, self.head_dim * 2
        )
        query, gate = q_gate.chunk(2, dim=-1)
        key = self.k_proj.forward(x).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )
        value = self.v_proj.forward(x).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )

        query = self.q_norm.forward(query)
        key = self.k_norm.forward(key)
        query, key = self.rotary.forward(positions, query, key)

        n_rep = self.num_qo_heads // self.num_kv_heads
        if n_rep > 1:
            key = key.repeat_interleave(n_rep, dim=2)
            value = value.repeat_interleave(n_rep, dim=2)

        q = query.transpose(1, 2)  # (B, heads, T, head_dim)
        new_k = key.transpose(1, 2)
        new_v = value.transpose(1, 2)
        if past_kv is None:
            assert past_valid_mask is None
            k, v = new_k, new_v
            key_valid = valid_mask
            past_len = 0
        else:
            assert past_valid_mask is not None
            past_k, past_v = past_kv
            assert past_k.shape[:2] == (batch_size, self.num_qo_heads)
            assert past_valid_mask.shape == (batch_size, past_k.shape[2])
            k = torch.cat([past_k, new_k], dim=2)
            v = torch.cat([past_v, new_v], dim=2)
            key_valid = torch.cat([past_valid_mask, valid_mask], dim=1)
            past_len = past_k.shape[2]

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        key_len = k.shape[2]
        q_idx = torch.arange(seq_len, device=x.device).view(1, 1, seq_len, 1)
        k_idx = torch.arange(key_len, device=x.device).view(1, 1, 1, key_len)
        causal = k_idx <= past_len + q_idx
        allowed = causal & key_valid[:, None, None, :]

        # A padded query can otherwise have no finite key. Let it attend to its own
        # padded slot; its output is ignored, while this keeps the tensor free of NaNs.
        padded_self = (~valid_mask)[:, None, :, None] & (k_idx == past_len + q_idx)
        allowed |= padded_self
        attn = attn.masked_fill(~allowed, float("-inf"))
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
        o = torch.matmul(attn, v).transpose(1, 2)
        o = o * torch.sigmoid(gate)
        o = self.o_proj.forward(o.reshape(batch_size, seq_len, -1))
        return o, (k, v), key_valid

    def project_for_cache(
        self,
        x        : torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project normalized MTP rows without materializing an attention result.

        Returned K/V retain their native KV-head count and can be written directly to
        ``MTPKVPool``.  Query and gate are returned separately because the engine only
        evaluates the final canonical row while caching every newly accepted row.
        """
        assert not self.paged
        assert x.ndim == 2 and positions.shape == (x.shape[0],)
        q_gate = self.q_proj.forward(x).view(-1, self.num_qo_heads, self.head_dim * 2)
        query, gate = q_gate.chunk(2, dim=-1)
        key = self.k_proj.forward(x).view(-1, self.num_kv_heads, self.head_dim)
        value = self.v_proj.forward(x).view(-1, self.num_kv_heads, self.head_dim)

        query = self.q_norm.forward(query)
        key = self.k_norm.forward(key)
        query, key = self.rotary.forward(positions, query, key)
        return query, gate, key, value

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
