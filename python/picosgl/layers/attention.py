from __future__ import annotations

from typing import Any

import torch
from picosgl.core import get_global_ctx
from picosgl.distributed import get_tp_info
from picosgl.utils import div_even, nvtx_annotate

from .base import BaseOP, StateLessOP
from .linear import (
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearRowParallel,
)
from .norm import RMSNorm
from .rotary import get_rope

class AttentionLayer(StateLessOP):
    def __init__(
        self,
        layer_id     : int,
        num_qo_heads : int,
        num_kv_heads : int,
        head_dim     : int,
        rotary_dim   : int,
        max_position : int,
        rope_base    : float,
        rope_scaling : tuple[tuple[str, Any], ...] | None,
        q_norm       : RMSNorm | None = None,
        k_norm       : RMSNorm | None = None,
    ):
        assert num_qo_heads % num_kv_heads == 0
        self.layer_id = layer_id
        self.head_dim = head_dim
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(num_qo_heads, tp_size)
        self.num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position,
            base=rope_base,
            rope_scaling=rope_scaling,
        )
        self.q_norm = q_norm
        self.k_norm = k_norm

    def forward(
        self, 
        qkv: torch.Tensor # [N, D_qo + D_kv + D_kv]
    ) -> torch.Tensor:
        ctx = get_global_ctx()
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        q, k = self.rotary.forward(ctx.batch.positions, q, k)
        q = q.view(-1, self.num_qo_heads, self.head_dim) # [N, N_qo_head, D_qo_head]
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch) # [N, N_qo_head, D_qo_head]
        return o.view(-1, self.qo_attn_dim) # [N, D_qo]


class RotaryAttention(BaseOP):
    def __init__(
        self,
        hidden_size : int,
        head_dim    : int,
        num_qo_heads: int,
        num_kv_heads: int,
        layer_id    : int,
        rotary_dim  : int,
        max_position: int,
        rope_base   : float,
        rope_scaling: tuple[tuple[str, Any], ...] | None,
        *,
        has_attn_bias: bool = False,
        qk_norm_eps  : float | None = None,
    ):
        self.qkv_proj = LinearQKVMerged(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            has_bias=has_attn_bias,
        )
        if qk_norm_eps is not None:
            self.q_norm = RMSNorm(head_dim, eps=qk_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=qk_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            rotary_dim=rotary_dim,
            max_position=max_position,
            rope_base=rope_base,
            rope_scaling=rope_scaling,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        self.o_proj = LinearOProj(
            head_dim * num_qo_heads,
            hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj.forward(x)
        del x
        o = self.attn.forward(qkv)
        return self.o_proj.forward(o)


class GatedRotaryAttention(BaseOP):
    """Rotary attention whose query projection also produces an output gate."""

    def __init__(
        self,
        hidden_size       : int,
        head_dim          : int,
        num_qo_heads      : int,
        num_kv_heads      : int,
        layer_id          : int,
        rotary_dim        : int,
        max_position      : int,
        rope_base         : float,
        rope_scaling      : tuple[tuple[str, Any], ...] | None,
        qk_norm_eps       : float,
        zero_centered_norm: bool,
    ):
        tp_size = get_tp_info().size
        self.head_dim = head_dim
        self.num_qo_heads = div_even(num_qo_heads, tp_size)
        self.num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)
        self._layer_id = layer_id

        self.q_proj = LinearColParallelMerged(
            hidden_size, [num_qo_heads * head_dim * 2], has_bias=False
        )
        self.k_proj = LinearColParallelMerged(
            hidden_size, [num_kv_heads * head_dim], has_bias=False
        )
        self.v_proj = LinearColParallelMerged(
            hidden_size, [num_kv_heads * head_dim], has_bias=False
        )
        self.o_proj = LinearRowParallel(
            num_qo_heads * head_dim, hidden_size, has_bias=False
        )
        self.q_norm = RMSNorm(
            head_dim, eps=qk_norm_eps, zero_centered=zero_centered_norm
        )
        self.k_norm = RMSNorm(
            head_dim, eps=qk_norm_eps, zero_centered=zero_centered_norm
        )
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position,
            base=rope_base,
            rope_scaling=rope_scaling,
        )

    def _project(
        self,
        x        : torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = x.shape[:-1]
        q_gate = self.q_proj.forward(x).view(
            shape + (self.num_qo_heads, self.head_dim * 2)
        )
        query, gate = q_gate.chunk(2, dim=-1)
        key = self.k_proj.forward(x).view(
            shape + (self.num_kv_heads, self.head_dim)
        )
        value = self.v_proj.forward(x).view(
            shape + (self.num_kv_heads, self.head_dim)
        )
        query = self.q_norm.forward(query)
        key = self.k_norm.forward(key)
        query, key = self.rotary.forward(positions, query, key)
        return query, gate, key, value

    def _project_output(
        self,
        output: torch.Tensor,
        gate  : torch.Tensor,
    ) -> torch.Tensor:
        output = output * torch.sigmoid(gate)
        return self.o_proj.forward(output.flatten(-2))

    @nvtx_annotate("MHA", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        query, gate, key, value = self._project(x, positions)
        ctx = get_global_ctx()
        output = ctx.attn_backend.forward(
            query,
            key.flatten(-2),
            value.flatten(-2),
            self._layer_id,
            ctx.batch,
        )
        output = output.reshape(-1, self.num_qo_heads, self.head_dim)
        return self._project_output(output, gate)

    def project_for_cache(
        self,
        x        : torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x.ndim == 2 and positions.shape == (x.shape[0],)
        return self._project(x, positions)


__all__ = ["AttentionLayer", "GatedRotaryAttention", "RotaryAttention"]
