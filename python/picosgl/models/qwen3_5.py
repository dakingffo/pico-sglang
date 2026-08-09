from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from picosgl.core import get_global_ctx
from picosgl.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearRowParallel,
    OPList,
    ParallelLMHead,
    StateLessOP,
    VocabParallelEmbedding,
)
from picosgl.utils import nvtx_annotate

from .base import BaseLLMModel

if TYPE_CHECKING:
    from .config import ModelConfig

# =====================================================================================
# Torch reference math (ported from transformers modeling_qwen3_5.py).  These are
# bit-for-bit compatible with the transformers torch fallback so that logits can be
# compared against it directly.
# =====================================================================================


def _causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Single-step causal depthwise conv update.

    hidden_states: (bs, conv_dim, 1); conv_state: (bs, conv_dim, kernel-1); weight: (conv_dim, 1, kernel).
    Updates ``conv_state`` in place and returns the conv output.
    """
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight, None, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    return out.to(hidden_states.dtype)


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device
        )
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2)
            @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim, dtype=value.dtype, device=value.device
    )
    last_recurrent_state = (
        torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device
        )
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# =====================================================================================
# Norms / rotary
# =====================================================================================


class Qwen3_5RMSNorm(BaseOP):
    """x * rsqrt(mean(x^2)+eps) * (1.0 + weight), weight initialized to zeros."""

    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = torch.zeros(dim)

    @nvtx_annotate("RMSNorm")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        output = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class Qwen3_5RMSNormGated(BaseOP):
    """rmsnorm(x) * weight * silu(gate), weight initialized to ones. Per head_v_dim."""

    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        # fp32 in the checkpoint (mamba_ssm_dtype=float32), mirrors Qwen3_5RMSNormGated
        self.weight = torch.ones(dim, dtype=torch.float32)

    @nvtx_annotate("RMSNormGated")
    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states * self.weight.float()
        hidden_states = hidden_states * F.silu(gate.float())
        return hidden_states.to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Qwen3_5RotaryEmbedding(StateLessOP):
    """Partial / interleaved RoPE (Qwen3.5 mrope).

    For pure text the T/H/W positions are identical, so the interleaved layout collapses to
    the standard RoPE applied on the first ``rotary_dim`` dims, with the rest passed through.
    """

    def __init__(self, head_dim: int, rotary_dim: int, max_position: int, base: float):
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self._max_position = max_position
        self._base = base
        self._cos_sin_cache: torch.Tensor | None = None

    def _get_cache(self, device: torch.device) -> torch.Tensor:
        cache = self._cos_sin_cache
        if cache is None or cache.device != device:
            # Inv-freqs are computed on the target device (the module is built on meta).
            inv_freq = 1.0 / (
                self._base
                ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=device) / self.rotary_dim)
            )
            t = torch.arange(self._max_position, device=device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)  # (max_pos, rotary_dim)
            cache = torch.cat([emb.cos(), emb.sin()], dim=-1)  # (max_pos, 2*rotary_dim)
            self._cos_sin_cache = cache
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self._get_cache(query.device)
        cos = cache[positions, : self.rotary_dim].unsqueeze(1)  # (T, 1, rotary_dim)
        sin = cache[positions, self.rotary_dim :].unsqueeze(1)
        q_rot, q_pass = query[..., : self.rotary_dim], query[..., self.rotary_dim :]
        k_rot, k_pass = key[..., : self.rotary_dim], key[..., self.rotary_dim :]
        q_embed = (q_rot * cos + _rotate_half(q_rot) * sin).to(query.dtype)
        k_embed = (k_rot * cos + _rotate_half(k_rot) * sin).to(key.dtype)
        return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


# =====================================================================================
# MLP / GatedDeltaNet / Attention
# =====================================================================================


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


class _Conv1d(BaseOP):
    def __init__(self, in_channels: int, kernel_size: int):
        self.weight = torch.empty(in_channels, 1, kernel_size)
        self.kernel_size = kernel_size

    @property
    def in_channels(self) -> int:
        return self.weight.shape[0]


class Qwen3_5GatedDeltaNet(BaseOP):
    """Linear attention layer (Gated Delta Net). State is stored per-request in the
    ctx.linear_state_pool, indexed by req.table_idx / linear-layer local index."""

    def __init__(self, config: ModelConfig, linear_layer_idx: int):
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.state_len = self.conv_kernel_size - 1
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self._linear_layer_idx = linear_layer_idx

        self.conv1d = _Conv1d(self.conv_dim, self.conv_kernel_size)
        self.dt_bias = torch.ones(self.num_v_heads)
        # A_log / gated-norm weight are fp32 in the checkpoint (mamba_ssm_dtype=float32)
        self.A_log = torch.zeros(self.num_v_heads, dtype=torch.float32)
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = LinearRowParallel(self.value_dim, config.hidden_size, has_bias=False)

        self.in_proj_qkv = LinearColParallelMerged(
            config.hidden_size, [self.key_dim * 2 + self.value_dim], has_bias=False
        )
        self.in_proj_z = LinearColParallelMerged(
            config.hidden_size, [self.value_dim], has_bias=False
        )
        self.in_proj_b = LinearColParallelMerged(
            config.hidden_size, [self.num_v_heads], has_bias=False
        )
        self.in_proj_a = LinearColParallelMerged(
            config.hidden_size, [self.num_v_heads], has_bias=False
        )

    @nvtx_annotate("GatedDeltaNet", layer_id_field="_linear_layer_idx")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        L = self._linear_layer_idx

        if batch.is_prefill:
            out = torch.empty_like(x)
            offset = 0
            for req in batch.reqs:
                seq_len = req.extend_len
                seg = self._forward_prefill_one(x[offset : offset + seq_len], req, pool, L)
                out[offset : offset + seq_len] = seg
                offset += seq_len
            return out
        else:
            return self._forward_decode(x, batch, pool, L)

    def _forward_prefill_one(
        self, x: torch.Tensor, req, pool, L: int
    ) -> torch.Tensor:
        seq_len = x.shape[0]
        use_state = req.cached_len > 0
        table_idx = req.table_idx

        mixed_qkv = self.in_proj_qkv.forward(x).transpose(0, 1).unsqueeze(0)  # (1, conv_dim, seq)
        z = self.in_proj_z.forward(x).reshape(1, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b.forward(x).unsqueeze(0)  # (1, seq, num_v_heads)
        a = self.in_proj_a.forward(x).unsqueeze(0)

        conv_in = mixed_qkv
        if use_state:
            conv_in = torch.cat([pool.conv_state[L, table_idx].unsqueeze(0), conv_in], dim=-1)
        total_len = conv_in.shape[-1]
        new_conv_state = F.pad(conv_in, (self.state_len - total_len, 0))
        conv_out = F.silu(
            F.conv1d(
                conv_in, self.conv1d.weight, None,
                padding=self.conv_kernel_size - 1, groups=self.conv_dim,
            )
        )
        mixed_qkv = conv_out[:, :, :total_len][:, :, -seq_len:]
        pool.conv_state[L, table_idx].copy_(new_conv_state[0])

        query, key, value = torch.split(mixed_qkv.transpose(1, 2), [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(1, seq_len, -1, self.head_k_dim)
        key = key.reshape(1, seq_len, -1, self.head_k_dim)
        value = value.reshape(1, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        # recurrent_state[L, table_idx] is (num_v_heads, head_k_dim, head_v_dim); the rule
        # expects (batch_size, num_heads, k_head_dim, v_head_dim).
        initial_state = (
            pool.recurrent_state[L, table_idx].unsqueeze(0) if use_state else None
        )
        core_attn_out, last_state = _chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=initial_state, output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
        pool.recurrent_state[L, table_idx].copy_(last_state[0])

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm.forward(core_attn_out, z).reshape(1, seq_len, -1)
        return self.out_proj.forward(core_attn_out).reshape(seq_len, -1)

    def _forward_decode(self, x: torch.Tensor, batch, pool, L: int) -> torch.Tensor:
        bs = batch.size
        table_idxs = torch.tensor([r.table_idx for r in batch.reqs], device=x.device)
        conv_state = pool.conv_state[L, table_idxs]  # (bs, conv_dim, state_len) [copy]
        recurrent_state = pool.recurrent_state[L, table_idxs]

        mixed_qkv = self.in_proj_qkv.forward(x.unsqueeze(1)).transpose(1, 2)  # (bs, conv_dim, 1)
        z = self.in_proj_z.forward(x).reshape(bs, 1, -1, self.head_v_dim)
        b = self.in_proj_b.forward(x).unsqueeze(1)  # (bs, 1, num_v_heads)
        a = self.in_proj_a.forward(x).unsqueeze(1)

        mixed_qkv = _causal_conv1d_update(mixed_qkv, conv_state, self.conv1d.weight)
        # NOTE: pool[... , tensor_idx] is advanced indexing -> a COPY, so `.copy_()` on it
        # would never reach the pool. Write back per request with a scalar index (a view),
        # exactly like the prefill path.
        for b_i, tidx in enumerate(table_idxs.tolist()):
            pool.conv_state[L, tidx].copy_(conv_state[b_i])

        query, key, value = torch.split(mixed_qkv.transpose(1, 2), [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(bs, 1, -1, self.head_k_dim)
        key = key.reshape(bs, 1, -1, self.head_k_dim)
        value = value.reshape(bs, 1, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        core_attn_out, last_state = _recurrent_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=recurrent_state, output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
        for b_i, tidx in enumerate(table_idxs.tolist()):
            pool.recurrent_state[L, tidx].copy_(last_state[b_i])

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm.forward(core_attn_out, z).reshape(bs, 1, -1)
        return self.out_proj.forward(core_attn_out).reshape(bs, -1)


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
            o = self._eager_attention(query, key, value)
        o = o.reshape(-1, self.num_qo_heads, self.head_dim)
        o = o * torch.sigmoid(gate)
        return self.o_proj.forward(o.reshape(-1, self.num_qo_heads * self.head_dim))

    def _eager_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        # eager layout is (T, heads, head_dim): repeat kv heads along dim=1
        n_rep = self.num_qo_heads // self.num_kv_heads
        if n_rep > 1:
            key = key.repeat_interleave(n_rep, dim=1)
            value = value.repeat_interleave(n_rep, dim=1)
        seq_len = query.shape[0]
        q, k, v = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)  # (heads, T, hd)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scaling  # (heads, T, T)
        mask = torch.triu(
            torch.ones(1, seq_len, seq_len, dtype=torch.bool, device=query.device), diagonal=1
        )
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(attn, v).transpose(0, 1)  # back to (T, heads, hd)


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


__all__ = ["Qwen3_5ForCausalLM"]
