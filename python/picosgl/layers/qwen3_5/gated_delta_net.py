from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from picosgl.core import get_global_ctx
from picosgl.layers import BaseOP, LinearColParallelMerged, LinearRowParallel
from picosgl.utils import nvtx_annotate

from .norm import Qwen3_5RMSNormGated

if TYPE_CHECKING:
    from picosgl.models.config import ModelConfig


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
    chunk_state_callback=None,
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
        # per-chunk boundary state (used by the hybrid prefix cache to cache states at
        # page boundaries). Only fires when the caller wants it.
        if chunk_state_callback is not None:
            chunk_state_callback(i, last_recurrent_state)

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


class _Conv1d(BaseOP):
    def __init__(self, in_channels: int, kernel_size: int):
        self.weight = torch.empty(in_channels, 1, kernel_size)
        self.kernel_size = kernel_size

    @property
    def in_channels(self) -> int:
        return self.weight.shape[0]


class Qwen3_5GatedDeltaNet(BaseOP):
    """Linear attention layer (Gated Delta Net). State is stored per-request in the
    ctx.linear_state, indexed by req.table_idx / linear-layer local index."""

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
        pool = ctx.linear_state
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
        elif batch.is_verify:
            # mini-prefill over each req's [old_bonus, drafts]; snapshots are written to
            # circular slots so rejection rollback is pure pointer arithmetic (no memcpy).
            out = torch.empty_like(x)
            offset = 0
            for req in batch.reqs:
                seq_len = req.extend_len
                seg = self._forward_verify_one(x[offset : offset + seq_len], req, pool, L)
                out[offset : offset + seq_len] = seg
                offset += seq_len
            return out
        else:
            return self._forward_decode(x, batch, pool, L)

    def _forward_prefill_one(
        self, x: torch.Tensor, req, pool, L: int
    ) -> torch.Tensor:
        ctx = get_global_ctx()
        state_table = ctx.state_table
        page_size = ctx.page_size
        seq_len = x.shape[0]
        use_state = req.cached_len > 0
        table_idx = req.table_idx
        C_start = req.cached_len
        # baseline = the state at the last processed position of the page holding C_start-1.
        # For a cache-hit prefill this is the borrowed tree slot; for a chunked continuation
        # it is the previous chunk's terminal state, written to this page by that chunk.
        if use_state:
            baseline_slot = int(state_table[table_idx, (C_start - 1) // page_size])

        mixed_qkv = self.in_proj_qkv.forward(x).transpose(0, 1).unsqueeze(0)  # (1, conv_dim, seq)
        z = self.in_proj_z.forward(x).reshape(1, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b.forward(x).unsqueeze(0)  # (1, seq, num_v_heads)
        a = self.in_proj_a.forward(x).unsqueeze(0)

        conv_in = mixed_qkv
        if use_state:
            conv_in = torch.cat(
                [pool.conv_state[baseline_slot, L].unsqueeze(0), conv_in], dim=-1
            )
        total_len = conv_in.shape[-1]
        new_conv_state = F.pad(conv_in, (self.state_len - total_len, 0))
        conv_out = F.silu(
            F.conv1d(
                conv_in, self.conv1d.weight, None,
                padding=self.conv_kernel_size - 1, groups=self.conv_dim,
            )
        )
        mixed_qkv = conv_out[:, :, :total_len][:, :, -seq_len:]

        # Snapshot each 64-chunk's boundary state into its page's slot. Chunk and page
        # boundaries are both multiples of 64, so the chunk boundary IS the page boundary
        # (page_size > 64 just means consecutive chunks overwrite the same page slot, and
        # the last one wins, which is the correct page-boundary state). The final chunk
        # uses new_conv_state[0] (== the generic slice for the non-padded case) so a
        # padded partial chunk still stores the state after the last real token.
        n_chunks = (seq_len + 63) // 64

        def chunk_state_cb(i: int, last_recurrent_state: torch.Tensor) -> None:
            # clamp the final (possibly padded) chunk to the last real token so the
            # snapshot lands in the last data page, never one page past it.
            end_pos = min(C_start + (i + 1) * 64, C_start + seq_len)
            page = (end_pos - 1) // page_size
            slot = int(state_table[table_idx, page])
            if i == n_chunks - 1:
                conv_slice = new_conv_state[0]
            elif use_state:
                conv_slice = conv_in[0, :, (i + 1) * 64 : (i + 1) * 64 + self.state_len]
            else:
                conv_slice = conv_in[0, :, (i + 1) * 64 - self.state_len : (i + 1) * 64]
            pool.conv_state[slot, L].copy_(conv_slice)
            pool.recurrent_state[slot, L].copy_(last_recurrent_state[0])

        query, key, value = torch.split(mixed_qkv.transpose(1, 2), [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(1, seq_len, -1, self.head_k_dim)
        key = key.reshape(1, seq_len, -1, self.head_k_dim)
        value = value.reshape(1, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        # recurrent_state[baseline_slot, L] is (num_v_heads, head_k_dim, head_v_dim); the
        # rule expects (batch_size, num_heads, k_head_dim, v_head_dim).
        initial_state = (
            pool.recurrent_state[baseline_slot, L].unsqueeze(0) if use_state else None
        )
        core_attn_out, _last_state = _chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=initial_state, output_final_state=True, use_qk_l2norm_in_kernel=True,
            chunk_state_callback=chunk_state_cb,
        )
        # NOTE: the final chunk's boundary state was already written by the callback.

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm.forward(core_attn_out, z).reshape(1, seq_len, -1)
        return self.out_proj.forward(core_attn_out).reshape(seq_len, -1)

    def _forward_verify_one(
        self, x: torch.Tensor, req, pool, L: int
    ) -> torch.Tensor:
        """Spec-decode verify step: process a request's n_drafts+1 tokens
        [old_bonus, draft_0..draft_{n_drafts-1}] as a mini-prefill, writing each token's
        post-state to the reserve slots R[0..K] (indexed via the state_table tail columns).

        The baseline (req.baseline_slot) may be a reserve slot (R[0] for rounds after the
        first) that candidate 0 also writes to; that is safe because the baseline is only
        read at round start, into locals, before any candidate write. Committing the
        accepted boundary is done by the scheduler via pure index ops (no state copy).
        Math == n_drafts+1 decode rounds (per-token recurrent rule), which test2 already
        equates to the chunked prefill rule.
        """
        seq_len = x.shape[0]  # n_drafts + 1
        table_idx = req.table_idx
        ctx = get_global_ctx()
        state_table = ctx.state_table
        rb = ctx.draft_state
        assert rb is not None
        baseline_slot = req.baseline_slot
        reserve_slots = [
            int(s) for s in state_table[table_idx, rb : rb + seq_len]
        ]

        mixed_qkv = self.in_proj_qkv.forward(x).transpose(0, 1).unsqueeze(0)  # (1, conv_dim, seq)
        z = self.in_proj_z.forward(x).reshape(1, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b.forward(x).unsqueeze(0)  # (1, seq, num_v_heads)
        a = self.in_proj_a.forward(x).unsqueeze(0)

        conv_in = torch.cat(
            [pool.conv_state[baseline_slot, L].unsqueeze(0), mixed_qkv], dim=-1
        )
        total_len = conv_in.shape[-1]
        conv_out = F.silu(
            F.conv1d(
                conv_in, self.conv1d.weight, None,
                padding=self.conv_kernel_size - 1, groups=self.conv_dim,
            )
        )
        mixed_qkv = conv_out[:, :, :total_len][:, :, -seq_len:]
        # conv_state after token j = the last state_len entries of [baseline | tokens 0..j]
        # = conv_in[:, :, j+1 : state_len+j+1]  (direct slice, no sequential conv).
        for j in range(seq_len):
            pool.conv_state[reserve_slots[j], L].copy_(
                conv_in[0, :, j + 1 : self.state_len + j + 1]
            )

        query, key, value = torch.split(mixed_qkv.transpose(1, 2), [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(1, seq_len, -1, self.head_k_dim)
        key = key.reshape(1, seq_len, -1, self.head_k_dim)
        value = value.reshape(1, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        # Per-token recurrent rule (n_drafts+1 sequential steps, exactly like n_drafts+1 decode
        # rounds). The rule never mutates its input, so the baseline can be passed as a
        # view; each step's final state is snapshotted to its circular slot.
        # per-token output is (num_v_heads, head_v_dim); flatten like the chunked rule
        # (which yields (1, seq, num_v_heads, head_v_dim)) before norm/out_proj.
        core_attn_out = torch.empty(
            seq_len, self.num_v_heads, self.head_v_dim, device=x.device, dtype=x.dtype
        )
        state = pool.recurrent_state[baseline_slot, L]
        for j in range(seq_len):
            out_j, state_j = _recurrent_gated_delta_rule(
                query[:, j : j + 1], key[:, j : j + 1], value[:, j : j + 1],
                g=g[:, j : j + 1], beta=beta[:, j : j + 1],
                initial_state=state.unsqueeze(0), output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            state = state_j[0]  # (num_v_heads, k_dim, v_dim)
            core_attn_out[j] = out_j[0, 0]
            pool.recurrent_state[reserve_slots[j], L].copy_(state)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm.forward(core_attn_out, z).reshape(1, seq_len, -1)
        return self.out_proj.forward(core_attn_out).reshape(seq_len, -1)

    def _forward_decode(self, x: torch.Tensor, batch, pool, L: int) -> torch.Tensor:
        ctx = get_global_ctx()
        state_table = ctx.state_table
        page_size = ctx.page_size
        bs = batch.size
        table_idxs = torch.tensor([r.table_idx for r in batch.reqs], device=x.device)
        # At decode-forward time cached_len = device_len - 1 = the position being processed.
        # Read the state after cached_len-1 (written by the previous forward / prefill
        # chunk) and write the state after cached_len into the page the new token lands in.
        read_pages = torch.tensor(
            [(r.cached_len - 1) // page_size for r in batch.reqs], device=x.device
        )
        write_pages = torch.tensor(
            [r.cached_len // page_size for r in batch.reqs], device=x.device
        )
        read_slots = state_table[table_idxs, read_pages]   # (bs,) int32 [copy]
        write_slots = state_table[table_idxs, write_pages]
        conv_state = pool.conv_state[read_slots, L]  # (bs, conv_dim, state_len) [copy]
        recurrent_state = pool.recurrent_state[read_slots, L]

        mixed_qkv = self.in_proj_qkv.forward(x.unsqueeze(1)).transpose(1, 2)  # (bs, conv_dim, 1)
        z = self.in_proj_z.forward(x).reshape(bs, 1, -1, self.head_v_dim)
        b = self.in_proj_b.forward(x).unsqueeze(1)  # (bs, 1, num_v_heads)
        a = self.in_proj_a.forward(x).unsqueeze(1)

        mixed_qkv = _causal_conv1d_update(mixed_qkv, conv_state, self.conv1d.weight)
        # NOTE: pool[... , tensor_idx] is advanced indexing -> a COPY, so `.copy_()` on it
        # would never reach the pool. Write back per request with a scalar index (a view),
        # exactly like the prefill path. The write page (hence slot) was allocated by the
        # scheduler before this forward (allocate_state_pages).
        new_slots_list = write_slots.tolist()
        for b_i, tidx in enumerate(table_idxs.tolist()):
            pool.conv_state[new_slots_list[b_i], L].copy_(conv_state[b_i])

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
            pool.recurrent_state[new_slots_list[b_i], L].copy_(last_state[b_i])

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm.forward(core_attn_out, z).reshape(bs, 1, -1)
        return self.out_proj.forward(core_attn_out).reshape(bs, -1)
