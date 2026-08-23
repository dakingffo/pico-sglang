from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _recurrent_gated_delta_kernel(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    write_slots,
    state_pool,
    output,
    stride_qb,
    stride_qs,
    stride_qh,
    stride_qk,
    stride_vb,
    stride_vs,
    stride_vh,
    stride_vv,
    stride_gb,
    stride_gs,
    stride_gh,
    stride_ib,
    stride_ih,
    stride_ik,
    stride_iv,
    stride_wb,
    stride_ws,
    stride_ps,
    stride_ph,
    stride_pk,
    stride_pv,
    stride_ob,
    stride_os,
    stride_oh,
    stride_ov,
    NUM_HEADS : tl.constexpr,
    HEAD_K_DIM: tl.constexpr,
    HEAD_V_DIM: tl.constexpr,
    SEQ_LEN   : tl.constexpr,
    BLOCK_K   : tl.constexpr,
    BLOCK_V   : tl.constexpr,
):
    bh = tl.program_id(0)
    v_block = tl.program_id(1)
    batch_idx = bh // NUM_HEADS
    head_idx = bh % NUM_HEADS

    k_off = tl.arange(0, BLOCK_K)
    v_off = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
    state_mask = (k_off[:, None] < HEAD_K_DIM) & (v_off[None, :] < HEAD_V_DIM)
    initial_ptr = (
        initial_state
        + batch_idx * stride_ib
        + head_idx * stride_ih
        + k_off[:, None] * stride_ik
        + v_off[None, :] * stride_iv
    )
    state = tl.load(initial_ptr, mask=state_mask, other=0.0).to(tl.float32)
    scale = 1.0 / tl.sqrt(float(HEAD_K_DIM))

    for seq_idx in range(0, SEQ_LEN):
        q_ptr = (
            query
            + batch_idx * stride_qb
            + seq_idx * stride_qs
            + head_idx * stride_qh
            + k_off * stride_qk
        )
        k_ptr = (
            key
            + batch_idx * stride_qb
            + seq_idx * stride_qs
            + head_idx * stride_qh
            + k_off * stride_qk
        )
        v_ptr = (
            value
            + batch_idx * stride_vb
            + seq_idx * stride_vs
            + head_idx * stride_vh
            + v_off * stride_vv
        )
        gate_ptr = g + batch_idx * stride_gb + seq_idx * stride_gs + head_idx * stride_gh
        beta_ptr = beta + batch_idx * stride_gb + seq_idx * stride_gs + head_idx * stride_gh

        q = tl.load(q_ptr, mask=k_off < HEAD_K_DIM, other=0.0).to(tl.float32) * scale
        k = tl.load(k_ptr, mask=k_off < HEAD_K_DIM, other=0.0).to(tl.float32)
        v = tl.load(v_ptr, mask=v_off < HEAD_V_DIM, other=0.0).to(tl.float32)
        decay = tl.exp(tl.load(gate_ptr).to(tl.float32))
        beta_t = tl.load(beta_ptr).to(tl.float32)

        state *= decay
        kv_mem = tl.sum(state * k[:, None], axis=0)
        delta = (v - kv_mem) * beta_t
        state += k[:, None] * delta[None, :]
        out = tl.sum(state * q[:, None], axis=0)

        output_ptr = (
            output
            + batch_idx * stride_ob
            + seq_idx * stride_os
            + head_idx * stride_oh
            + v_off * stride_ov
        )
        tl.store(output_ptr, out, mask=v_off < HEAD_V_DIM)

        slot = tl.load(write_slots + batch_idx * stride_wb + seq_idx * stride_ws)
        state_ptr = (
            state_pool
            + slot * stride_ps
            + head_idx * stride_ph
            + k_off[:, None] * stride_pk
            + v_off[None, :] * stride_pv
        )
        tl.store(state_ptr, state, mask=state_mask)


def recurrent_gated_delta_triton(
    query       : torch.Tensor,
    key         : torch.Tensor,
    value       : torch.Tensor,
    g           : torch.Tensor,
    beta        : torch.Tensor,
    initial_state: torch.Tensor,
    write_slots : torch.Tensor,
    state_pool  : torch.Tensor,
) -> torch.Tensor:
    """Run a short GDN recurrence for a request batch and snapshot every step.

    query/key: ``(B, S, H, K)``; value: ``(B, S, H, V)``; initial_state:
    ``(B, H, K, V)``; write_slots: ``(B, S)``; state_pool: ``(slots, H, K, V)``.
    Query and key are expected to be L2-normalized by the caller.
    """
    assert query.is_cuda and key.is_cuda and value.is_cuda
    assert query.ndim == key.ndim == value.ndim == 4
    B, S, H, K = query.shape
    V = value.shape[-1]
    assert key.shape == query.shape
    assert value.shape[:3] == (B, S, H)
    assert g.shape == beta.shape == (B, S, H)
    assert initial_state.shape == (B, H, K, V)
    assert write_slots.shape == (B, S)
    assert state_pool.ndim == 4 and state_pool.shape[1:] == (H, K, V)
    assert write_slots.dtype in (torch.int32, torch.int64)

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    initial_state = initial_state.contiguous()
    write_slots = write_slots.contiguous()
    output = torch.empty((B, S, H, V), dtype=value.dtype, device=value.device)

    block_k = triton.next_power_of_2(K)
    block_v = 8 if V >= 8 else triton.next_power_of_2(V)
    grid = (B * H, triton.cdiv(V, block_v))
    _recurrent_gated_delta_kernel[grid](
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        write_slots,
        state_pool,
        output,
        *query.stride(),
        *value.stride(),
        *g.stride(),
        *initial_state.stride(),
        *write_slots.stride(),
        *state_pool.stride(),
        *output.stride(),
        NUM_HEADS=H,
        HEAD_K_DIM=K,
        HEAD_V_DIM=V,
        SEQ_LEN=S,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        num_warps=8 if block_k * block_v >= 1024 else 4,
    )
    return output
