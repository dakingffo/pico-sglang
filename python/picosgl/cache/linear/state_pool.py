from __future__ import annotations

import torch


class LinearStatePool:
    """Per-request linear attention (GatedDeltaNet) states, indexed by req.table_idx.

    A unified circular buffer: the pool always has a depth dimension. Each request has a
    per-request slot pointer ``slots[table_idx]``; the invariant is that the slot pointed
    to holds the request's *committed* state.

        depth = 1      (non-MTP)  every write lands in slot 0, pointer stays 0 ->
                                 behavior identical to a single-depth pool.
        depth = K + 1  (MTP)      circular buffer holding the K+1 boundary snapshots of a
                                 verify round (C+1 .. C+K+1), so rejection rollback is a
                                 pure pointer arithmetic (``rollback_to``), zero memcpy.

    conv_state:      (num_linear_layers, depth, max_req + 1, conv_dim, kernel_size - 1)
    recurrent_state: (num_linear_layers, depth, max_req + 1, num_v_heads, head_k_dim, head_v_dim)

    The pool lives on ``ctx`` (sibling of ``ctx.kv_cache``). Slots must be zeroed when a
    request finishes so recycled ``table_idx`` values don't observe stale state.
    """

    def __init__(
        self,
        num_linear_layers: int,
        max_req: int,
        conv_dim: int,
        kernel_size: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        depth: int = 1,
    ) -> None:
        # +1 for the dummy request slot
        self.conv_state = torch.zeros(
            num_linear_layers, depth, max_req + 1, conv_dim, kernel_size - 1,
            device=device, dtype=dtype,
        )
        self.recurrent_state = torch.zeros(
            num_linear_layers, depth, max_req + 1, num_v_heads, head_k_dim, head_v_dim,
            device=device, dtype=dtype,
        )
        self.slots = torch.zeros(max_req + 1, dtype=torch.int32, device=device)
        self.depth = depth
        self.device = device
        self.dtype = dtype

    def reset(self, table_idx: int) -> None:
        self.conv_state[:, :, table_idx].zero_()
        self.recurrent_state[:, :, table_idx].zero_()
        self.slots[table_idx] = 0

    def rollback_to(self, reqs, num_sampled: int) -> None:
        if self.depth == 1:
            return
        for req in reqs:
            self.slots[req.table_idx] = (
                (self.slots[req.table_idx] + num_sampled) % self.depth
            )

    def advance_batch(self, reqs) -> None:
        if self.depth == 1:
            return
        for req in reqs:
            self.slots[req.table_idx] = (
                (self.slots[req.table_idx] + 1) % self.depth
            )
