from __future__ import annotations

import torch


class LinearStatePool:
    """Per-request linear attention (GatedDeltaNet) states, indexed by req.table_idx.

    conv_state:      (num_linear_layers, max_req + 1, conv_dim, kernel_size - 1)
    recurrent_state: (num_linear_layers, max_req + 1, num_v_heads, head_k_dim, head_v_dim)

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
    ) -> None:
        # +1 for the dummy request slot
        self.conv_state = torch.zeros(
            num_linear_layers, max_req + 1, conv_dim, kernel_size - 1,
            device=device, dtype=dtype,
        )
        self.recurrent_state = torch.zeros(
            num_linear_layers, max_req + 1, num_v_heads, head_k_dim, head_v_dim,
            device=device, dtype=dtype,
        )
        self.device = device
        self.dtype = dtype

    def reset(self, table_idx: int) -> None:
        """Zero a request's states (called when the request is freed)."""
        self.conv_state[:, table_idx].zero_()
        self.recurrent_state[:, table_idx].zero_()
