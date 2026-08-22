from __future__ import annotations

import torch


class MTPKVPool:
    """Fixed-capacity KV storage for the single-layer MTP transformer.

    Every target request owns ``window_size`` persistent slots.  Draft batches share a
    second region with ``num_spec_tokens`` slots per batch row; that region is ephemeral
    and may be overwritten as soon as ``MTPEngine.draft`` returns.

    ``persistent_table`` and ``scratch_table`` contain physical token-slot indices.  The
    persistent rows are circular buffers, so advancing a full carry window never moves
    K/V data.
    """

    def __init__(
        self,
        max_running_req : int,
        window_size     : int,
        max_batch_size  : int,
        num_spec_tokens : int,
        num_kv_heads    : int,
        head_dim        : int,
        dtype           : torch.dtype,
        device          : torch.device,
    ) -> None:
        assert max_running_req > 0
        assert window_size > 0
        assert max_batch_size > 0
        assert num_spec_tokens > 0

        self.max_running_req = max_running_req
        self.window_size     = window_size
        self.max_batch_size  = max_batch_size
        self.num_spec_tokens = num_spec_tokens
        self.num_kv_heads    = num_kv_heads
        self.head_dim        = head_dim
        self.device          = device
        self.dtype           = dtype

        num_persistent = max_running_req * window_size
        num_scratch    = max_batch_size * num_spec_tokens
        self.num_slots = num_persistent + num_scratch

        self.persistent_table = torch.arange(
            num_persistent, dtype=torch.int32, device=device
        ).view(max_running_req, window_size)
        self.scratch_table = torch.arange(
            num_persistent,
            self.num_slots,
            dtype=torch.int32,
            device=device,
        ).view(max_batch_size, num_spec_tokens)

        self.k = torch.empty(
            self.num_slots, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        self.v = torch.empty_like(self.k)

        # Metadata is scheduler-side state.  Keeping these tiny counters on the host
        # avoids a synchronization merely to decide which fixed slots an append owns.
        self._heads   = [0] * max_running_req
        self._lengths = [0] * max_running_req

    def reset(self, table_idx: int) -> None:
        self._check_table_idx(table_idx)
        self._heads[table_idx]   = 0
        self._lengths[table_idx] = 0

    def append_persistent(self, table_idx: int, count: int) -> torch.Tensor:
        """Reserve ``count`` chronological slots and advance the request's ring."""
        self._check_table_idx(table_idx)
        assert 0 <= count <= self.window_size
        if count == 0:
            return self.persistent_table[table_idx, :0]

        head   = self._heads[table_idx]
        length = self._lengths[table_idx]
        columns: list[int] = []
        for _ in range(count):
            if length < self.window_size:
                column = (head + length) % self.window_size
                length += 1
            else:
                column = head
                head = (head + 1) % self.window_size
            columns.append(column)

        self._heads[table_idx]   = head
        self._lengths[table_idx] = length
        column_tensor = torch.tensor(columns, dtype=torch.int64, device=self.device)
        return self.persistent_table[table_idx].index_select(0, column_tensor)

    def persistent_indices(self, table_idx: int) -> torch.Tensor:
        """Return one request's valid persistent slots in chronological order."""
        self._check_table_idx(table_idx)
        head   = self._heads[table_idx]
        length = self._lengths[table_idx]
        if length == 0:
            return self.persistent_table[table_idx, :0]
        columns = (torch.arange(length, device=self.device) + head) % self.window_size
        return self.persistent_table[table_idx].index_select(0, columns)

    def scratch_slots(self, batch_rows: torch.Tensor, depth: int) -> torch.Tensor:
        """Return the physical slot at ``depth`` for each current draft-batch row."""
        assert batch_rows.ndim == 1
        assert 0 <= depth < self.num_spec_tokens
        assert batch_rows.numel() <= self.max_batch_size
        return self.scratch_table[batch_rows, depth]

    def batch_indices(
        self,
        table_indices: list[int],
        batch_rows    : torch.Tensor,
        scratch_depth: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build right-padded chronological KV indices for one attention call.

        ``scratch_depth`` is the number of already-materialized speculative rows to
        append after each request's persistent window.
        """
        batch_size = len(table_indices)
        assert batch_rows.shape == (batch_size,)
        assert batch_size <= self.max_batch_size
        assert 0 <= scratch_depth <= self.num_spec_tokens

        persistent = [self.persistent_indices(idx) for idx in table_indices]
        lengths     = [row.numel() + scratch_depth for row in persistent]
        max_length  = max(lengths)
        indices = torch.zeros(
            batch_size, max_length, dtype=torch.int32, device=self.device
        )
        valid = torch.zeros(
            batch_size, max_length, dtype=torch.bool, device=self.device
        )
        for i, row in enumerate(persistent):
            length = row.numel()
            indices[i, :length] = row
            if scratch_depth:
                indices[i, length : length + scratch_depth] = self.scratch_table[
                    batch_rows[i], :scratch_depth
                ]
            valid[i, : lengths[i]] = True
        return indices, valid

    def store(self, slots: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
        assert slots.ndim == 1
        assert key.shape == value.shape == (
            slots.numel(), self.num_kv_heads, self.head_dim
        )
        self.k.index_copy_(0, slots.to(torch.int64), key)
        self.v.index_copy_(0, slots.to(torch.int64), value)

    def cache_lengths(self, table_indices: list[int], scratch_depth: int = 0) -> list[int]:
        assert 0 <= scratch_depth <= self.num_spec_tokens
        return [self._lengths[idx] + scratch_depth for idx in table_indices]

    def _check_table_idx(self, table_idx: int) -> None:
        assert 0 <= table_idx < self.max_running_req, (
            f"MTP table_idx {table_idx} outside [0, {self.max_running_req})"
        )
