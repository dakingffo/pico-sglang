from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from picosgl.core import Request, get_global_ctx


@dataclass(frozen=True)
class GatedDeltaConfig:
    num_k_heads: int
    num_v_heads: int
    head_k_dim : int
    head_v_dim : int
    conv_dim   : int
    state_len  : int
    layer_idx  : int

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim


@dataclass
class GatedDeltaForwardInput:
    mixed_qkv  : torch.Tensor
    gate       : torch.Tensor
    beta       : torch.Tensor
    conv_weight: torch.Tensor
    A_log      : torch.Tensor
    dt_bias    : torch.Tensor
    config     : GatedDeltaConfig


@dataclass
class GatedDeltaInput:
    query        : torch.Tensor
    key          : torch.Tensor
    value        : torch.Tensor
    gate         : torch.Tensor
    beta         : torch.Tensor
    A_log        : torch.Tensor
    dt_bias      : torch.Tensor
    initial_state: torch.Tensor | None


def _causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state   : torch.Tensor,
    weight       : torch.Tensor,
) -> torch.Tensor:
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    output = F.conv1d(hidden_states_new, weight, groups=hidden_size)
    return F.silu(output[:, :, -seq_len:]).to(hidden_states.dtype)


class BaseLinearAttentionBackend(ABC):
    def forward(self, inputs: GatedDeltaForwardInput) -> torch.Tensor:
        batch = get_global_ctx().batch
        if batch.is_prefill:
            return self._forward_prefill(inputs)
        elif batch.is_verify:
            return self._forward_verify(inputs)
        else:
            return self._forward_decode(inputs)

    def _make_input(
        self,
        mixed_qkv    : torch.Tensor,
        gate         : torch.Tensor,
        beta         : torch.Tensor,
        inputs       : GatedDeltaForwardInput,
        initial_state: torch.Tensor | None,
    ) -> GatedDeltaInput:
        config = inputs.config
        query, key, value = torch.split(
            mixed_qkv,
            [config.key_dim, config.key_dim, config.value_dim],
            dim=-1,
        )
        seq_len = mixed_qkv.shape[1]
        return GatedDeltaInput(
            query=query.reshape(mixed_qkv.shape[0], seq_len, -1, config.head_k_dim),
            key=key.reshape(mixed_qkv.shape[0], seq_len, -1, config.head_k_dim),
            value=value.reshape(mixed_qkv.shape[0], seq_len, -1, config.head_v_dim),
            gate=gate,
            beta=beta,
            A_log=inputs.A_log,
            dt_bias=inputs.dt_bias,
            initial_state=initial_state,
        )

    def _forward_prefill(self, inputs: GatedDeltaForwardInput) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state
        state_table = ctx.state_table
        page_size = ctx.page_size
        config = inputs.config
        layer_idx = config.layer_idx
        output = torch.empty(
            inputs.mixed_qkv.shape[0], config.num_v_heads, config.head_v_dim,
            dtype=inputs.mixed_qkv.dtype, device=inputs.mixed_qkv.device,
        )

        offset = 0
        for req in batch.reqs:
            seq_len = req.extend_len
            end_offset = offset + seq_len
            use_state = req.cached_len > 0
            cached_len = req.cached_len
            if use_state:
                baseline_slot = int(
                    state_table[req.table_idx, (cached_len - 1) // page_size]
                )

            mixed_qkv = inputs.mixed_qkv[offset:end_offset].transpose(0, 1).unsqueeze(0)
            conv_input = mixed_qkv
            if use_state:
                conv_input = torch.cat(
                    [pool.conv_state[baseline_slot, layer_idx].unsqueeze(0), conv_input],
                    dim=-1,
                )
            total_len = conv_input.shape[-1]
            conv_output = F.silu(
                F.conv1d(
                    conv_input,
                    inputs.conv_weight,
                    padding=config.state_len,
                    groups=config.conv_dim,
                )
            )
            mixed_qkv = conv_output[:, :, :total_len][:, :, -seq_len:].transpose(1, 2)
            gate = inputs.gate[offset:end_offset].unsqueeze(0)
            beta = inputs.beta[offset:end_offset].unsqueeze(0)
            initial_state = (
                pool.recurrent_state[baseline_slot, layer_idx].unsqueeze(0)
                if use_state else None
            )

            segments = []
            start = 0
            history_prefix = config.state_len if use_state else 0
            while start < seq_len:
                global_start = cached_len + start
                page_end = (global_start // page_size + 1) * page_size
                end = min(seq_len, start + page_end - global_start)
                segment_input = self._make_input(
                    mixed_qkv[:, start:end],
                    gate[:, start:end],
                    beta[:, start:end],
                    inputs,
                    initial_state,
                )
                segment_output, initial_state = self.prefill(segment_input)
                segments.append(segment_output)

                page = (cached_len + end - 1) // page_size
                slot = int(state_table[req.table_idx, page])
                history = conv_input[0, :, : history_prefix + end]
                if history.shape[-1] < config.state_len:
                    history = F.pad(history, (config.state_len - history.shape[-1], 0))
                pool.conv_state[slot, layer_idx].copy_(history[:, -config.state_len:])
                pool.recurrent_state[slot, layer_idx].copy_(initial_state[0])
                start = end

            output[offset:end_offset] = torch.cat(segments, dim=1).squeeze(0)
            offset = end_offset

        return output

    def _forward_verify(self, inputs: GatedDeltaForwardInput) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state
        state_table = ctx.state_table
        reserve_offset = ctx.draft_offset
        config = inputs.config
        layer_idx = config.layer_idx
        assert inputs.mixed_qkv.is_cuda
        assert state_table is not None and reserve_offset is not None

        groups = batch.linear_verify_metadata
        if groups is None:
            entries_by_len: dict[int, list[tuple[Request, int]]] = {}
            offset = 0
            for req in batch.reqs:
                entries_by_len.setdefault(req.extend_len, []).append((req, offset))
                offset += req.extend_len
            assert offset == inputs.mixed_qkv.shape[0]

            groups = {}
            for seq_len, entries in entries_by_len.items():
                row_indices = torch.tensor(
                    [row for _req, start in entries for row in range(start, start + seq_len)],
                    dtype=torch.int64,
                    device=inputs.mixed_qkv.device,
                )
                table_indices = torch.tensor(
                    [req.table_idx for req, _start in entries],
                    dtype=torch.int64,
                    device=inputs.mixed_qkv.device,
                )
                baseline_slots = torch.tensor(
                    [req.baseline_slot for req, _start in entries],
                    dtype=torch.int64,
                    device=inputs.mixed_qkv.device,
                )
                write_slots = state_table[
                    table_indices, reserve_offset : reserve_offset + seq_len
                ]
                groups[seq_len] = row_indices, baseline_slots, write_slots
            batch.linear_verify_metadata = groups

        output = torch.empty(
            inputs.mixed_qkv.shape[0], config.num_v_heads, config.head_v_dim,
            dtype=inputs.mixed_qkv.dtype, device=inputs.mixed_qkv.device,
        )
        for seq_len, (row_indices, baseline_slots, write_slots) in groups.items():
            batch_size = len(baseline_slots)
            mixed_qkv = inputs.mixed_qkv.index_select(0, row_indices)
            mixed_qkv = mixed_qkv.view(batch_size, seq_len, config.conv_dim).transpose(1, 2)
            conv_state = pool.conv_state[baseline_slots, layer_idx]
            conv_input = torch.cat([conv_state, mixed_qkv], dim=-1)
            mixed_qkv = F.silu(
                F.conv1d(conv_input, inputs.conv_weight, groups=config.conv_dim)
            ).transpose(1, 2)

            conv_snapshots = (
                conv_input.unfold(-1, config.state_len, 1)[:, :, 1:]
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            pool.conv_state[:, layer_idx].index_copy_(
                0,
                write_slots.flatten().to(torch.int64),
                conv_snapshots.flatten(0, 1),
            )

            group_input = self._make_input(
                mixed_qkv,
                inputs.gate.index_select(0, row_indices).view(
                    batch_size, seq_len, config.num_v_heads
                ),
                inputs.beta.index_select(0, row_indices).view(
                    batch_size, seq_len, config.num_v_heads
                ),
                inputs,
                pool.recurrent_state[baseline_slots, layer_idx],
            )
            group_output = self.verify(
                group_input,
                write_slots,
                pool.recurrent_state[:, layer_idx],
            )
            output.index_copy_(0, row_indices, group_output.flatten(0, 1))

        return output

    def _forward_decode(self, inputs: GatedDeltaForwardInput) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state
        state_table = ctx.state_table
        page_size = ctx.page_size
        config = inputs.config
        layer_idx = config.layer_idx
        table_indices = torch.tensor(
            [req.table_idx for req in batch.reqs], device=inputs.mixed_qkv.device
        )
        read_pages = torch.tensor(
            [(req.cached_len - 1) // page_size for req in batch.reqs],
            device=inputs.mixed_qkv.device,
        )
        write_pages = torch.tensor(
            [req.cached_len // page_size for req in batch.reqs],
            device=inputs.mixed_qkv.device,
        )
        read_slots = state_table[table_indices, read_pages]
        write_slots = state_table[table_indices, write_pages]
        conv_state = pool.conv_state[read_slots, layer_idx]
        recurrent_state = pool.recurrent_state[read_slots, layer_idx]

        mixed_qkv = inputs.mixed_qkv.unsqueeze(1).transpose(1, 2)
        mixed_qkv = _causal_conv1d_update(mixed_qkv, conv_state, inputs.conv_weight)
        write_slots_list = write_slots.tolist()
        for batch_idx, slot in enumerate(write_slots_list):
            pool.conv_state[slot, layer_idx].copy_(conv_state[batch_idx])

        recurrent_input = self._make_input(
            mixed_qkv.transpose(1, 2),
            inputs.gate.unsqueeze(1),
            inputs.beta.unsqueeze(1),
            inputs,
            recurrent_state,
        )
        output, final_state = self.decode(recurrent_input)
        for batch_idx, slot in enumerate(write_slots_list):
            pool.recurrent_state[slot, layer_idx].copy_(final_state[batch_idx])
        return output.squeeze(1)

    @abstractmethod
    def prefill(
        self,
        inputs: GatedDeltaInput,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def decode(
        self,
        inputs: GatedDeltaInput,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def verify(
        self,
        inputs     : GatedDeltaInput,
        write_slots: torch.Tensor,
        state_pool : torch.Tensor,
    ) -> torch.Tensor: ...


__all__ = [
    "BaseLinearAttentionBackend",
    "GatedDeltaConfig",
    "GatedDeltaForwardInput",
    "GatedDeltaInput",
]
