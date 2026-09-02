from __future__ import annotations

import torch
import torch.nn.functional as F

from picosgl.kernel import recurrent_gated_delta_triton

from .base import BaseLinearAttentionBackend, GatedDeltaInput
from .reference import _chunk_gated_delta_rule, _l2norm, _recurrent_gated_delta_rule


def _expand_gva(
    query: torch.Tensor,
    key  : torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    repeats = value.shape[2] // query.shape[2]
    if repeats > 1:
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)
    return query, key


def _prepare(inputs: GatedDeltaInput):
    query, key = _expand_gva(inputs.query, inputs.key, inputs.value)
    beta = inputs.beta.sigmoid()
    gate = -inputs.A_log.float().exp() * F.softplus(
        inputs.gate.float() + inputs.dt_bias
    )
    return query, key, inputs.value, gate, beta


class NativeLinearAttentionBackend(BaseLinearAttentionBackend):
    def prefill(
        self,
        inputs: GatedDeltaInput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query, key, value, gate, beta = _prepare(inputs)
        output, final_state = _chunk_gated_delta_rule(
            query,
            key,
            value,
            g=gate,
            beta=beta,
            initial_state=inputs.initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        assert final_state is not None
        return output, final_state

    def decode(
        self,
        inputs: GatedDeltaInput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query, key, value, gate, beta = _prepare(inputs)
        output, final_state = _recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=gate,
            beta=beta,
            initial_state=inputs.initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        assert final_state is not None
        return output, final_state

    def verify(
        self,
        inputs     : GatedDeltaInput,
        write_slots: torch.Tensor,
        state_pool : torch.Tensor,
    ) -> torch.Tensor:
        query, key, value, gate, beta = _prepare(inputs)
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
        assert inputs.initial_state is not None
        return recurrent_gated_delta_triton(
            query,
            key,
            value,
            gate,
            beta,
            inputs.initial_state,
            write_slots,
            state_pool,
        )


__all__ = ["NativeLinearAttentionBackend"]
