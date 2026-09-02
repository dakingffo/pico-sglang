from __future__ import annotations

import torch

from .base import GatedDeltaInput
from .native import NativeLinearAttentionBackend


class FlashLinearAttentionBackend(NativeLinearAttentionBackend):
    def __init__(self) -> None:
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )

        self._chunk = chunk_gated_delta_rule
        self._recurrent = fused_recurrent_gated_delta_rule

    def prefill(
        self,
        inputs: GatedDeltaInput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, final_state = self._chunk(
            q=inputs.query,
            k=inputs.key,
            v=inputs.value,
            g=inputs.gate,
            beta=inputs.beta,
            A_log=inputs.A_log,
            dt_bias=inputs.dt_bias,
            initial_state=inputs.initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            state_v_first=False,
        )
        assert final_state is not None
        return output, final_state

    def decode(
        self,
        inputs: GatedDeltaInput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, final_state = self._recurrent(
            q=inputs.query,
            k=inputs.key,
            v=inputs.value,
            g=inputs.gate,
            beta=inputs.beta,
            A_log=inputs.A_log,
            dt_bias=inputs.dt_bias,
            initial_state=inputs.initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            state_v_first=False,
        )
        assert final_state is not None
        return output, final_state


__all__ = ["FlashLinearAttentionBackend"]
