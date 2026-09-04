from __future__ import annotations

import torch

from .native import NativeLinearAttentionBackend


class FlashLinearAttentionBackend(NativeLinearAttentionBackend):
    def __init__(self) -> None:
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )

        self._chunk = chunk_gated_delta_rule
        self._recurrent = fused_recurrent_gated_delta_rule

    def _prefill(
        self,
        query        : torch.Tensor,
        key          : torch.Tensor,
        value        : torch.Tensor,
        gate         : torch.Tensor,
        beta         : torch.Tensor,
        A_log        : torch.Tensor,
        dt_bias      : torch.Tensor,
        initial_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, final_state = self._chunk(
            q=query,
            k=key,
            v=value,
            g=gate,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            state_v_first=False,
        )
        assert final_state is not None
        return output, final_state

    def _decode(
        self,
        query        : torch.Tensor,
        key          : torch.Tensor,
        value        : torch.Tensor,
        gate         : torch.Tensor,
        beta         : torch.Tensor,
        A_log        : torch.Tensor,
        dt_bias      : torch.Tensor,
        initial_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, final_state = self._recurrent(
            q=query,
            k=key,
            v=value,
            g=gate,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            state_v_first=False,
        )
        assert final_state is not None
        return output, final_state


__all__ = ["FlashLinearAttentionBackend"]
