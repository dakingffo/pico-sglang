from __future__ import annotations

import torch
import torch.nn.functional as F

from picosgl.layers import BaseOP
from picosgl.utils import nvtx_annotate


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
        # fp32 in the checkpoint (mamba_ssm_dtype=float32)
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
