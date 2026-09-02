import torch
import torch.nn.functional as F

from picosgl.utils import nvtx_annotate

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(
        self,
        size         : int,
        eps          : float,
        zero_centered: bool = False,
    ) -> None:
        self.eps = eps
        self.zero_centered = zero_centered
        self.weight = torch.zeros(size) if zero_centered else torch.empty(size)
        if zero_centered:
            from flashinfer import gemma_rmsnorm

            self.rmsnorm = gemma_rmsnorm
        else:
            from flashinfer import rmsnorm

            self.rmsnorm = rmsnorm

    @nvtx_annotate("RMSNorm")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.zero_centered and not x.is_cuda:
            x_f = x.float()
            output = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
            output = output * (1.0 + self.weight.float())
            return output.type_as(x)
        if self.zero_centered and x.ndim > 2:
            shape = x.shape
            return self.rmsnorm(
                x.reshape(-1, shape[-1]), self.weight, self.eps
            ).reshape(shape)
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        if self.zero_centered and not x.is_cuda:
            x.copy_(self.forward(x))
        elif self.zero_centered and x.ndim > 2:
            x_flat = x.reshape(-1, x.shape[-1])
            self.rmsnorm(x_flat, self.weight, self.eps, out=x_flat)
        else:
            self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormGated(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.ones(size, dtype=torch.float32)

    @nvtx_annotate("RMSNormGated")
    def forward(
        self,
        hidden_states: torch.Tensor,
        gate         : torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states * self.weight.float()
        hidden_states = hidden_states * F.silu(gate.float())
        return hidden_states.to(input_dtype)


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self,
        x       : torch.Tensor,
        residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        else:
            self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
            return x, residual
