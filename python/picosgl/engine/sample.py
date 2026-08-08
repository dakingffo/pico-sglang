from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from picosgl.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from picosgl.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k       : torch.Tensor | None = None
    top_p       : torch.Tensor | None = None

    @property
    def is_greedy(self) -> bool:
        return self.temperatures is None


def make_device_tensor(data: list, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits      : torch.Tensor,
    temperatures: torch.Tensor,
    top_k       : torch.Tensor | int | None,
    top_p       : torch.Tensor | float | None,
) -> torch.Tensor:
    import flashinfer.sampling as sampling
    
    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)
    elif top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)
    elif top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)
    else:
        assert top_k is not None and top_p is not None
        return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)

        return BatchSamplingArgs(
            temperatures,
            top_k=(
                make_device_tensor(top_ks, torch.int32, self.device) 
                if any(k != self.vocab_size for k in top_ks) else None
            ),
            top_p=(
                make_device_tensor(top_ps, torch.float32, self.device) 
                if any(p < 1.0 for p in top_ps) else None
            )
        )

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        if args.is_greedy:  # greedy sampling
            return torch.argmax(logits, dim=-1)
        else:
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
