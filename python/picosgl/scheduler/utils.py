from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from picosgl.core import SamplingParams

    from .prefill import ChunkedRequest


@dataclass
class PendingRequest:
    uid            : int
    input_ids      : torch.Tensor
    sampling_params: SamplingParams
    chunked_req    : ChunkedRequest | None = None

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


@dataclass
class ScheduleResult:
    reqs          : list[PendingRequest]
    output_indices: list[torch.Tensor]
