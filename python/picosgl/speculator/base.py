from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch


@dataclass
class DraftState:
    """Polymorphic per-request spec-decode state held by the target-side VerifyManager.

    Only these two fields are visible across the target/drafter boundary; drafter engines
    keep their richer per-request state in subclasses (e.g. ``MTPState``). The VerifyManager
    allocates a ``DraftState`` per table_idx, the drafter fills it in via ``client.step``.
    """

    draft_tokens: list[int]           = field(default_factory=list)
    draft_probs : torch.Tensor | None = None


class EngineBase(ABC):
    """Drafter-side engine: fills ``DraftState.draft_tokens`` / ``draft_probs`` for a batch.

    Lives in the drafter process, drives the standalone drafter model on its own device /
    stream. Drafter runs are blocking on the target side (VerifyManager waits for the reply),
    so ``stream`` is for the engine's own internal overlap, not target-side pipelining.
    """

    device          : torch.device
    stream          : torch.cuda.Stream
    num_spec_tokens : int
    vocab_size      : int

    @abstractmethod
    def draft(self, states: list[DraftState]) -> None:
        """Fill each state's draft_tokens / draft_probs for this round."""

    def destroy(self) -> None:
        """Release engine-side resources (CUDA graph pools etc.) on shutdown."""
        pass
