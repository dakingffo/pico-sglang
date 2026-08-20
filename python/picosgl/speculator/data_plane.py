from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import torch

from picosgl.kernel import init_pynccl_drafter_target_separation


class DataPlaneSizes(NamedTuple):
    max_hidden_rows: int
    hidden_size    : int
    max_prob_rows  : int
    vocab_size     : int


class DataPlane(ABC):
    device: torch.device
    rank  : int

    @abstractmethod
    def init_rank0(self, nccl_uid: bytes, sizes: DataPlaneSizes) -> None:
        """Target side: initialize the communicator (blocks until rank1 inits)."""

    @abstractmethod
    def init_rank1(self, nccl_uid: bytes, sizes: DataPlaneSizes) -> None:
        """Drafter side: initialize the communicator (blocks until rank0 inits)."""

    @abstractmethod
    def send_hidden(self, hidden: torch.Tensor) -> None:
        """Target → drafter: the appended carry-hidden rows (or a request's init window)."""

    @abstractmethod
    def recv_hidden(self, num_rows: int) -> torch.Tensor:
        """Drafter side of the same collective; returns the target's rows on its device."""

    @abstractmethod
    def send_probs(self, probs: torch.Tensor) -> None:
        """Drafter → target: the draft_probs rows for this step's sampling requests."""

    @abstractmethod
    def recv_probs(self, num_rows: int) -> torch.Tensor:
        """Target side of the same collective; returns the drafter's rows on its device."""

    def destroy(self) -> None:
        pass


class NCCLDataPlane(DataPlane):
    def __init__(
        self,
        device     : torch.device,
        rank       : int,
        world_size : int = 2,
        dtype      : torch.dtype = torch.bfloat16,
    ) -> None:
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.dtype = dtype
        self._nccld = None
        self._hidden_in: torch.Tensor | None = None
        self._hidden_out: torch.Tensor | None = None
        self._probs_in: torch.Tensor | None = None
        self._probs_out: torch.Tensor | None = None

    def init_rank0(self, nccl_uid: bytes, sizes: DataPlaneSizes) -> None:
        self._nccld = init_pynccl_drafter_target_separation(
            rank=0, world_size=self.world_size, nccl_uid=nccl_uid
        )
        self._alloc_bufs(sizes)

    def init_rank1(self, nccl_uid: bytes, sizes: DataPlaneSizes) -> None:
        self._nccld = init_pynccl_drafter_target_separation(
            rank=1, world_size=self.world_size, nccl_uid=nccl_uid
        )
        self._alloc_bufs(sizes)

    def _alloc_bufs(self, sizes: DataPlaneSizes) -> None:
        max_hidden, hidden_size, max_prob, vocab_size = sizes
        self._hidden_in = torch.zeros(
            max_hidden, hidden_size, dtype=self.dtype, device=self.device
        )
        self._hidden_out = torch.zeros(
            2 * max_hidden, hidden_size, dtype=self.dtype, device=self.device
        )
        self._probs_in = torch.zeros(
            max_prob, vocab_size, dtype=torch.float32, device=self.device
        )
        self._probs_out = torch.zeros(
            2 * max_prob, vocab_size, dtype=torch.float32, device=self.device
        )

    def send_hidden(self, hidden: torch.Tensor) -> None:
        rows = hidden.shape[0]
        assert self._hidden_in is not None and rows <= self._hidden_in.shape[0]
        self._hidden_in[:rows].copy_(hidden)  # rest stale; receiver only reads [:rows]
        self._nccld.all_gather(self._hidden_out, self._hidden_in)

    def recv_hidden(self, num_rows: int) -> torch.Tensor:
        self._nccld.all_gather(self._hidden_out, self._hidden_in)
        return self._hidden_out[:num_rows].clone()

    def send_probs(self, probs: torch.Tensor) -> None:
        rows = probs.shape[0]
        assert self._probs_in is not None and rows <= self._probs_in.shape[0]
        self._probs_in[:rows].copy_(probs)
        self._nccld.all_gather(self._probs_out, self._probs_in)

    def recv_probs(self, num_rows: int) -> torch.Tensor:
        S = self._probs_out.shape[0] // 2
        self._nccld.all_gather(self._probs_out, self._probs_in)
        return self._probs_out[S : S + num_rows].clone()

    def destroy(self) -> None:
        if self._nccld is not None:
            torch.cuda.synchronize(self.device)
