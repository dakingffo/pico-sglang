from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import torch

from picosgl.kernel.pynccl import init_pynccl_p2p


class DataPlaneSizes(NamedTuple):
    """Fixed NCCL buffer sizes for the 2-rank data plane, agreed at handshake time.

    Both sides allocate identical buffers (hidden: max_hidden_rows x hidden_size in the
    model dtype; probs: max_prob_rows x vocab_size fp32) so the all_gather pairing never
    desyncs. Actual per-message row counts ride the control plane.
    """

    max_hidden_rows: int
    hidden_size    : int
    max_prob_rows  : int
    vocab_size     : int


class DataPlane(ABC):
    """Two-leg data plane between target rank0 and the drafter process.

    All ops are blocking and must be called in the same order on both sides per round:
    one ``send_hidden``/``recv_hidden`` pair (target → drafter), then one
    ``send_probs``/``recv_probs`` pair (drafter → target) iff any request sampled. The
    target blocks inside ``client.step`` until the drafter has run its draft and replied
    (single buffer, no overrun).
    """

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
    def recv_hidden(self, num_rows: int, hidden_size: int) -> torch.Tensor:
        """Drafter side of the same collective; returns the target's rows on its device."""

    @abstractmethod
    def send_probs(self, probs: torch.Tensor) -> None:
        """Drafter → target: the draft_probs rows for this step's sampling requests."""

    @abstractmethod
    def recv_probs(self, num_rows: int, vocab_size: int) -> torch.Tensor:
        """Target side of the same collective; returns the drafter's rows on its device."""

    def destroy(self) -> None:
        pass


class NCCLDataPlane(DataPlane):
    """Production data plane: a standalone 2-rank NCCL communicator.

    Implements each leg with the all_gather-as-P2P trick on the 2-rank comm, whose output
    is [rank0_input | rank1_input]. Leg A (hidden target→drafter): rank0 fills its half
    with hidden, drafter contributes zeros, both read out[0:S). Leg B (probs drafter→
    target): drafter fills its half, rank0 zeros, both read out[S:2S). Both sides issue
    A then B in the same order so the collectives pair; the drafter's draft work happens
    between the two legs on its side, which is what makes rank0's ``recv_probs`` block.
    """

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
        self._nccld = init_pynccl_p2p(rank=0, world_size=self.world_size, nccl_uid=nccl_uid)
        self._alloc_bufs(sizes)

    def init_rank1(self, nccl_uid: bytes, sizes: DataPlaneSizes) -> None:
        self._nccld = init_pynccl_p2p(rank=1, world_size=self.world_size, nccl_uid=nccl_uid)
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

    def recv_hidden(self, num_rows: int, hidden_size: int) -> torch.Tensor:
        self._nccld.all_gather(self._hidden_out, self._hidden_in)
        return self._hidden_out[:num_rows].clone()

    def send_probs(self, probs: torch.Tensor) -> None:
        rows = probs.shape[0]
        assert self._probs_in is not None and rows <= self._probs_in.shape[0]
        self._probs_in[:rows].copy_(probs)
        self._nccld.all_gather(self._probs_out, self._probs_in)

    def recv_probs(self, num_rows: int, vocab_size: int) -> torch.Tensor:
        S = self._probs_out.shape[0] // 2
        self._nccld.all_gather(self._probs_out, self._probs_in)
        return self._probs_out[S : S + num_rows].clone()

    def destroy(self) -> None:
        if self._nccld is not None:
            torch.cuda.synchronize(self.device)


class PipeDataPlane(DataPlane):
    """Test-injected data plane: tensors over ``multiprocessing.Queue`` (CPU round-trip).

    Local split-process tests cannot bootstrap a 2-rank NCCL communicator (WSL2 / single
    GPU), so they inject a pair of these instead; production always uses ``NCCLDataPlane``.
    The two ends are created with ``make_pair`` before any ``mp.Process`` is started.
    """

    def __init__(
        self,
        device     : torch.device,
        rank       : int,
        to_drafter : "multiprocessing.Queue",
        to_target  : "multiprocessing.Queue",
        dtype      : torch.dtype = torch.bfloat16,
    ) -> None:
        self.device = device
        self.rank = rank
        self.dtype = dtype  # the model hidden dtype (bf16); the probs leg is always fp32
        self._to_drafter = to_drafter
        self._to_target = to_target

    @staticmethod
    def make_pair(device: torch.device) -> tuple["PipeDataPlane", "PipeDataPlane"]:
        import multiprocessing as mp

        to_drafter: "multiprocessing.Queue" = mp.Queue()
        to_target : "multiprocessing.Queue" = mp.Queue()
        return (
            PipeDataPlane(device, 0, to_drafter, to_target),
            PipeDataPlane(device, 1, to_drafter, to_target),
        )

    def init_rank0(self, nccl_uid: bytes, sizes: DataPlaneSizes) -> None:
        pass

    def init_rank1(self, nccl_uid: bytes, sizes: DataPlaneSizes) -> None:
        pass

    def send_hidden(self, hidden: torch.Tensor) -> None:
        # numpy has no bf16; the widening fp32 round-trip is bit-exact for an already-bf16
        # value (recv_hidden casts back to self.dtype), mirroring the NCCL plane's bf16 pipe.
        self._to_drafter.put(hidden.detach().cpu().to(torch.float32).numpy())

    def recv_hidden(self, num_rows: int, hidden_size: int) -> torch.Tensor:
        arr = self._to_drafter.get()
        return torch.from_numpy(arr).to(self.device, dtype=self.dtype)

    def send_probs(self, probs: torch.Tensor) -> None:
        self._to_target.put(probs.detach().cpu().numpy())

    def recv_probs(self, num_rows: int, vocab_size: int) -> torch.Tensor:
        arr = self._to_target.get()
        return torch.from_numpy(arr).to(self.device)
