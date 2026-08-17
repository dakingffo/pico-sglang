from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from picosgl.kernel.pynccl import create_nccl_uid_bytes
from picosgl.core import SamplingParams
from picosgl.message import (
    BaseDrafterMsg,
    DraftHandshakeAckMsg,
    DraftHandshakeMsg,
    DraftInitMsg,
    DraftRemoveMsg,
    DraftReplyMsg,
    DraftStepMsg,
)
from picosgl.utils import ZmqPullQueue, ZmqPushQueue

from .data_plane import DataPlane, DataPlaneSizes, NCCLDataPlane

if TYPE_CHECKING:
    from picosgl.scheduler.config import SchedulerConfig


class DrafterClient:
    """Target/rank0 side of the split-process drafter.

    Control plane is zmq (PUSH drafts / PULL replies); the heavy tensors (carry/appended
    hidden, draft_probs) cross the ``data_plane``. ``__init__`` runs the handshake: ships
    the NCCL uid + agreed buffer sizes, blocks until the drafter's communicator is up, then
    drains the ack. Every ``step`` blocks until the drafter has drafted (single buffer, no
    overrun); ``VerifyManager`` calls it from ``schedule_next_batch``.
    """

    def __init__(
        self,
        config      : SchedulerConfig,
        device      : torch.device,
        vocab_size  : int,
        hidden_size : int,
        *,
        data_plane  : DataPlane | None = None,
        window_size : int | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_spec_tokens = config.speculative_num_draft_tokens
        self.window_size = window_size if window_size is not None else config.speculator_window_size
        self.data_plane = (
            data_plane if data_plane is not None
            else NCCLDataPlane(device, rank=0, dtype=config.dtype)
        )
        self.sender = ZmqPushQueue(
            config.zmq_drafter_addr, create=True, encoder=BaseDrafterMsg.encoder
        )
        self.receiver = ZmqPullQueue(
            config.zmq_drafter_reply_addr, create=True, decoder=DraftReplyMsg.decoder
        )
        self._handshake()

    def _handshake(self) -> None:
        K = self.num_spec_tokens
        max_hidden_rows = max(self.window_size, self.config.max_running_req * (K + 1))
        max_prob_rows = self.config.max_running_req * K
        sizes = DataPlaneSizes(max_hidden_rows, self.hidden_size, max_prob_rows, self.vocab_size)
        uid = create_nccl_uid_bytes()
        # NB ordering: the uid must be enqueued (zmq is async) BEFORE the blocking NCCL
        # init, because the drafter needs the uid to call ITS init, and ncclCommInitRank
        # blocks until both ranks arrive. The ack is drained only after our own init.
        self.sender.put(
            DraftHandshakeMsg(
                nccl_uid=uid,
                max_hidden_rows=max_hidden_rows,
                hidden_size=self.hidden_size,
                max_prob_rows=max_prob_rows,
                vocab_size=self.vocab_size,
                window_size=self.window_size,
            )
        )
        self.data_plane.init_rank0(uid, sizes)
        ack = self.receiver.get()
        assert isinstance(ack, DraftHandshakeAckMsg), (
            f"expected DraftHandshakeAckMsg, got {ack!r}"
        )

    def init(
        self,
        uid             : int,
        table_idx       : int,
        carry_positions : list[int],
        carry_tokens    : list[int],
        hidden          : torch.Tensor,
        sampling_params : SamplingParams,
    ) -> None:
        """Seed the drafter's per-request state with a prefill terminal window."""
        self.sender.put(
            DraftInitMsg(
                uid=uid,
                table_idx=table_idx,
                carry_positions=list(carry_positions),
                carry_tokens=list(carry_tokens),
                sampling_params=sampling_params,
            )
        )
        self.data_plane.send_hidden(hidden)

    def step(
        self,
        reqs           : list,
        appended_hidden: torch.Tensor | None,
        has_sampling   : bool,
    ) -> tuple[DraftReplyMsg, torch.Tensor | None]:
        """Blocking draft round. Returns the reply and the drafter's draft_probs rows
        (None if no request sampled this round)."""
        self.sender.put(DraftStepMsg(reqs=reqs))
        if appended_hidden is not None and appended_hidden.shape[0] > 0:
            self.data_plane.send_hidden(appended_hidden)
        probs: torch.Tensor | None = None
        if has_sampling:
            rows = sum(r.n_drafts for r in reqs if r.sampling)
            probs = self.data_plane.recv_probs(rows, self.vocab_size)
        reply = self.receiver.get()
        return reply, probs

    def remove(self, uid: int) -> None:
        """Drop the drafter's per-request state (finish / abort)."""
        self.sender.put(DraftRemoveMsg(uid=uid))

    def destroy(self) -> None:
        self.data_plane.destroy()
        self.sender.stop()
        self.receiver.stop()


class DraftBroadcastReceiver:
    """Non-primary-rank stand-in for ``DrafterClient``.

    Only rank0 has the zmq/NCCL client; other TP ranks receive the draft results rank0
    broadcasts (``scheduler.io``) and present the same ``step`` interface so
    ``VerifyManager`` is rank-agnostic. ``init``/``remove`` are no-ops — no drafter state
    lives on non-primary ranks.
    """

    def __init__(self, scheduler_io, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self._io = scheduler_io

    def step(
        self,
        reqs           : list,
        appended_hidden: torch.Tensor | None,
        has_sampling   : bool,
    ) -> tuple[DraftReplyMsg, torch.Tensor | None]:
        return self._io.recv_draft_from_rank0(self.vocab_size)

    def init(self, *args, **kwargs) -> None:
        pass

    def remove(self, uid: int) -> None:
        pass

    def destroy(self) -> None:
        pass
