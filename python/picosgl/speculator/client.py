from __future__ import annotations

from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import torch

from picosgl.kernel.pynccl import create_nccl_uid_bytes
from picosgl.core import SamplingParams
from picosgl.message import (
    BaseDrafterMsg,
    DraftHandshakeAckMsg,
    DraftHandshakeMsg,
    DraftInitMsg,
    DraftRemoveMsg,
    DraftReply,
    DraftReplyMsg,
    DraftStepMsg,
)
from picosgl.utils import ZmqPullQueue, ZmqPushQueue

from .data_plane import DataPlaneSizes, NCCLDataPlane
from .drafters.mtp import MTPEngine, MTPState

if TYPE_CHECKING:
    from picosgl.scheduler.config import SchedulerConfig
    from picosgl.scheduler.io import SchedulerIOMixin

class DrafterClientBase(ABC):
    def __init__(
        self,
        config      : SchedulerConfig,
        device      : torch.device,
        vocab_size  : int,
        hidden_size : int,
        window_size : int | None = None,
    ):
        self.config = config
        self.device = device
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_spec_tokens = config.speculative_num_draft_tokens
        self.window_size = window_size if window_size is not None else config.speculator_window_size

    @abstractmethod
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
        ...

    @abstractmethod
    def step(
        self,
        reqs           : list,
        appended_hidden: torch.Tensor | None,
        has_sampling   : bool,
    ) -> tuple[DraftReplyMsg, torch.Tensor | None]:
        """Blocking draft round. Returns the reply and the drafter's draft_probs rows
        (None if no request sampled this round)."""
        ...

    @abstractmethod
    def remove(self, uid: int) -> None:
        """Drop the drafter's per-request state (finish / abort)."""
        ...

    @abstractmethod
    def destroy(self) -> None:
        ...


class LocalDrafterClient(DrafterClientBase):
    def __init__(
        self,
        config      : SchedulerConfig,
        device      : torch.device,
        vocab_size  : int,
        hidden_size : int,
        window_size : int | None       = None,
        engine      : MTPEngine | None = None,
    ):
        super().__init__(config, device, vocab_size, hidden_size, window_size)
        self.direct = True
        self.engine = engine
        self.states: dict[int, MTPState] = {}

    def init(
        self,
        uid             : int,
        table_idx       : int,
        carry_positions : list[int],
        carry_tokens    : list[int],
        hidden          : torch.Tensor,
        sampling_params : SamplingParams,
    ) -> None:
        self.states[uid] = MTPState(
            sampling_params=sampling_params,
            carry_positions=list(carry_positions),
            carry_tokens=list(carry_tokens),
            carry_hidden=hidden,
            window_size=self.window_size,
        )

    def step(
        self,
        reqs           : list,
        appended_hidden: torch.Tensor | None,
        has_sampling   : bool,
    ) -> tuple[DraftReplyMsg, torch.Tensor | None]:
        total_rows = sum(len(r.append_positions) for r in reqs)
        if total_rows > 0:
            off = 0
            for r in reqs:
                if n:= len(r.append_positions):
                    st = self.states[r.uid]
                    st.update_carry(
                        r.append_positions, r.append_tokens, appended_hidden[off : off + n]
                    )
                    off += n
            assert off == total_rows
        for st, r in zip((self.states[r.uid] for r in reqs), reqs):
            st.n_drafts = r.n_drafts

        self.engine.draft([self.states[r.uid] for r in reqs])

        probs: torch.Tensor | None = None
        sampling_reqs = [r for r in reqs if r.sampling]
        if sampling_reqs:
            probs = torch.cat(
                [self.states[r.uid].draft_probs[: r.n_drafts] for r in sampling_reqs],
                dim=0,
            )
        reply = DraftReplyMsg(
            reqs=[
                DraftReply(uid=r.uid, draft_tokens=self.states[r.uid].draft_tokens)
                for r in reqs
            ]
        )
        return reply, probs

    def remove(self, uid: int) -> None:
        self.states.pop(uid, None)

    def destroy(self) -> None:
        return


class RemoteDrafterClient(DrafterClientBase):
    def __init__(
        self,
        config      : SchedulerConfig,
        device      : torch.device,
        vocab_size  : int,
        hidden_size : int,
        window_size : int | None       = None,
    ):
        super().__init__(config, device, vocab_size, hidden_size, window_size)
        self.direct = False
        self.data_plane = NCCLDataPlane(device, rank=0, dtype=config.dtype)
        self.sender = ZmqPushQueue(
            config.zmq_drafter_addr, create=True, encoder=BaseDrafterMsg.encoder
        )
        self.receiver = ZmqPullQueue(
            config.zmq_drafter_reply_addr, create=True, decoder=DraftReplyMsg.decoder
        )
        
        K = self.num_spec_tokens
        max_hidden_rows = max(self.window_size, self.config.max_running_req * (K + 1))
        max_prob_rows = self.config.max_running_req * K
        sizes = DataPlaneSizes(max_hidden_rows, self.hidden_size, max_prob_rows, self.vocab_size)
        uid = create_nccl_uid_bytes()
        # uid must go out before the blocking NCCL init (both ranks block in ncclCommInitRank).
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
        self.sender.put(DraftRemoveMsg(uid=uid))

    def destroy(self) -> None:
        self.data_plane.destroy()
        self.sender.stop()
        self.receiver.stop()


class BroadcastDrafterClient(DrafterClientBase):
    """Non-primary-rank stand-in for ``DrafterClientBase``.

    Only rank0 has the zmq/NCCL client; other TP ranks receive the draft results rank0
    broadcasts (``scheduler.io``) and present the same ``step`` interface so
    ``VerifyManager`` is rank-agnostic. ``init``/``remove`` are no-ops — no drafter state
    lives on non-primary ranks.
    """

    def __init__(
        self,         
        config      : SchedulerConfig,
        device      : torch.device,
        vocab_size  : int,
        hidden_size : int,
        window_size : int | None              = None,
        scheduler_io: SchedulerIOMixin | None = None
    ):
        super().__init__(config, device, vocab_size, hidden_size, window_size)
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
