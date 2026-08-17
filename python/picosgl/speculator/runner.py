from __future__ import annotations

import torch

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
from picosgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger

from .base import EngineBase
from .data_plane import DataPlane, DataPlaneSizes
from .drafters.mtp import MTPState

logger = init_logger(__name__)


class DrafterRunner:
    """Drafter-process event loop: handshake, init, step, remove.

    Mirrors the old in-model ``VerifyManager._draft`` one request at a time through
    ``EngineBase``; the target blocks on each step (single data-plane buffer, no overrun),
    so this loop is strictly sequential. The rolling carry window (positions / tokens /
    hidden) is maintained here in ``MTPState`` — the target ships committed rows over the
    control plane (ids) and data plane (hidden) every step.
    """

    def __init__(
        self,
        engine    : EngineBase,
        data_plane: DataPlane,
        recv      : ZmqPullQueue,
        reply     : ZmqPushQueue,
    ) -> None:
        self.engine = engine
        self.data_plane = data_plane
        self.recv = recv
        self.reply = reply
        self.states: dict[int, MTPState] = {}
        self.window_size = 0
        self.hidden_size = 0
        self.vocab_size = 0

    def run_forever(self) -> None:
        handshake = self.recv.get()
        assert isinstance(handshake, DraftHandshakeMsg), (
            f"expected DraftHandshakeMsg, got {handshake!r}"
        )
        self.window_size = handshake.window_size
        self.hidden_size = handshake.hidden_size
        self.vocab_size = handshake.vocab_size
        sizes = DataPlaneSizes(
            handshake.max_hidden_rows, handshake.hidden_size,
            handshake.max_prob_rows, handshake.vocab_size,
        )
        # NB ordering: ncclCommInitRank blocks until both ranks call it, so we must NOT
        # gate our init on the target's ack — the target drains our ack after ITS init.
        self.data_plane.init_rank1(handshake.nccl_uid, sizes)
        self.reply.put(DraftHandshakeAckMsg())
        logger.info(
            "Drafter ready: window_size=%d hidden_size=%d vocab_size=%d device=%s",
            self.window_size, self.hidden_size, self.vocab_size, self.engine.device,
        )

        try:
            while True:
                msg = self.recv.get()
                if isinstance(msg, DraftInitMsg):
                    self._on_init(msg)
                elif isinstance(msg, DraftStepMsg):
                    self._on_step(msg)
                elif isinstance(msg, DraftRemoveMsg):
                    self.states.pop(msg.uid, None)
                else:
                    raise NotImplementedError(f"unknown drafter message: {msg!r}")
        except KeyboardInterrupt:
            pass

    def _on_init(self, msg: DraftInitMsg) -> None:
        hidden = self.data_plane.recv_hidden(len(msg.carry_positions), self.hidden_size)
        self.states[msg.uid] = MTPState(
            sampling_params=msg.sampling_params,
            carry_positions=msg.carry_positions,
            carry_tokens=msg.carry_tokens,
            carry_hidden=hidden,
            window_size=self.window_size,
        )

    def _on_step(self, msg: DraftStepMsg) -> None:
        # appends: one recv_hidden for the whole step, then update_carry per request
        # (row order matches the reqs' append_positions lengths).
        total_rows = sum(len(r.append_positions) for r in msg.reqs)
        if total_rows > 0:
            hidden = self.data_plane.recv_hidden(total_rows, self.hidden_size)
            off = 0
            for r in msg.reqs:
                n = len(r.append_positions)
                if n:
                    self.states[r.uid].update_carry(
                        r.append_positions, r.append_tokens, hidden[off : off + n]
                    )
                    off += n
            assert off == total_rows

        for st, r in zip((self.states[r.uid] for r in msg.reqs), msg.reqs):
            st.n_drafts = r.n_drafts
        self.engine.draft([self.states[r.uid] for r in msg.reqs])

        sampling_reqs = [r for r in msg.reqs if r.sampling]
        if sampling_reqs:
            rows = sum(r.n_drafts for r in sampling_reqs)
            probs = torch.cat(
                [self.states[r.uid].draft_probs[: r.n_drafts] for r in sampling_reqs],
                dim=0,
            )
            self.data_plane.send_probs(probs)

        self.reply.put(
            DraftReplyMsg(
                reqs=[
                    DraftReply(uid=r.uid, draft_tokens=self.states[r.uid].draft_tokens)
                    for r in msg.reqs
                ]
            )
        )
