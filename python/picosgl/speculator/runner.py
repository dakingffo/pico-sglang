from __future__ import annotations

from collections.abc import Callable

import torch

from picosgl.message import (
    DraftHandshakeAckMsg,
    DraftHandshakeMsg,
    DraftInitMsg,
    DraftRemoveMsg,
    DraftReply,
    DraftReplyMsg,
    DraftStepMsg,
)
from picosgl.message.queue import ZmqPullQueue, ZmqPushQueue
from picosgl.utils import init_logger

from .base import EngineBase
from .data_plane import DataPlane, DataPlaneSizes
from .drafters.mtp import MTPState

logger = init_logger(__name__)


class SpeculatorRunner:
    def __init__(
        self,
        engine_factory  : Callable[[], EngineBase],
        data_plane      : DataPlane,
        recv            : ZmqPullQueue,
        reply           : ZmqPushQueue,
        on_engine_ready : Callable[[EngineBase], None] | None = None,
        data_plane_sizes: DataPlaneSizes | None                = None,
    ) -> None:
        self.engine_factory = engine_factory
        self.engine: EngineBase | None = None
        self.data_plane = data_plane
        self.recv = recv
        self.reply = reply
        self.on_engine_ready = on_engine_ready
        self.data_plane_sizes = data_plane_sizes
        self.states: dict[int, MTPState] = {}
        self.window_size = 0
        self.hidden_size = 0
        self.vocab_size = 0

    def run_forever(self) -> None:
        self.engine = self.engine_factory()
        if self.data_plane_sizes is not None:
            self.data_plane.prepare_rank1(self.data_plane_sizes)
        if self.on_engine_ready is not None:
            self.on_engine_ready()

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
        if self.data_plane_sizes is not None:
            assert sizes == self.data_plane_sizes
        self.data_plane.init_rank1(handshake.connection_id, sizes)
        self.reply.put(DraftHandshakeAckMsg())
        logger.info(
            "Speculator ready: window_size=%d hidden_size=%d vocab_size=%d device=%s",
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
                    raise NotImplementedError(f"unknown speculator message: {msg!r}")
        except KeyboardInterrupt:
            pass

    def _on_init(self, msg: DraftInitMsg) -> None:
        hidden = self.data_plane.recv_hidden(len(msg.carry_positions))
        self.states[msg.uid] = MTPState(
            table_idx=msg.table_idx,
            sampling_params=msg.sampling_params,
            window_positions=msg.carry_positions,
            window_tokens=msg.carry_tokens,
            window_hidden=hidden,
            window_size=self.window_size,
        )

    def _on_step(self, msg: DraftStepMsg) -> None:
        assert self.engine is not None
        total_rows = sum(len(r.append_positions) for r in msg.reqs)
        if total_rows > 0:
            hidden = self.data_plane.recv_hidden(total_rows)
        off = 0
        for r in msg.reqs:
            st = self.states[r.uid]
            st.n_drafts = r.n_drafts
            if n := len(r.append_positions):
                st.update_window(r.append_positions, r.append_tokens, hidden[off : off + n])
                off += n

        self.engine.draft([self.states[r.uid] for r in msg.reqs])

        sampling_reqs = [r for r in msg.reqs if r.sampling]
        if sampling_reqs:
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
