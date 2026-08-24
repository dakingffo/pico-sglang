from __future__ import annotations

from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import torch

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
from picosgl.message.queue import ZmqPullQueue, ZmqPushQueue

from .data_plane import DataPlane, make_data_plane_sizes
from .drafters.mtp import (
    MTPHiddenFeature,
    MTPSpeculatorConfig,
)
from .base import SpeculatorHiddenBase

if TYPE_CHECKING:
    from picosgl.scheduler.config import SchedulerConfig
    from picosgl.scheduler.io import SchedulerIOMixin


class SpeculatorClientBase(ABC):
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
        speculator_config = config.speculator_config
        assert isinstance(speculator_config, MTPSpeculatorConfig)
        self.num_spec_tokens = speculator_config.num_draft_tokens
        self.window_size = (
            window_size if window_size is not None else speculator_config.window_size
        )

    @staticmethod
    def _unwrap_hidden(hidden_feature: SpeculatorHiddenBase) -> torch.Tensor:
        assert isinstance(hidden_feature, MTPHiddenFeature)
        return hidden_feature.full_hidden

    def _prepare_init(
        self,
        end_position   : int,
        token_ids      : torch.Tensor,
        hidden_feature : SpeculatorHiddenBase,
    ) -> tuple[list[int], list[int], torch.Tensor]:
        full_hidden = self._unwrap_hidden(hidden_feature)
        window_len = min(self.window_size, full_hidden.shape[0])
        positions = list(range(end_position + 1 - window_len, end_position + 1))
        tokens = token_ids[positions].tolist()
        return positions, tokens, full_hidden[-window_len:].contiguous()

    @abstractmethod
    def init(
        self,
        uid            : int,
        table_idx      : int,
        end_position   : int,
        token_ids      : torch.Tensor,
        hidden_feature : SpeculatorHiddenBase,
        sampling_params: SamplingParams,
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


class MainSpeculatorClient(SpeculatorClientBase):
    def __init__(
        self,
        config      : SchedulerConfig,
        device      : torch.device,
        vocab_size  : int,
        hidden_size : int,
        window_size : int | None              = None,
        scheduler_io: SchedulerIOMixin | None = None,
        data_plane   : DataPlane | None        = None,
    ):
        super().__init__(config, device, vocab_size, hidden_size, window_size)
        self._io = scheduler_io
        assert data_plane is not None
        self.data_plane = data_plane
        self.sender = ZmqPushQueue(
            config.zmq_drafter_addr, create=True, encoder=BaseDrafterMsg.encoder
        )
        self.receiver = ZmqPullQueue(
            config.zmq_drafter_reply_addr, create=True, decoder=DraftReplyMsg.decoder
        )

        sizes = make_data_plane_sizes(
            self.config, self.hidden_size, self.vocab_size, self.window_size
        )
        uid = self.data_plane.make_connection_id()
        # Connection metadata must go out before transport initialization, which blocks
        # until the worker consumes the same handshake.
        self.sender.put(
            DraftHandshakeMsg(
                connection_id=uid,
                max_hidden_rows=sizes.max_hidden_rows,
                hidden_size=self.hidden_size,
                max_prob_rows=sizes.max_prob_rows,
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
        uid            : int,
        table_idx      : int,
        end_position   : int,
        token_ids      : torch.Tensor,
        hidden_feature : SpeculatorHiddenBase,
        sampling_params: SamplingParams,
    ) -> None:
        carry_positions, carry_tokens, hidden = self._prepare_init(
            end_position, token_ids, hidden_feature
        )
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
        if appended_hidden is not None:
            self.data_plane.send_hidden(appended_hidden)
        probs: torch.Tensor | None = None
        if has_sampling:
            rows = sum(r.n_drafts for r in reqs if r.sampling)
            probs = self.data_plane.recv_probs(rows)
        reply = self.receiver.get()
        if self._io is not None and self.config.tp_info.size > 1:
            self._io._send_draft_to_ranks(reply, probs)
        return reply, probs

    def remove(self, uid: int) -> None:
        self.sender.put(DraftRemoveMsg(uid=uid))

    def destroy(self) -> None:
        self.data_plane.destroy()
        self.sender.stop()
        self.receiver.stop()


class BroadcastSpeculatorClient(SpeculatorClientBase):
    """Non-primary-rank stand-in for ``SpeculatorClientBase``.

    Only rank0 has the ZMQ/data-plane client; other TP ranks receive the draft results rank0
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
        return self._io._recv_draft_from_rank0(self.vocab_size)

    def init(self, *args, **kwargs) -> None:
        pass

    def remove(self, uid: int) -> None:
        pass

    def destroy(self) -> None:
        pass
