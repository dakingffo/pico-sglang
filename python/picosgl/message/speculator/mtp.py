from __future__ import annotations

from dataclasses import dataclass

from picosgl.core import SamplingParams

from ..utils import deserialize_type
from .base import BaseDrafterMsg


class BaseMTPMsg(BaseDrafterMsg):
    @staticmethod
    def decoder(json: dict) -> BaseMTPMsg:
        return deserialize_type(globals(), json)


@dataclass
class DraftHandshakeMsg(BaseMTPMsg):
    connection_id  : bytes
    max_hidden_rows: int
    hidden_size    : int
    max_prob_rows  : int
    vocab_size     : int
    window_size    : int


@dataclass
class DraftHandshakeAckMsg(BaseMTPMsg):
    pass


@dataclass
class DraftInitMsg(BaseMTPMsg):
    uid            : int
    table_idx      : int
    carry_positions: list[int]
    carry_tokens   : list[int]
    sampling_params: SamplingParams


@dataclass
class DraftStepReq(BaseMTPMsg):
    uid             : int
    n_drafts        : int
    append_positions: list[int]
    append_tokens   : list[int]
    sampling        : bool


@dataclass
class DraftStepMsg(BaseMTPMsg):
    reqs: list[DraftStepReq]


@dataclass
class DraftReply(BaseMTPMsg):
    uid         : int
    draft_tokens: list[int]


@dataclass
class DraftReplyMsg(BaseMTPMsg):
    reqs: list[DraftReply]


@dataclass
class DraftRemoveMsg(BaseMTPMsg):
    uid: int
