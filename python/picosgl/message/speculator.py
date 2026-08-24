from __future__ import annotations

from dataclasses import dataclass

from picosgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


class BaseDrafterMsg:
    """Control-plane message between the target (rank0) and the speculator process.

    Payloads are scalars / int lists / SamplingParams only — the heavy tensors
    (carry_hidden, appended hidden, draft_probs) cross the NCCL data plane, keyed by the
    counts carried here. Same encode/decode scheme as BaseBackendMsg; each message module
    uses its own ``globals()`` so ``deserialize_type`` can resolve nested dataclasses.
    """

    def encoder(self) -> dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: dict) -> BaseDrafterMsg:
        return deserialize_type(globals(), json)


@dataclass
class DraftHandshakeMsg(BaseDrafterMsg):
    """rank0 → drafter: the NCCL unique id (128 raw bytes) + agreed buffer sizing.

    The data plane's fixed NCCL buffers (hidden: max_hidden_rows x hidden_size bf16,
    probs: max_prob_rows x vocab_size fp32) and the carry window size must be identical on
    both sides or the all_gather pairing desyncs; rank0 computes them from its config and
    ships them here so the drafter never derives them independently. ``window_size`` is
    the rolling carry-window row count the drafter trims to in MTPState.update_carry.
    """
    nccl_uid        : bytes
    max_hidden_rows : int
    hidden_size     : int
    max_prob_rows   : int
    vocab_size      : int
    window_size     : int


@dataclass
class DraftHandshakeAckMsg(BaseDrafterMsg):
    """drafter → rank0: data-plane communicator initialized, ready for DraftInitMsg."""
    pass


@dataclass
class DraftInitMsg(BaseDrafterMsg):
    """rank0 → drafter: a request's prefill terminal window (last W accepted rows).

    carry_positions / carry_tokens ride the control plane; the window's carry_hidden
    arrives over NCCL immediately after this message. drafter allocates MTPState keyed by
    ``uid``. ``sampling_params`` is per-request (drafter samples with the request's own
    temperature / top-k/p, greedy or not).
    """
    uid             : int
    table_idx       : int
    carry_positions : list[int]
    carry_tokens    : list[int]
    sampling_params : SamplingParams


@dataclass
class DraftStepReq(BaseDrafterMsg):
    """One request's draft request within a step batch.

    append_positions / append_tokens are the rows the target committed since the last
    step (empty on a request's first verify round); the matching hidden rows arrive over
    NCCL in the same order as the reqs in DraftStepMsg. ``sampling`` marks a non-greedy
    request: both sides use it to decide the draft_probs leg and its row layout.
    """
    uid              : int
    n_drafts         : int
    append_positions : list[int]
    append_tokens    : list[int]
    sampling         : bool


@dataclass
class DraftStepMsg(BaseDrafterMsg):
    reqs: list[DraftStepReq]


@dataclass
class DraftReply(BaseDrafterMsg):
    uid          : int
    draft_tokens : list[int]  # exactly n_drafts tokens (draft_probs rode NCCL)


@dataclass
class DraftReplyMsg(BaseDrafterMsg):
    reqs: list[DraftReply]


@dataclass
class DraftRemoveMsg(BaseDrafterMsg):
    """rank0 → drafter: drop the request's MTPState (finish / abort)."""
    uid: int
