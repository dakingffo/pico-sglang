from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from picosgl.engine import Sampler

from ...base import EngineBase
from .state import MTPState

if TYPE_CHECKING:
    from picosgl.models.drafters import Qwen3_5MTPDrafter


class MTPEngine(EngineBase):
    """Eager MTP draft engine: exact port of the old in-model ``VerifyManager._draft``.

    Drafts each state one request at a time through the standalone drafter, producing
    bit-identical ``draft_tokens`` / ``draft_probs`` to the old in-process path. This is the
    correctness baseline for the drafter process; batched / CUDA-graph paths (``MTPGraph``)
    are layered on later without changing this contract. Sampling uses the request's own
    ``SamplingParams`` carried over the control plane (``is_greedy`` / temperature / top-k/p),
    drawn with the same ``Sampler.draft_token`` / ``_target_dist`` as the old path.
    """

    def __init__(
        self,
        drafter          : Qwen3_5MTPDrafter,
        device           : torch.device,
        vocab_size       : int,
        num_spec_tokens  : int,
    ) -> None:
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.num_spec_tokens = num_spec_tokens
        self.vocab_size = vocab_size
        self.drafter = drafter
        self.sampler = Sampler(device, vocab_size)

    def draft(self, states: list[MTPState]) -> None:
        for st in states:
            self._draft(st)

    def _draft(self, st: MTPState) -> None:
        K = self.num_spec_tokens
        n_drafts = st.n_drafts
        sampling = not st.sampling_params.is_greedy
        st.draft_tokens = []
        st.draft_probs = (
            torch.zeros(K, self.vocab_size, dtype=torch.float32, device=self.device)
            if sampling else None
        )
        if n_drafts == 0:
            return

        params = st.sampling_params
        start = 0 if st.mtp_kv is None else st.mtp_kv[0].shape[1]  # carried KV rows

        carry_tok = torch.tensor(st.carry_tokens[start:], dtype=torch.int32, device=self.device)
        carry_pos = torch.tensor(
            st.carry_positions[start:], dtype=torch.int64, device=self.device
        )
        _, logits, h, st.mtp_kv = self.drafter.draft(
            carry_tok, carry_pos, st.carry_hidden[start:], st.mtp_kv
        )
        st.draft_tokens.append(self.sampler.draft_token(logits[-1], params))
        if sampling:
            st.draft_probs[0] = self.sampler._target_dist(logits[-1], params)
        mtp_hidden = h[-1]
        mtp_kv = st.mtp_kv

        first_pos = st.carry_positions[-1]
        for j in range(1, n_drafts):
            draft_tok = torch.tensor([st.draft_tokens[-1]], dtype=torch.int32, device=self.device)
            draft_pos = torch.tensor([first_pos + j], dtype=torch.int64, device=self.device)
            _, logits, h, mtp_kv = self.drafter.draft(
                draft_tok, draft_pos, mtp_hidden.unsqueeze(0), mtp_kv
            )
            st.draft_tokens.append(self.sampler.draft_token(logits[-1], params))
            if sampling:
                st.draft_probs[j] = self.sampler._target_dist(logits[-1], params)
            mtp_hidden = h[-1]
