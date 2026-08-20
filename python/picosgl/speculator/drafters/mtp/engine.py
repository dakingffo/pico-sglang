from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from picosgl.engine import Sampler

from ...base import EngineBase
from .state import MTPState

if TYPE_CHECKING:
    from picosgl.models.drafters import Qwen3_5MTPDrafter


class MTPEngine(EngineBase):
    """Eager batched MTP draft engine.

    Carry windows are left-padded and recomputed together at depth zero. Later draft depths
    reuse that round's dense batched KV. ``draft_sequential`` retains the original per-request
    implementation as a numerical oracle. Sampling uses each request's own parameters and
    the same target distribution consumed by rejection sampling.
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
        self._draft_batch(states)

    def draft_sequential(self, states: list[MTPState]) -> None:
        """Correctness oracle for tests and numerical debugging."""
        for st in states:
            self._draft_request(st)

    def _draft_batch(self, states: list[MTPState]) -> None:
        """Draft all requests depth-by-depth with a dense, left-padded carry window."""
        K = self.num_spec_tokens
        for st in states:
            assert 0 <= st.n_drafts <= K
            st.draft_tokens = []
            # The batched path recomputes the bounded carry window each round. Drop a
            # sequential-oracle cache if callers switch paths while debugging.
            st.mtp_kv = None
            st.draft_probs = (
                torch.zeros(K, self.vocab_size, dtype=torch.float32, device=self.device)
                if not st.sampling_params.is_greedy else None
            )

        active_states = [st for st in states if st.n_drafts > 0]
        if not active_states:
            return

        batch_size = len(active_states)
        max_window = max(len(st.window_tokens) for st in active_states)
        assert max_window > 0
        hidden_size = active_states[0].window_hidden.shape[-1]
        input_ids = torch.zeros(
            batch_size, max_window, dtype=torch.int32, device=self.device
        )
        positions = torch.zeros(
            batch_size, max_window, dtype=torch.int64, device=self.device
        )
        hidden = torch.zeros(
            batch_size,
            max_window,
            hidden_size,
            dtype=active_states[0].window_hidden.dtype,
            device=self.device,
        )
        valid = torch.zeros(
            batch_size, max_window, dtype=torch.bool, device=self.device
        )

        for i, st in enumerate(active_states):
            length = len(st.window_tokens)
            assert length == len(st.window_positions) == st.window_hidden.shape[0]
            start = max_window - length
            input_ids[i, start:] = torch.as_tensor(
                st.window_tokens, dtype=torch.int32, device=self.device
            )
            positions[i, start:] = torch.as_tensor(
                st.window_positions, dtype=torch.int64, device=self.device
            )
            hidden[i, start:] = st.window_hidden
            valid[i, start:] = True

        logits, mtp_hidden, kv, kv_valid = self.drafter.draft_batch(
            input_ids, positions, hidden, valid
        )
        next_tokens = self._sample_batch(logits, active_states, 0)
        token_matrix = torch.full(
            (batch_size, K), -1, dtype=torch.int32, device=self.device
        )
        token_matrix[:, 0] = next_tokens
        mtp_hidden = mtp_hidden[:, -1:, :]

        n_drafts = torch.tensor(
            [st.n_drafts for st in active_states], dtype=torch.int64, device=self.device
        )
        first_positions = torch.tensor(
            [st.window_positions[-1] for st in active_states],
            dtype=torch.int64,
            device=self.device,
        )
        for j in range(1, max(st.n_drafts for st in active_states)):
            step_valid = n_drafts > j
            step_ids = torch.where(step_valid, next_tokens, torch.zeros_like(next_tokens))[:, None]
            step_positions = (first_positions + j)[:, None]
            logits, mtp_hidden, kv, kv_valid = self.drafter.draft_batch(
                step_ids,
                step_positions,
                mtp_hidden,
                step_valid[:, None],
                kv,
                kv_valid,
            )
            next_tokens = self._sample_batch(logits, active_states, j)
            token_matrix[:, j] = torch.where(step_valid, next_tokens, token_matrix[:, j])

        token_rows = token_matrix.tolist()  # one synchronization for the whole draft batch
        for st, row in zip(active_states, token_rows):
            st.draft_tokens = row[: st.n_drafts]

    def _sample_batch(
        self,
        logits: torch.Tensor,
        states: list[MTPState],
        step: int,
    ) -> torch.Tensor:
        """Vectorize greedy rows; preserve each sampling request's exact distribution."""
        tokens = logits.argmax(dim=-1).to(torch.int32)
        for i, st in enumerate(states):
            if step >= st.n_drafts or st.sampling_params.is_greedy:
                continue
            dist = self.sampler._target_dist(logits[i], st.sampling_params)
            assert st.draft_probs is not None
            st.draft_probs[step] = dist
            tokens[i] = torch.multinomial(dist, 1)[0].to(torch.int32)
        return tokens

    def _draft_request(self, st: MTPState) -> None:
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

        window_tok = torch.tensor(st.window_tokens[start:], dtype=torch.int32, device=self.device)
        window_pos = torch.tensor(
            st.window_positions[start:], dtype=torch.int64, device=self.device
        )
        _, logits, h, st.mtp_kv = self.drafter.draft(
            window_tok, window_pos, st.window_hidden[start:], st.mtp_kv
        )
        st.draft_tokens.append(self.sampler.draft_token(logits[-1], params))
        if sampling:
            st.draft_probs[0] = self.sampler._target_dist(logits[-1], params)
        mtp_hidden = h[-1]
        mtp_kv = st.mtp_kv

        first_pos = st.window_positions[-1]
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
