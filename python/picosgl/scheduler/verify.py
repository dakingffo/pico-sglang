from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import torch

from picosgl.core import Batch, Request
from picosgl.message import DetokenizeMsg
from picosgl.utils import align_ceil

if TYPE_CHECKING:
    from picosgl.engine import VerifyOutput

    from .scheduler import ForwardInput, Scheduler


@dataclass
class _VerifyState:
    """Per-request spec-decode state (keyed by req.table_idx).

    carry_positions / carry_hidden hold the MTP draft carry window with the DeepSeek-MTP
    *leading-token* pairing: entry r is (token at position p, main hidden at position p-1),
    so feeding the window through ``mtp.draft`` predicts position p+1 (the row after the
    window's last token). The window therefore always ends at the current bonus position.

    Pairing is empirically settled (experiments/mtp_pairing_test.py): the module predicts
    p+1 from (token_p, hidden_X, pos p) for either alignment, but same-position hidden
    (X=p) correlates better with the main model's next-token logits (cosine 0.86 vs 0.72).
    Same-position is however unattainable for the window's last row -- the bonus token is
    sampled, so its own hidden is never computed (and for a partial accept the hidden at
    that position belongs to the rejected draft). Leading-token is therefore forced for the
    bonus row and, by extension, for the whole carry. It affects only draft acceptance
    rate, never output correctness (rejection sampling guarantees the target either way).

    draft_probs is (K, vocab) fp32 -- the MTP head's distribution per draft step (used for
    the residual rejection sampling) -- or None when the request is greedy (argmax drafts,
    which carry no distribution).
    """

    draft_tokens   : list[int]
    draft_probs    : torch.Tensor | None
    carry_positions: list[int]
    carry_hidden   : torch.Tensor          # (W_eff, hidden) main-model hidden, leading-token
    last_paged_until: int
    # upper bound of THIS round's verify window (cached_len + K_r + 1), set by
    # schedule_next_batch and cleared by process. Non-None only while a round that
    # scheduled this req is in flight -- abort_req uses it to free the window's pages.
    round_window_end: int | None = None


class VerifyManager:
    """Speculative-decoding orchestrator for the MTP path.

    Replaces DecodeManager when ``--enable-mtp`` (fixed at startup -- this manager is never
    swapped for a decode manager at runtime). Owns no sampling math (that lives in
    Engine / Sampler.reject_sample); this class only schedules verify batches and commits
    their results:

    * ``schedule_next_batch`` builds a verify ForwardInput: every running req processed as a
      K_r+1-token mini-prefill [old_bonus, draft_0..draft_{K_r-1}], with pages allocated for
      exactly the missing region and drafts staged into the token pool.
    * ``process`` derives num_sampled by comparing the verify forward's extend_token against
      this manager's drafts (count-by-comparison), commits via ``complete_n`` + linear-state
      rollback (pure pointer arithmetic), frees the rejected suffix, emits the committed
      tokens, and re-seeds the next round's carry + drafts.

    Position alignment: the verify forward processes positions C..C+K_r, logits[C+i] is the
    target's prediction for the token at position C+i+1, and draft_i (token at C+1+i) is
    rejected against it. Accepting num_sampled tokens commits positions C+1..C+num_sampled,
    so the new bonus (written to token_pool at C+num_sampled) is sampled from logits at
    C+num_sampled-1. ``device_len == cached_len + 1`` always holds outside a verify round.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        num_spec_tokens: int,
        page_size: int,
        window_size: int = 128,
    ) -> None:
        self.scheduler = scheduler
        self.num_spec_tokens = num_spec_tokens
        self.page_size = page_size
        self.window_size = window_size

        self.device = scheduler.device
        self.token_pool = scheduler.token_pool
        self.page_table = scheduler.engine.page_table
        self.cache_manager = scheduler.cache_manager
        self.sampler = scheduler.engine.sampler
        self.mtp = scheduler.engine.model.mtp
        self.vocab_size = self.sampler.vocab_size

        self.running_reqs: dict[int, Request] = {}
        self._state: dict[int, _VerifyState] = {}
        # A verify round must be fully processed (process) before the next one is
        # scheduled: the next round's drafts (carry update) and page allocation
        # (last_paged_until) are both derived from THIS round's commit, which only
        # exists after process runs. Overlapping rounds (scheduling round N+1 while
        # N is in flight) would re-use stale drafts and re-allocate the same pages.
        self.round_inflight = False

    # ================================ manager interface ================================

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0

    @property
    def inflight_tokens(self) -> int:
        return sum(
            align_ceil(req.remain_len, self.page_size)
            for req in self.running_reqs.values()
        )

    def filter_reqs(self, reqs: Iterable[Request]) -> None:
        self.running_reqs |= {req.uid: req for req in reqs if req.can_decode}

    def remove_req(self, req: Request) -> None:
        self.running_reqs.pop(req.uid, None)
        self._state.pop(req.table_idx, None)

    def abort_req(self, uid: int) -> Request | None:
        req = self.running_reqs.pop(uid, None)
        if req is not None:
            st = self._state.pop(req.table_idx, None)
            # Mid-round abort: this req's verify-window pages [cached_len, round_window_end)
            # were allocated in schedule_next_batch but process() (which frees the rejected
            # suffix and restores device_len) will never run for it. cache_req in
            # _free_req_resources only covers [:cached_len], so without this the window's
            # pages leak one round per abort.
            if st is not None and st.round_window_end is not None:
                C, D = req.cached_len, st.round_window_end
                if D > C:
                    self.cache_manager._free(self.page_table[req.table_idx, C:D])
        return req

    def add_req(self, req: Request, req_hidden: torch.Tensor) -> None:
        """Hand off a finished prefill req to the verify loop.

        ``req_hidden`` is this req's rows of the prefill batch's full hidden (positions
        [c0, C), C = req.cached_len; the sampled bonus already sits at token_pool[.., C]).
        Seeds the carry window (last min(W, len) rows, leading-token paired) and generates
        the first round's drafts. The last row (the bonus at C) is necessarily paired with
        hidden at C-1 -- hidden_C does not exist after prefill since the bonus is sampled,
        not processed. From round 2 on the same is true of the new bonus each round, so the
        whole loop stays leading-token (see _VerifyState).
        """
        C = req.cached_len
        W = self.window_size
        W_eff = min(W, req_hidden.shape[0])
        st = _VerifyState(
            draft_tokens=[],
            draft_probs=None,
            # window ends at the bonus position C (token_pool[C] = bonus)
            carry_positions=list(range(C - W_eff + 1, C + 1)),
            carry_hidden=req_hidden[-W_eff:].contiguous(),
            last_paged_until=C,
        )
        self.running_reqs[req.uid] = req
        self._state[req.table_idx] = st
        self._draft_loop(req, st)

    # ==================================== scheduling ====================================

    def schedule_next_batch(self) -> ForwardInput | None:
        """Build a verify ForwardInput for all running reqs, or None.

        Temporarily mutates each req's device_len to cached_len + K_r + 1 so fi.py's
        extend-prefill branch, ``_make_positions`` and the linear layer slicing all agree;
        ``process`` restores it.
        """
        if not self.running_reqs or self.round_inflight:
            return None
        reqs = sorted(self.running_reqs.values())
        K = self.num_spec_tokens

        # ---- temp device_len: K_r = min(K, remain_len - 1) keeps even a full commit's
        # ---- bonus position within [0, max_device_len).
        for req in reqs:
            C = req.cached_len
            K_r = min(K, req.remain_len - 1)
            req.device_len = C + K_r + 1
            # remember the window upper bound for mid-round abort page cleanup
            self._state[req.table_idx].round_window_end = C + K_r + 1

        # ---- page allocation: only the missing [last_paged_until, C+K_r+1) region. Never
        # ---- allocate with (cached_len=C, device_len=C+K+1): that would re-allocate the
        # ---- existing position C and leak one page per round.
        for req in reqs:
            st = self._state[req.table_idx]
            C = req.cached_len
            req.cached_len = st.last_paged_until
            self.cache_manager.allocate_paged([req])
            req.cached_len = C

        # ---- stage this round's drafts into the token pool (positions C+1..C+K_r).
        for req in reqs:
            st = self._state[req.table_idx]
            C = req.cached_len
            if st.draft_tokens:
                self.token_pool[req.table_idx, C + 1 : C + 1 + len(st.draft_tokens)].copy_(
                    torch.tensor(st.draft_tokens, dtype=torch.int32, device=self.device)
                )

        # ---- batch assembly + draft fields for reject_sample.
        batch = Batch(reqs=reqs, phase="verify")
        batch.padded_reqs = reqs
        bs = len(reqs)
        draft_tokens = torch.full((bs, K), -1, dtype=torch.int32, device=self.device)
        draft_probs: torch.Tensor | None = None
        for i, req in enumerate(reqs):
            st = self._state[req.table_idx]
            if st.draft_tokens:
                draft_tokens[i, : len(st.draft_tokens)].copy_(
                    torch.tensor(st.draft_tokens, dtype=torch.int32, device=self.device)
                )
            if st.draft_probs is not None:
                if draft_probs is None:
                    draft_probs = torch.zeros(
                        bs, K, self.vocab_size, dtype=torch.float32, device=self.device
                    )
                draft_probs[i, :K] = st.draft_probs
        batch.draft_tokens = draft_tokens
        batch.draft_probs = draft_probs

        batch.positions = self.scheduler._make_positions(batch, self.device)
        input_mapping = self.scheduler._make_input_tuple(batch, self.device)
        batch.out_loc = self.page_table[input_mapping]
        self.scheduler.engine.attn_backend.prepare_metadata(batch)
        sample_args = self.sampler.prepare(batch)
        # verify writes committed tokens itself (process); the write tuple is unused.
        empty_write = (
            torch.tensor([], dtype=torch.int64, device=self.device),
            torch.tensor([], dtype=torch.int64, device=self.device),
        )
        from picosgl.scheduler.scheduler import ForwardInput  # circular: scheduler imports us

        self.round_inflight = True
        return ForwardInput(batch, sample_args, input_mapping, empty_write)

    # ====================================== commit ======================================

    def process(
        self, batch: Batch, output: VerifyOutput
    ) -> tuple[list[DetokenizeMsg], set[Request]]:
        """Commit a verify round and return (reply_msgs, finished_reqs).

        Runs inside the scheduler's lazy_free_region (suffix frees are batched). For each
        req: derive num_sampled, restore device_len, commit, roll back the linear state,
        free the rejected suffix, emit the committed tokens, and seed the next round.
        """
        extend_token = output.next_tokens_gpu  # (bs, K+1) int32
        full_hidden = output.full_hidden       # (T, hidden)
        eos = self.scheduler.eos_token_id
        pool = getattr(self.scheduler.engine.ctx, "linear_state", None)
        reply: list[DetokenizeMsg] = []
        new_finished: set[Request] = set()
        offset = 0
        self.round_inflight = False

        for i, req in enumerate(batch.reqs):
            num_rows = req.extend_len  # temp: K_r + 1 (rows this round in logits/full_hidden)
            row_start = offset
            offset += num_rows
            st = self._state.get(req.table_idx)
            if st is None:  # aborted between schedule and process
                continue
            K_r = num_rows - 1
            C = req.cached_len
            seq = extend_token[i]  # (K+1,) int32, -1 padded beyond K_r+1

            # ---- derive num_sampled + new bonus (count-by-comparison): first index j
            # ---- where the target's commit differs from draft_j -> num_sampled = j+1 and
            # ---- the bonus is extend_token[j]; all matched -> num_sampled = K_r+1.
            num_sampled = K_r + 1
            bonus_idx = K_r
            for j in range(K_r):
                if int(seq[j].item()) != st.draft_tokens[j]:
                    num_sampled = j + 1
                    bonus_idx = j
                    break
            new_bonus = int(seq[bonus_idx].item())

            # ---- commit: restore the real device_len, advance by num_sampled, roll the
            # ---- linear-state slot pointer to the committed boundary (zero memcpy).
            req.device_len = C + 1
            req.complete_n(num_sampled)
            if pool is not None:
                pool.rollback_to([req], num_sampled)
            self.token_pool[req.table_idx, req.cached_len] = new_bonus
            st.last_paged_until = C + min(num_sampled, self.num_spec_tokens) + 1
            if num_sampled <= K_r:
                suffix = self.page_table[req.table_idx, C + num_sampled + 1 : C + K_r + 1]
                self.cache_manager._free(suffix)

            # ---- emit the num_sampled newly-committed tokens (accepted drafts + new bonus)
            # ---- in position order. The old_bonus at position C was emitted when it was
            # ---- created (prefill sample / previous round's bonus), so it is not repeated.
            # ---- EOS may appear at ANY committed index (a draft agreed with the target);
            # ---- it terminates the stream (committed tokens after EOS are appended for KV
            # ---- bookkeeping only, never emitted).
            committed = [int(seq[j].item()) for j in range(num_sampled)]
            finish = not req.can_decode
            stop = num_sampled
            for idx, tok in enumerate(committed):
                if not req.sampling_params.ignore_eos and tok == eos:
                    stop = idx + 1
                    finish = True
                    break
            # ---- MTP sends ONE DetokenizeMsg per req per round with the full committed
            # ---- token list (the detokenize worker extends decoded_ids by the list, so
            # ---- each uid appears at most once per batch and the per-uid in-batch
            # ---- surr-advance is unnecessary). The msg carries the raw committed tokens
            # ---- INCLUDING any trailing EOS, exactly like the non-MTP decode path; the
            # ---- worker strips a trailing EOS so it is never user-visible.
            reply.append(
                DetokenizeMsg(uid=req.uid, next_token=committed[:stop], finished=finish)
            )

            # ---- append_host: input_ids grows to the new device_len (all committed tokens,
            # ---- including any trailing EOS; it is KV bookkeeping, not the user-visible
            # ---- stream which the EOS check above already stopped).
            req.append_host(torch.tensor(seq[:num_sampled].tolist(), dtype=torch.int32))

            if finish and req not in self.scheduler.finished_reqs:
                # A partial commit on finish (num_sampled <= K_r) leaks the bonus-position
                # page: it sits inside this round's allocated window but is excluded from
                # both the suffix free ([C+num_sampled+1, C+K_r+1)) and cache_req's
                # page_indices ([:cached_len]). Can-decode finishes are always full commits
                # (num_sampled == K_r+1 == remain_len), so this fires only for EOS finishes.
                if num_sampled <= K_r:
                    self.cache_manager._free(
                        self.page_table[req.table_idx, C + num_sampled : C + num_sampled + 1]
                    )
                self.remove_req(req)
                self.scheduler._free_req_resources(req)
                new_finished.add(req)
            else:
                self._update_carry_and_draft(req, st, full_hidden, row_start, C, num_sampled)
            # round over for this req: the window pages are committed (or the req is gone)
            st.round_window_end = None

        return reply, new_finished

    # ====================================== drafting ====================================

    def _update_carry_and_draft(
        self,
        req: Request,
        st: _VerifyState,
        full_hidden: torch.Tensor,
        row_start: int,
        C: int,
        num_sampled: int,
    ) -> None:
        """Grow the carry window with this round's committed positions and redraft.

        full_hidden rows [row_start, row_start+num_sampled) are the main hidden at
        positions [C, C+num_sampled-1]; paired leading-token with the committed positions
        [C+1, C+num_sampled] they extend the carry (token_p, hidden_{p-1}). The window is
        truncated to the last ``window_size`` entries.
        """
        W = self.window_size
        st.carry_positions.extend(range(C + 1, C + 1 + num_sampled))
        new_hidden = full_hidden[row_start : row_start + num_sampled]
        st.carry_hidden = (
            torch.cat([st.carry_hidden, new_hidden], dim=0)
            if st.carry_hidden is not None
            else new_hidden
        )
        excess = len(st.carry_positions) - W
        if excess > 0:
            st.carry_positions = st.carry_positions[excess:]
            st.carry_hidden = st.carry_hidden[excess:]
        self._draft_loop(req, st)

    def _draft_loop(self, req: Request, st: _VerifyState) -> None:
        """Generate the next round's K_r drafts from the current carry window.

        Step-0 feeds the whole window (tokens + leading-token hidden) through
        ``mtp.draft`` with past_kv=None; the last row's logits predict the token right after
        the window -> draft_0, and the window's KV is returned. Step-j (j>=1) feeds
        [draft_{j-1}] at position C'+j with the MTP output hidden from step j-1 and the
        carried KV -> draft_j, where C' = the window's last position (the bonus). Drafts are
        drawn via the request's sampler distribution (greedy = argmax).
        """
        K = self.num_spec_tokens
        remain = req.remain_len
        K_r = min(K, remain - 1) if remain > 0 else 0
        sampling = not req.sampling_params.is_greedy
        st.draft_tokens = []
        st.draft_probs = (
            torch.zeros(K, self.vocab_size, dtype=torch.float32, device=self.device)
            if sampling
            else None
        )
        if K_r == 0:
            return

        params = req.sampling_params
        carry_tokens = self.token_pool[req.table_idx, st.carry_positions]  # (W_eff,) int32
        carry_positions = torch.tensor(
            st.carry_positions, dtype=torch.int64, device=self.device
        )
        _, logits, h, kv = self.mtp.draft(
            carry_tokens, carry_positions, st.carry_hidden, None
        )
        draft = self.sampler.draft_token(logits[-1], params)
        st.draft_tokens.append(draft)
        if sampling:
            st.draft_probs[0] = self.sampler._target_dist(logits[-1], params)
        mtp_hidden = h[-1]
        C_prime = st.carry_positions[-1]
        for j in range(1, K_r):
            tok = torch.tensor([draft], dtype=torch.int32, device=self.device)
            pos = torch.tensor([C_prime + j], dtype=torch.int64, device=self.device)
            _, logits, h, kv = self.mtp.draft(
                tok, pos, mtp_hidden.unsqueeze(0), kv
            )
            draft = self.sampler.draft_token(logits[-1], params)
            st.draft_tokens.append(draft)
            if sampling:
                st.draft_probs[j] = self.sampler._target_dist(logits[-1], params)
            mtp_hidden = h[-1]
