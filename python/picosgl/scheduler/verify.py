from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from picosgl.engine import VerifyOutput, Sampler
from picosgl.core import Batch, Request, Context
from picosgl.message import DetokenizeMsg

from .ar import ARManagerBase, ForwardInput

if TYPE_CHECKING:
    from .cache import CacheManager
    from .config import SchedulerConfig
    from .table import TableManager


@dataclass
class VerifyState:
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

    last_commit is set by ``settle`` at the next schedule's start and consumed by
    ``process`` at the same iteration's end: (C, num_sampled, committed, K_r) for THIS
    round. ``process`` cannot re-derive num_sampled because the next round's drafting
    overwrites draft_tokens; it reads the settled bookkeeping instead.
    """

    draft_tokens    : list[int]
    draft_probs     : torch.Tensor | None
    carry_positions : list[int]
    carry_hidden    : torch.Tensor          # (W_eff, hidden) main-model hidden, leading-token
    last_paged_until: int
    # upper bound of the round MOST RECENTLY SCHEDULED for this req (cached_len + K_r + 1),
    # set by schedule_next_batch. Never cleared by settle; process clears it only when it
    # still equals the round it just processed (i.e. no newer round was scheduled for this
    # req -- a continuing req's next schedule has already overwritten it). abort_req uses
    # it to free a mid-round abort's window pages.
    round_window_end: int | None = None
    # settle writes this at schedule time, process reads it at the same iteration's end.
    last_commit     : tuple[int, int, list[int], int] | None = None  # (C, num_sampled, committed, K_r)
    # Carried MTP-layer window K/V in (heads, L, head_dim) layout, main-hidden-derived
    # (only committed tokens with their real main hidden -- never draft-position rows).
    # Entry r == K/V for carry_positions[r]; L == number of materialized leading positions,
    # bounded at window_size. Set by _draft_loop's step-0 materialization; front-sliced by
    # _update_carry when the window truncates. None until the first draft round (the first
    # round materializes the whole window, exactly like the pre-optimization step-0).
    mtp_kv          : tuple[torch.Tensor, torch.Tensor] | None = None


class VerifyManager(ARManagerBase):
    """Speculative-decoding orchestrator for the MTP path.

    Replaces DecodeManager when ``--enable-mtp`` (fixed at startup -- this manager is never
    swapped for a decode manager at runtime). Owns no sampling math (that lives in
    Engine / Sampler.reject_sample); this class only schedules verify batches and commits
    their results. Three responsibilities are split across the loop:

    * ``settle`` (called by the scheduler at every iteration's schedule start) commits the
      in-flight round from the previous forward -- derives num_sampled by comparing the
      verify forward's extend_token against the drafts (count-by-comparison), advances the
      reqs via ``complete_n`` + linear-state rollback, writes the new bonus to the token
      pool, extends the carry, and stores the commit into ``VerifyState.last_commit``.
      PURE internal accounting -- it never emits user-facing output.
    * ``schedule_next_batch`` drafts the next round (from the settled carry) and builds a
      K_r+1-token verify batch: every running req processed as a mini-prefill
      [old_bonus, draft_0..draft_{K_r-1}], with pages allocated for exactly the missing
      region and drafts staged into the token pool.
    * ``process`` (called by the scheduler at the iteration's end) emits the committed
      tokens, frees the rejected suffix, and finishes EOS / max_tokens requests. It reads
      the settled bookkeeping from ``last_commit``; it never re-derives the commit.

    Position alignment: the verify forward processes positions C..C+K_r, logits[C+i] is the
    target's prediction for the token at position C+i+1, and draft_i (token at C+1+i) is
    rejected against it. Accepting num_sampled tokens commits positions C+1..C+num_sampled,
    so the new bonus (written to token_pool at C+num_sampled) is sampled from logits at
    C+num_sampled-1. ``device_len == cached_len + 1`` always holds outside a verify round.
    """

    def __init__(
        self,
        config: SchedulerConfig,
        device: torch.device,
        sampler: Sampler,
        mtp: torch.nn.Module,
        cache_manager: CacheManager,
        table_manager: TableManager,
        eos_token_id: int,
        num_spec_tokens: int,
        window_size: int = 128,
    ) -> None:
        super().__init__(config, device, cache_manager, table_manager, eos_token_id)
        self.num_spec_tokens = num_spec_tokens
        self.window_size = window_size

        self.sampler = sampler
        self.mtp = mtp
        self.vocab_size = self.sampler.vocab_size

        self._state: dict[int, VerifyState] = {}
        self._pending: tuple[ForwardInput, VerifyOutput] | None = None


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
            # pages leak one round per abort. Cached_len is the pre-commit C when the abort
            # lands before settle (whole window), C+num_sampled after settle (bonus page +
            # suffix) -- round_window_end is only cleared by process, so it is always live.
            if st is not None and st.round_window_end is not None:
                C, D = req.cached_len, st.round_window_end
                if D > C:
                    self.cache_manager._free(self.page_table[req.table_idx, C:D])
        return req

    def on_prefill_done(self, req: Request, full_hidden: torch.Tensor, mapping) -> None:
        """Hand off a finished prefill req to the verify loop.

        ``full_hidden`` is this req's rows of the prefill batch's full hidden (positions
        [c0, C), C = req.cached_len; the sampled bonus already sits at token_pool[.., C]).
        Seeds the carry window (last min(W, len) rows, leading-token paired) only -- the
        next round's drafts are generated by schedule_next_batch from that carry. The last
        row (the bonus at C) is necessarily paired with hidden at C-1 -- hidden_C does not
        exist after prefill since the bonus is sampled, not processed. From round 2 on the
        same is true of the new bonus each round, so the whole loop stays leading-token
        (see VerifyState).
        """
        req_hidden = full_hidden[mapping == req.table_idx]
        C = req.cached_len
        W = self.window_size
        W_eff = min(W, req_hidden.shape[0])
        st = VerifyState(
            draft_tokens=[],
            draft_probs=None,
            # window ends at the bonus position C (token_pool[C] = bonus)
            carry_positions=list(range(C - W_eff + 1, C + 1)),
            carry_hidden=req_hidden[-W_eff:].contiguous(),
            last_paged_until=C,
        )
        self.running_reqs[req.uid] = req
        self._state[req.table_idx] = st

    # ====================================== settle ======================================

    def settle(self, ctx: Context) -> None:
        """Commit the in-flight verify round (pure internal accounting, zero output).

        Called by the scheduler as the first line of ``_schedule_next_batch``. The round's
        forward finished last iteration, so ``copy_done_event.synchronize()`` is near-zero
        wait; this deferred commit is what lets schedule_next_batch draft from THIS round's
        carry and process read its num_sampled while the NEXT forward already overlaps.
        """
        pending, self._pending = self._pending, None
        if pending is None or not pending[0].batch.is_verify:
            return
        forward_input, output = pending
        output.copy_done_event.synchronize()

        extend_token = output.next_tokens_gpu  # (bs, K+1) int32
        full_hidden = output.full_hidden       # (T, hidden)
        pool = getattr(ctx, "linear_state", None)
        offset = 0

        for i, req in enumerate(forward_input.batch.reqs):
            num_rows = req.extend_len  # temp: K_r + 1 (rows this round in logits/full_hidden)
            row_start = offset
            offset += num_rows
            st = self._state.get(req.table_idx)
            if st is None:  # aborted between forward and settle
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

            # ---- carry extension only (drafting happens in schedule_next_batch).
            self._update_carry(st, full_hidden, row_start, C, num_sampled)

            # ---- bookkeeping for process (same iteration's _process_last_data). process
            # ---- cannot re-derive num_sampled: the next schedule's drafting overwrites
            # ---- st.draft_tokens. round_window_end is NOT cleared here -- only process
            # ---- clears it (conditionally, see its docstring).
            st.last_commit = (
                C, num_sampled, [int(seq[j].item()) for j in range(num_sampled)], K_r
            )

            # ---- finish detection, identical to process's. With overlap, schedule(R+1)
            # ---- runs BEFORE process(R): if a req that EOS-terminates here stayed in
            # ---- running_reqs, schedule(R+1) would draft + pre-allocate its window pages
            # ---- [C+num_sampled+1, C+num_sampled+K_r'+1), and process(R) -- which only
            # ---- frees round R's own pages -- would leave those pre-allocated pages
            # ---- leaking (free_pages 8188/8192 in the shell e2e). Popping it now means
            # ---- the finished req is never drafted or scheduled again; process still
            # ---- emits its last round and frees round R's pages (via st + last_commit).
            finish = not req.can_decode
            if not req.sampling_params.ignore_eos:
                for j in range(num_sampled):
                    if int(seq[j].item()) == self.eos_token_id:
                        finish = True
                        break
            if finish:
                self.running_reqs.pop(req.uid, None)

    # ==================================== scheduling ====================================

    def schedule_next_batch(self) -> Batch | None:
        """Build a verify batch for all running reqs, or None.

        Drafts this round (from the settle-extended carry), temporarily mutates each req's
        device_len to cached_len + K_r + 1 so prepare_batch's positions / input tuple / the
        linear layer slicing all agree (settle restores it), allocates exactly the missing
        page region via the last_paged_until trick, and stages the drafts into the token
        pool. Returns a plain Batch; prepare_batch does the uniform prep.
        """
        if not self.running_reqs:
            return None
        reqs = sorted(self.running_reqs.values())
        K = self.num_spec_tokens

        # ---- 1) draft this round from the settled carry. Finished reqs are already out of
        # ---- running_reqs (settle popped them), so remain_len is never <= 0 here.
        for req in reqs:
            self._draft_loop(req, self._state[req.table_idx])

        # ---- 2) temp device_len: K_r = min(K, remain_len - 1) keeps even a full commit's
        # ---- bonus position within [0, max_device_len).
        for req in reqs:
            C = req.cached_len
            K_r = min(K, req.remain_len - 1)
            req.device_len = C + K_r + 1
            # remember the window upper bound for mid-round abort page cleanup
            self._state[req.table_idx].round_window_end = C + K_r + 1

        # ---- 3) page allocation: only the missing [last_paged_until, C+K_r+1) region. Never
        # ---- allocate with (cached_len=C, device_len=C+K+1): that would re-allocate the
        # ---- existing position C and leak one page per round.
        for req in reqs:
            st = self._state[req.table_idx]
            C = req.cached_len
            req.cached_len = st.last_paged_until
            self.cache_manager.allocate_paged([req])
            req.cached_len = C

        # ---- 4) stage this round's drafts into the token pool (positions C+1..C+K_r).
        for req in reqs:
            st = self._state[req.table_idx]
            C = req.cached_len
            if st.draft_tokens:
                self.token_pool[req.table_idx, C + 1 : C + 1 + len(st.draft_tokens)].copy_(
                    torch.tensor(st.draft_tokens, dtype=torch.int32, device=self.device)
                )

        # ---- 5) batch assembly + draft fields for reject_sample.
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
        return batch

    def after_forward(self, forward_input: ForwardInput, output: VerifyOutput) -> None:
        """Pure manager hook for a verify round: filter + stash the output for settle.

        Note: the next-round input write (which also covers the MTP prefill bonus) is done
        in scheduler._forward's non-verify branch, so there is no super() chain here and no
        empty-method call.
        """
        if forward_input.batch.is_verify:
            self.filter_reqs(forward_input.batch.reqs)
            self._pending = (forward_input, output)

    # ====================================== commit ======================================

    def process(
        self, 
        ctx          : Context, 
        forward_input: ForwardInput, 
        output       : VerifyOutput
    ) -> list[DetokenizeMsg]:
        """Emit a settled verify round's committed tokens and free the rejected pages.

        Runs at the iteration's end (same iteration as the settle that committed this round)
        inside a lazy_free_region. Purely user-facing output + page frees: the commit
        (num_sampled / complete_n / bonus / carry) was already done by settle; this reads
        ``st.last_commit`` and never re-derives it (draft_tokens were overwritten by the
        next round's drafting).
        """
        batch = forward_input.batch
        if not batch.is_verify:
            return super().process(ctx, forward_input, output)  # prefill commit

        reply: list[DetokenizeMsg] = []
        new_finished: set[Request] = set()
        with self.cache_manager.lazy_free_region():
            for req in batch.reqs:
                st = self._state.get(req.table_idx)
                if st is None or st.last_commit is None:  # aborted before settle
                    continue
                C, num_sampled, committed, K_r = st.last_commit

                # ---- EOS may appear at ANY committed index (a draft agreed with the
                # ---- target); it terminates the stream (committed tokens after EOS are
                # ---- appended for KV bookkeeping only, never emitted).
                stop, finish = num_sampled, not req.can_decode
                for idx, tok in enumerate(committed):
                    if not req.sampling_params.ignore_eos and tok == self.eos_token_id:
                        stop, finish = idx + 1, True
                        break
                # ---- one DetokenizeMsg per req per round with the full committed token
                # ---- list (the detokenize worker extends decoded_ids by the list). The msg
                # ---- carries the raw committed tokens INCLUDING any trailing EOS; the
                # ---- worker strips a trailing EOS so it is never user-visible.
                reply.append(
                    DetokenizeMsg(uid=req.uid, next_token=committed[:stop], finished=finish)
                )

                # ---- append_host: input_ids grows to the new device_len (all committed
                # ---- tokens, including any trailing EOS; it is KV bookkeeping, not the
                # ---- user-visible stream which the EOS check above already stopped).
                req.append_host(torch.tensor(committed, dtype=torch.int32))

                # ---- free the rejected suffix. The bonus page [C+num_sampled] is kept for
                # ---- a continuing req (it is the next round's carry token).
                if num_sampled <= K_r:
                    suffix = self.page_table[req.table_idx, C + num_sampled + 1 : C + K_r + 1]
                    self.cache_manager._free(suffix)

                if finish and req not in self.finished_reqs:
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
                    self._free_req_resources(ctx, req)
                    new_finished.add(req)

                # ---- round over for this req. Clear round_window_end ONLY if it still
                # ---- equals the round just processed (no newer round was scheduled for
                # ---- this req yet -- a continuing req's schedule has already overwritten
                # ---- it with the next round's upper bound, which must stay live for a
                # ---- mid-round abort).
                if st.round_window_end == C + K_r + 1:
                    st.round_window_end = None
                st.last_commit = None

        self.finished_reqs = new_finished
        return reply

    # ====================================== drafting ====================================

    def _update_carry(
        self,
        st: VerifyState,
        full_hidden: torch.Tensor,
        row_start: int,
        C: int,
        num_sampled: int,
    ) -> None:
        """Grow the carry window with this round's committed positions (no drafting).

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
            # Drop the same oldest positions from the carried KV. Safe because keys were
            # RoPE-rotated at their absolute positions (relative angles preserved) and the
            # causal mask is index-based (index order == position order after a front-drop).
            # The guard is REQUIRED: on a K_r==0 finish round _update_carry runs after no
            # materialization, so mtp_kv is still None. Must happen here (settle), not in
            # _draft_loop -- deferring it would let dropped positions leak into the next
            # round's attention (the causal mask attends to all of past_k).
            if excess > 0 and st.mtp_kv is not None:
                k, v = st.mtp_kv
                st.mtp_kv = (k[:, excess:].contiguous(), v[:, excess:].contiguous())

    def _draft_loop(self, req: Request, st: VerifyState) -> None:
        """Generate the next round's K_r drafts from the current carry window.

        Step-0 materializes only the window tail not yet covered by ``st.mtp_kv`` (the
        carried main-hidden K/V from prior rounds) through ``mtp.draft`` with that KV as
        past_kv; the last materialized row's logits predict the token right after the
        window -> draft_0. Since a window position's MTP K/V depends only on (token, leading
        main hidden) -- both fixed once committed -- the carried KV is bit-identical to a
        whole-window recompute, so draft_0 (and hence all downstream verify results) is
        unchanged by the incremental computation. Step-j (j>=1) feeds [draft_{j-1}] at
        position C'+j with the MTP output hidden from step j-1 and the working KV (a local
        copy extended with draft rows, discarded at round end) -> draft_j, where C' = the
        window's last position (the bonus). Drafts are drawn via the request's sampler
        distribution (greedy = argmax).
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
        # Step-0 materializes only the window tail not already in the carried KV. The MTP
        # layer's K/V for a window position depends only on (token, leading main hidden),
        # both fixed once committed, so the carried KV is bit-identical to a whole-window
        # recompute and only the num_sampled newly-committed positions need a forward.
        start = 0 if st.mtp_kv is None else st.mtp_kv[0].shape[1]
        assert start < len(st.carry_positions)  # settle appends >= 1 committed position
        carry_tokens = self.token_pool[req.table_idx, st.carry_positions[start:]]
        carry_positions = torch.tensor(
            st.carry_positions[start:], dtype=torch.int64, device=self.device
        )
        _, logits, h, kv = self.mtp.draft(
            carry_tokens, carry_positions, st.carry_hidden[start:], st.mtp_kv
        )
        # Keep ONLY the main-hidden window KV (incl. the just-materialized tail). Steps
        # j>=1 below extend a LOCAL copy of kv and discard it -- never write it back here.
        st.mtp_kv = kv
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
