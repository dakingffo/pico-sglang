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


# ---- Round-shape vocabulary (symbols used throughout this file) -----------------
#
# Per request, per verify round:
#   K           draft budget = num_spec_tokens: the max drafts a req may produce.
#   n_drafts    ACTUAL drafts this req produced this round, = min(K, remain_len - 1).
#               Equals K only when enough tokens remain (this is the per-req
#               actualization of the budget K; early code spelled it K subscript r).
#   verify_rows = n_drafts + 1: rows this req contributes to the verify forward,
#               [old_bonus, draft_0 .. draft_{n_drafts-1}].
#   C           cached_len at round start (already-committed length).
#   num_sampled committed tokens this round = accepted drafts + 1: the +1 is the
#               new bonus (token at position C+num_sampled, the next round's carry),
#               so num_sampled >= 1 always and ranges over [1, n_drafts + 1]. Only
#               the accepted-draft COUNT is capped at n_drafts.
#   T           total verify-batch rows = sum of verify_rows over all reqs.
#
# Fixed-shape tensors pad to the budget, so "K" / "K+1" in a shape are caps, not
# the per-req actuals:
#   extend_token[i] is (K+1,), -1-padded past verify_rows;
#   draft_tokens (bs, K); draft_probs (bs, K, vocab).
#
# MTP carry window:
#   W           window_size (128): max (token, hidden) pairs carried.
#   window_len  effective window = min(W, rows available).


@dataclass
class VerifyState:
    draft_tokens    : list[int]
    draft_probs     : torch.Tensor | None
    carry_positions : list[int]
    carry_hidden    : torch.Tensor # (window_len, hidden) main-model hidden, leading-token
    next_alloc_pos: int                 # pages [0, next_alloc_pos) already allocated; next round allocates from here
    verify_window_end: int | None = None  # this round's verify-window upper bound (C+n_drafts+1); abort frees [cached_len, it)
    last_commit     : tuple[int, int, list[int], int] | None = None  
                      # (C, num_sampled, committed, n_drafts)
    mtp_kv          : tuple[torch.Tensor, torch.Tensor] | None = None


class VerifyManager(ARManagerBase):
    """Speculative-decoding orchestrator for the MTP path.

    Replaces DecodeManager when ``--enable-mtp`` (fixed at startup -- this manager is never
    swapped for a decode manager at runtime). Owns no sampling math (that lives in
    Engine / Sampler.reject_sample); this class only schedules verify batches and commits
    their results. 
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
            if st is not None and st.verify_window_end is not None:
                C, D = req.cached_len, st.verify_window_end
                if D > C:
                    self.cache_manager._free(self.page_table[req.table_idx, C:D])
        return req

    def on_prefill_done(self, req: Request, full_hidden: torch.Tensor, mapping) -> None:
        """Hand off a finished prefill req to the verify loop.

        The last row (the bonus at C) is necessarily paired with hidden at C-1.
        hidden_C does not exist after prefill since the bonus is sampled, not processed.
        """
        req_hidden = full_hidden[mapping == req.table_idx]
        C = req.cached_len
        W = self.window_size
        window_len = min(W, req_hidden.shape[0])
        st = VerifyState(
            draft_tokens=[],
            draft_probs=None,
            # window ends at the bonus position C (token_pool[C] = bonus)
            carry_positions=list(range(C - window_len + 1, C + 1)),
            carry_hidden=req_hidden[-window_len:].contiguous(),
            next_alloc_pos=C,
        )
        self.running_reqs[req.uid] = req
        self._state[req.table_idx] = st

    def settle(self, ctx: Context) -> None:
        """Commit the in-flight verify round.

        Called by the scheduler as the first line of ``_schedule_next_batch``.
        This deferred commit is what lets schedule_next_batch draft from THIS round's
        carry and process read its num_sampled while the NEXT forward already overlaps.
        """
        pending, self._pending = self._pending, None
        if pending is None or not pending[0].batch.is_verify:
            return
        forward_input, output = pending
        output.copy_done_event.synchronize()

        extend_token = output.next_tokens_gpu  # (bs, K+1) int32
        full_hidden = output.full_hidden       # (sum of verify_rows, hidden)
        pool = getattr(ctx, "linear_state", None)
        offset = 0

        for i, req in enumerate(forward_input.batch.reqs):
            verify_rows = req.extend_len  # = n_drafts + 1 (this req's rows in logits/full_hidden)
            row_start = offset
            offset += verify_rows
            st = self._state.get(req.table_idx)
            if st is None:  # aborted between forward and settle
                continue
            n_drafts = verify_rows - 1
            C = req.cached_len
            seq = extend_token[i]  # (K+1,) int32, -1 padded beyond verify_rows

            num_sampled = n_drafts + 1
            bonus_idx = n_drafts
            for j in range(n_drafts):
                if int(seq[j].item()) != st.draft_tokens[j]:
                    num_sampled = j + 1
                    bonus_idx = j
                    break
            new_bonus = int(seq[bonus_idx].item())

            req.device_len = C + 1
            req.complete_n(num_sampled)
            if pool is not None:
                pool.rollback_to([req], num_sampled)
            self.token_pool[req.table_idx, req.cached_len] = new_bonus
            st.next_alloc_pos = C + min(num_sampled, self.num_spec_tokens) + 1

            self._update_carry(st, full_hidden, row_start, C, num_sampled)

            st.last_commit = (
                C, num_sampled, [int(seq[j].item()) for j in range(num_sampled)], n_drafts
            )

            finish = not req.can_decode
            if not req.sampling_params.ignore_eos:
                for j in range(num_sampled):
                    if int(seq[j].item()) == self.eos_token_id:
                        finish = True
                        break
            if finish:
                self.running_reqs.pop(req.uid, None)

    def schedule_next_batch(self) -> Batch | None:
        if not self.running_reqs:
            return None
        reqs = sorted(self.running_reqs.values())
        K = self.num_spec_tokens
        for req in reqs:
            self._draft_loop(req, self._state[req.table_idx])

        for req in reqs:
            C = req.cached_len
            n_drafts = min(K, req.remain_len - 1)
            req.device_len = C + n_drafts + 1
            self._state[req.table_idx].verify_window_end = C + n_drafts + 1

        for req in reqs:
            st = self._state[req.table_idx]
            C = req.cached_len
            req.cached_len = st.next_alloc_pos
            self.cache_manager.allocate_paged([req])
            req.cached_len = C

        for req in reqs:
            st = self._state[req.table_idx]
            C = req.cached_len
            if st.draft_tokens:
                self.token_pool[req.table_idx, C + 1 : C + 1 + len(st.draft_tokens)].copy_(
                    torch.tensor(st.draft_tokens, dtype=torch.int32, device=self.device)
                )

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
                C, num_sampled, committed, n_drafts = st.last_commit

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
                if num_sampled <= n_drafts:
                    suffix = self.page_table[req.table_idx, C + num_sampled + 1 : C + n_drafts + 1]
                    self.cache_manager._free(suffix)

                if finish and req not in self.finished_reqs:
                    # A partial commit on finish (num_sampled <= n_drafts) leaks the bonus-position
                    # page: it sits inside this round's allocated window but is excluded from
                    # both the suffix free ([C+num_sampled+1, C+n_drafts+1)) and cache_req's
                    # page_indices ([:cached_len]). Can-decode finishes are always full commits
                    # (num_sampled == n_drafts+1 == remain_len), so this fires only for EOS finishes.
                    if num_sampled <= n_drafts:
                        self.cache_manager._free(
                            self.page_table[req.table_idx, C + num_sampled : C + num_sampled + 1]
                        )
                    self.remove_req(req)
                    self._free_req_resources(ctx, req)
                    new_finished.add(req)

                # ---- round over for this req. Clear verify_window_end ONLY if it still
                # ---- equals the round just processed (no newer round was scheduled for
                # ---- this req yet -- a continuing req's schedule has already overwritten
                # ---- it with the next round's upper bound, which must stay live for a
                # ---- mid-round abort).
                if st.verify_window_end == C + n_drafts + 1:
                    st.verify_window_end = None
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
            # The guard is REQUIRED: on an n_drafts==0 finish round _update_carry runs after no
            # materialization, so mtp_kv is still None. Must happen here (settle), not in
            # _draft_loop -- deferring it would let dropped positions leak into the next
            # round's attention (the causal mask attends to all of past_k).
            if excess > 0 and st.mtp_kv is not None:
                k, v = st.mtp_kv
                st.mtp_kv = (k[:, excess:].contiguous(), v[:, excess:].contiguous())

    def _draft_loop(self, req: Request, st: VerifyState) -> None:
        """Generate the next round's n_drafts drafts from the current carry window.

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
        n_drafts = min(K, remain - 1) if remain > 0 else 0
        sampling = not req.sampling_params.is_greedy
        st.draft_tokens = []
        st.draft_probs = (
            torch.zeros(K, self.vocab_size, dtype=torch.float32, device=self.device)
            if sampling
            else None
        )
        if n_drafts == 0:
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
        for j in range(1, n_drafts):
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
