from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from picosgl.engine import VerifyOutput
from picosgl.core import Batch, Request, Context
from picosgl.message import DetokenizeMsg, DraftReplyMsg, DraftStepReq
from picosgl.speculator import DraftState
from picosgl.utils import div_ceil

from .ar import ARManagerBase, ForwardInput

if TYPE_CHECKING:
    from picosgl.speculator import DraftBroadcastReceiver, DrafterClient

    from .cache import CacheManager
    from .config import SchedulerConfig
    from .table import TableManager

DraftBroadcastSend = Callable[[DraftReplyMsg, torch.Tensor | None], None]
DraftBroadcastRecv = Callable[[int], tuple[DraftReplyMsg, torch.Tensor | None]]


class VerifyManager(ARManagerBase):
    """MTP verify manager with the drafter split into its own process.

    The target keeps only a thin per-request ``DraftState`` (draft_tokens / draft_probs);
    the rolling carry window (positions / tokens / hidden, ``mtp_kv``) lives in the
    drafter's ``MTPState``. ``on_prefill_done`` seeds the drafter with the request's
    terminal window (``client.init``); ``schedule_next_batch`` sends the committed rows
    since the last step and blocks on the draft reply (``client.step``); ``process``
    buffers committed rows in ``_pending_carry`` for the next step and notifies the
    drafter on finish / abort. On non-primary TP ranks ``client`` is a
    ``DraftBroadcastReceiver`` and results arrive via rank0's broadcast.
    """

    def __init__(
        self,
        config        : SchedulerConfig,
        device        : torch.device,
        cache_manager : CacheManager,
        table_manager : TableManager,
        eos_token_id  : int,
        client        : DrafterClient | DraftBroadcastReceiver,
        vocab_size    : int,
        broadcast     : tuple[DraftBroadcastSend | None, DraftBroadcastRecv | None] | None = None,
    ) -> None:
        super().__init__(config, device, cache_manager, table_manager, eos_token_id)
        self.page_table = table_manager.page_table
        self.num_spec_tokens = config.speculative_num_draft_tokens
        self.window_size = config.speculator_window_size
        self.client = client
        self.vocab_size = vocab_size
        self._broadcast = broadcast if broadcast is not None else (None, None)
        self._state_table: dict[int, DraftState] = {}
        # committed rows since the last step, keyed by uid -> (positions, tokens, hidden)
        self._pending_carry: dict[int, tuple[list[int], list[int], torch.Tensor]] = {}

    def abort_req(self, uid: int) -> tuple[Request | None, bool]:
        inflight: bool = uid in self.inflight_uids[1]
        self.inflight_uids[1].discard(uid)
        req = self.running_reqs.pop(uid, None)
        if req is None:
            return None, False
        else:
            req.aborted = True
            self._state_table.pop(req.table_idx, None)
            self._pending_carry.pop(uid, None)
            self.client.remove(uid)
            return req, inflight

    def on_prefill_done(self, req: Request, full_hidden: torch.Tensor, mapping) -> None:
        req_hidden = full_hidden[mapping == req.table_idx]
        C = req.cached_len
        window_len = min(self.window_size, req_hidden.shape[0])
        # window ends at the bonus position C (token_pool[C] = bonus)
        positions = list(range(C + 1 - window_len, C + 1))
        tokens = self.token_pool[req.table_idx, positions].tolist()
        hidden = req_hidden[-window_len:].contiguous()
        self.client.init(
            req.uid, req.table_idx, positions, tokens, hidden, req.sampling_params
        )
        # MTP verify reserve: allocate the K+1 reserve slots once and set the baseline to
        # the prefill terminal state (state after position C-1). Pure index bookkeeping.
        if self.cache_manager.state_pool is not None:
            begin = self.cache_manager.draft_state
            slots = self.cache_manager._allocate(needed_states = self.num_spec_tokens + 1)[1]
            self.cache_manager.state_table[req.table_idx, begin: begin + self.num_spec_tokens + 1] = slots
            req.baseline_slot = int(
                self.cache_manager.state_table[req.table_idx, (C - 1) // self.page_size]
            )
        self.running_reqs[req.uid] = req
        self._state_table[req.table_idx] = DraftState()

    def schedule_next_batch(self) -> Batch | None:
        K = self.num_spec_tokens

        self.inflight_uids[0] = self.inflight_uids[1]
        self.inflight_uids[1] = set()

        scheduled_token = 0
        reqs = []
        step_reqs = []
        hidden_rows = []
        for uid, req in self.running_reqs.items():
            if scheduled_token >= self.token_budget:
                break
            elif uid not in self.inflight_uids[0]:
                self.inflight_uids[1].add(uid)
                reqs.append(req)
                scheduled_token += K
                C = req.cached_len
                remain = req.remain_len
                n_drafts = min(K, remain - 1) if remain > 0 else 0
                req.device_len = C + n_drafts + 1
                pc = self._pending_carry.pop(uid, None)
                ap = pc[0] if pc is not None else []
                at = pc[1] if pc is not None else []
                step_reqs.append(
                    DraftStepReq(
                        uid=uid,
                        n_drafts=n_drafts,
                        append_positions=ap,
                        append_tokens=at,
                        sampling=not req.sampling_params.is_greedy,
                    )
                )
                if pc is not None:
                    hidden_rows.append(pc[2])

        if not reqs:
            return None

        appended_hidden = torch.cat(hidden_rows, dim=0) if hidden_rows else None
        has_sampling = any(sr.sampling for sr in step_reqs)

        send_bc, recv_bc = self._broadcast
        if self.client is not None:
            reply, probs = self.client.step(step_reqs, appended_hidden, has_sampling)
            if send_bc is not None:
                send_bc(reply, probs)
        else:
            assert recv_bc is not None, "non-primary verify rank needs a draft broadcast"
            reply, probs = recv_bc(self.vocab_size)

        reply_by_uid = {r.uid: r for r in reply.reqs}
        probs_by_uid: dict[int, torch.Tensor] = {}
        if probs is not None:
            off = 0
            for sr in step_reqs:
                if sr.sampling:
                    probs_by_uid[sr.uid] = probs[off : off + sr.n_drafts]
                    off += sr.n_drafts

        for req in reqs:
            st = self._state_table[req.table_idx]
            C = req.cached_len
            st.draft_tokens = reply_by_uid[req.uid].draft_tokens
            if req.uid in probs_by_uid:
                st.draft_probs = probs_by_uid[req.uid]
            if st.draft_tokens:
                self.token_pool[req.table_idx, C + 1: C + 1 + len(st.draft_tokens)].copy_(
                    torch.tensor(st.draft_tokens, dtype=torch.int32, device=self.device)
                )

        batch = Batch(reqs=reqs, phase="verify")
        bs = len(reqs)
        draft_tokens = torch.full((bs, K), -1, dtype=torch.int32, device=self.device)
        draft_probs: torch.Tensor | None = None
        for i, req in enumerate(reqs):
            st = self._state_table[req.table_idx]
            if st.draft_tokens:
                draft_tokens[i, : len(st.draft_tokens)].copy_(
                    torch.tensor(st.draft_tokens, dtype=torch.int32, device=self.device)
                )
            if st.draft_probs is not None:
                if draft_probs is None:
                    draft_probs = torch.zeros(
                        bs, K, self.vocab_size, dtype=torch.float32, device=self.device
                    )
                n = st.draft_probs.shape[0]
                draft_probs[i, :n] = st.draft_probs
        batch.draft_tokens = draft_tokens
        batch.draft_probs = draft_probs
        return batch

    def process(
        self,
        ctx          : Context,
        forward_input: ForwardInput,
        output       : VerifyOutput
    ) -> tuple[list[DetokenizeMsg], list[Request]]:
        batch = forward_input.batch
        if not batch.is_verify:
            return super().process(ctx, forward_input, output)  # prefill commit

        extend_token = output.next_tokens_gpu  # (bs, K+1) int32
        full_hidden = output.full_hidden       # (sum of num_sampled, hidden)
        offset = 0
        committed: list[tuple[int, int, list[int], int]] = [tuple()] * len(batch.reqs)

        for i, (req, extend_token) in enumerate(zip(forward_input.batch.reqs, extend_token)):
            num_sampled = req.extend_len  # = n_drafts + 1 (this req's rows in logits/full_hidden)
            row_start = offset
            offset += num_sampled
            if req.uid not in self.running_reqs:
                continue
            st = self._state_table.get(req.table_idx, None)
            n_drafts = num_sampled - 1
            C = req.cached_len

            sampled_token = []
            for j in range(n_drafts):
                tok = int(extend_token[j].item())
                if tok != st.draft_tokens[j]:
                    sampled_token.append(tok)
                    break
                sampled_token.append(tok)
            else:
                sampled_token.append(int(extend_token[n_drafts].item()))  # full-accept bonus
            num_sampled = len(sampled_token)

            req.device_len = C + 1
            req.complete_n(num_sampled)
            # commit the accepted states: pure index ops on the reserve (pin/swap/refill)
            self.cache_manager.state_commit_verify(req, C, num_sampled)
            self.token_pool[req.table_idx, req.cached_len] = sampled_token[-1]
            # buffer the committed rows (positions C+1..C+num_sampled, their token ids and
            # main-model hidden) for the drafter's next update_carry.
            self._pending_carry[req.uid] = (
                list(range(C + 1, C + 1 + num_sampled)),
                sampled_token,
                full_hidden[row_start: row_start + num_sampled],
            )
            committed[i] = (C, num_sampled, sampled_token, n_drafts)

        reply: list[DetokenizeMsg] = []
        new_finished: set[Request] = set()

        with self.cache_manager.lazy_free_region():
            for req, com in zip(batch.reqs, committed):
                if req.uid not in self.running_reqs:
                    continue
                st = self._state_table.get(req.table_idx, None)
                C, num_sampled, sampled_token, n_drafts = com

                stop, finish = num_sampled, not req.can_decode
                for idx, tok in enumerate(sampled_token):
                    if not req.sampling_params.ignore_eos and tok == self.eos_token_id:
                        stop, finish = idx + 1, True
                        break

                reply.append(
                    DetokenizeMsg(uid=req.uid, next_token=sampled_token[:stop], finished=finish)
                )
                req.append_host(torch.tensor(sampled_token, dtype=torch.int32))

                if num_sampled <= n_drafts:
                    ps = self.page_size
                    C_end = C + num_sampled
                    suffix_end = C + n_drafts + 1  # exclusive
                    for page in range(C_end // ps, div_ceil(suffix_end, ps)):
                        if C_end - 1 < page * ps:
                            p_start = page * ps
                            self.cache_manager._free(
                                self.page_table[req.table_idx, p_start: p_start + ps]
                            )

                if finish and req not in self.finished_reqs:
                    self._finish_req(req)
                    new_finished.add(req)

        self.inflight_uids[0] = set()
        self.finished_reqs = new_finished
        return reply, new_finished

    def _finish_req(self, req: Request) -> None:
        self.running_reqs.pop(req.uid, None)
        self._state_table.pop(req.table_idx, None)
        self._pending_carry.pop(req.uid, None)
        self.client.remove(req.uid)
