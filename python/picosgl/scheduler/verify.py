from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from picosgl.engine import VerifyOutput, Sampler
from picosgl.core import Batch, Request, Context
from picosgl.message import DetokenizeMsg
from picosgl.utils import div_ceil

from .ar import ARManagerBase, ForwardInput

if TYPE_CHECKING:
    from .cache import CacheManager
    from .config import SchedulerConfig
    from .table import TableManager


@dataclass
class VerifyState:
    draft_tokens     : list[int]
    draft_probs      : torch.Tensor | None
    carry_positions  : list[int]
    carry_hidden     : torch.Tensor  # (window_len, hidden) main-model hidden, leading-token
    mtp_kv           : tuple[torch.Tensor, torch.Tensor] | None = None


class VerifyManager(ARManagerBase):
    def __init__(
        self,
        config         : SchedulerConfig,
        device         : torch.device,
        sampler        : Sampler,
        mtp            : torch.nn.Module,
        cache_manager  : CacheManager,
        table_manager  : TableManager,
        eos_token_id   : int,
        num_spec_tokens: int,
        window_size    : int = 128,
    ) -> None:
        super().__init__(config, device, cache_manager, table_manager, eos_token_id)
        self.num_spec_tokens = num_spec_tokens
        self.window_size = window_size
        self.sampler = sampler
        self.mtp = mtp
        self.vocab_size = self.sampler.vocab_size
        self._state_table: dict[int, VerifyState] = {}

    def abort_req(self, uid: int) -> Request | None:
        inflight: bool = uid in self.inflight_uids[1]
        self.inflight_uids[1].discard(uid)
        req = self.running_reqs.pop(uid, None)
        if req is None:
            return None
        self._state_table.pop(req.table_idx, None)
        if inflight:
            C, D = req.cached_len, req.device_len
            ps = self.page_size
            for page in range(div_ceil(C, ps), div_ceil(D, ps)):
                p_start = page * ps
                self.cache_manager._free(
                    self.page_table[req.table_idx, p_start: p_start + ps]
                )
        return req

    def on_prefill_done(self, req: Request, full_hidden: torch.Tensor, mapping) -> None:
        req_hidden = full_hidden[mapping == req.table_idx]
        C = req.cached_len
        window_len = min(self.window_size, req_hidden.shape[0])
        st = VerifyState(
            draft_tokens=[],
            draft_probs=None,
            # window ends at the bonus position C (token_pool[C] = bonus)
            carry_positions=list(range(C + 1 - window_len, C + 1)),
            carry_hidden=req_hidden[-window_len:].contiguous(),
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
        self._state_table[req.table_idx] = st

    def schedule_next_batch(self) -> Batch | None:
        K = self.num_spec_tokens

        self.inflight_uids[0] = self.inflight_uids[1]
        self.inflight_uids[1] = []

        scheduled_token = 0
        reqs = []
        for uid, req in self.running_reqs.items():
            if scheduled_token >= self.token_budget:
                break
            elif uid not in self.inflight_uids[0]: 
                self.inflight_uids[1].append(uid)
                reqs.append(req)
                scheduled_token += K
                st = self._state_table[req.table_idx]
                C = req.cached_len
                n_drafts = self._draft(req, st)
                req.device_len = C + n_drafts + 1
                if st.draft_tokens:
                    self.token_pool[req.table_idx, C + 1: C + 1 + len(st.draft_tokens)].copy_(
                        torch.tensor(st.draft_tokens, dtype=torch.int32, device=self.device)
                    )

        if not reqs:
            return None
        
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
                draft_probs[i, :K] = st.draft_probs
        batch.draft_tokens = draft_tokens
        batch.draft_probs = draft_probs
        return batch

    def process(
        self,
        ctx          : Context,
        forward_input: ForwardInput,
        output       : VerifyOutput
    ) -> list[DetokenizeMsg]:
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
            self._update_carry(st, full_hidden, row_start, C, num_sampled)
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
                    self._free_req_resources(ctx, req)
                    new_finished.add(req)

        self.inflight_uids[0] = []
        self.finished_reqs = new_finished
        return reply

    def _finish_req(self, req: Request) -> None:
        self.running_reqs.pop(req.uid, None)
        self._state_table.pop(req.table_idx, None)

    def _update_carry(
        self,
        st         : VerifyState,
        full_hidden: torch.Tensor,
        row_start  : int,
        C          : int,
        num_sampled: int,
    ) -> None:
        st.carry_positions.extend(range(C + 1, C + 1 + num_sampled))
        new_hidden = full_hidden[row_start: row_start + num_sampled]
        st.carry_hidden = (
            torch.cat([st.carry_hidden, new_hidden], dim=0)
            if st.carry_hidden is not None else new_hidden
        )

        reserved_len = max(0, len(st.carry_positions) - self.window_size)
        st.carry_positions = st.carry_positions[reserved_len:]
        st.carry_hidden = st.carry_hidden[reserved_len:]
        if st.mtp_kv is not None:
            k, v = st.mtp_kv
            st.mtp_kv = (k[:, reserved_len:].contiguous(), v[:, reserved_len:].contiguous())

    def _draft(self, req: Request, st: VerifyState) -> int:
        K = self.num_spec_tokens
        remain = req.remain_len
        n_drafts = min(K, remain - 1) if remain > 0 else 0
        sampling = not req.sampling_params.is_greedy
        st.draft_tokens = []
        st.draft_probs = (
            torch.zeros(K, self.vocab_size, dtype=torch.float32, device=self.device)
            if sampling else None
        )
        if n_drafts == 0:
            return 0

        params = req.sampling_params
        start = 0 if st.mtp_kv is None else st.mtp_kv[0].shape[1] # shape = (num_head, seq_len, head_dim)

        carry_tok = self.token_pool[req.table_idx, st.carry_positions[start:]]
        carry_pos = torch.tensor(
            st.carry_positions[start:], dtype=torch.int64, device=self.device
        )
        _, logits, h, st.mtp_kv = self.mtp.draft(
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
            _, logits, h, mtp_kv = self.mtp.draft(
                draft_tok, draft_pos, mtp_hidden.unsqueeze(0), mtp_kv
            )
            st.draft_tokens.append(self.sampler.draft_token(logits[-1], params))
            if sampling:
                st.draft_probs[j] = self.sampler._target_dist(logits[-1], params)
            mtp_hidden = h[-1]
        return n_drafts