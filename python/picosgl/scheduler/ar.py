from __future__ import annotations

from typing import TYPE_CHECKING
from itertools import chain

import torch

from picosgl.core import Request, ChunkedRequest, Context
from picosgl.engine import ForwardInput, ForwardOutput
from picosgl.message import DetokenizeMsg
from picosgl.utils import align_ceil, div_ceil

if TYPE_CHECKING:
    from .cache import CacheManager
    from .config import SchedulerConfig
    from .table import TableManager


class ARManagerBase:
    def __init__(
        self,
        config       : SchedulerConfig,
        device       : torch.device,
        cache_manager: CacheManager,
        table_manager: TableManager,
        eos_token_id : int,
    ) -> None:
        self.config = config
        self.cache_manager = cache_manager
        self.table_manager = table_manager
        self.token_pool = table_manager.token_pool
        self.page_table = table_manager.page_table
        self.eos_token_id = eos_token_id
        self.device = device
        self.token_budget = config.decode_batch_budget
        self.running_reqs: dict[int, Request] = {}
        self.inflight_uids: list[set[int], set[int]] = [set(), set()] # (last, next)
        self.finished_reqs: set[Request] = set()

    @property
    def page_size(self) -> int:
        return self.config.page_size

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0

    @property
    def need_tokens(self) -> int:
        return sum(
            align_ceil(req.remain_len, self.page_size)
            for req in self.running_reqs.values()
        )

    def abort_req(self, uid: int) -> Request | None:
        inflight: bool = uid in self.inflight_uids[1]
        self.inflight_uids[1].discard(uid)
        req = self.running_reqs.pop(uid, None)
        if req is None:
            return None
        if inflight:
            C, D = req.cached_len, req.device_len
            ps = self.page_size
            for page in range(div_ceil(C, ps), div_ceil(D, ps)):
                p_start = page * ps
                self.cache_manager._free(
                    self.page_table[req.table_idx, p_start: p_start + ps]
                )
        return req

    def _free_req_resources(self, ctx: Context, req: Request) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def on_prefill_done(self, req: Request, full_hidden, mapping) -> None:
        """Prefill -> AR handoff hook. decode: no-op; verify (MTP): seed the carry."""
        return None

    def process(
        self,
        ctx          : Context,
        forward_input: ForwardInput,
        output       : ForwardOutput
    ) -> list[DetokenizeMsg]:
        batch = forward_input.batch
        next_tokens_cpu = output.next_tokens_cpu
        reply: list[DetokenizeMsg] = []
        new_finished: set[Request] = set()

        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedRequest):
                    continue
                if not batch.is_prefill and req.uid not in self.running_reqs:
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                req.complete_n(1)

                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                if finished and req not in self.finished_reqs:
                    self._finish_req(req)
                    self._free_req_resources(ctx, req)
                    new_finished.add(req)
                elif batch.is_prefill:
                    if req.can_decode:
                        self.running_reqs[req.uid] = req
                    if batch.full_hidden is not None:
                        self.on_prefill_done(req, batch.full_hidden, forward_input.input_tuple[0])
                    self.cache_manager.cache_req(req, finished=False)

        self.inflight_uids[0] = []
        self.finished_reqs = new_finished
        return reply

    def _finish_req(self, req: Request) -> None:
        self.running_reqs.pop(req.uid, None)