from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from picosgl.core import Request, ChunkedRequest, Context
from picosgl.engine import ForwardInput, ForwardOutput
from picosgl.message import DetokenizeMsg
from picosgl.utils import align_ceil

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
        self.running_reqs: dict[int, Request] = {}
        self.finished_reqs: set[Request] = set()

    @property
    def page_size(self) -> int:
        return self.config.page_size

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0

    @property
    def inflight_tokens(self) -> int:
        return sum(
            align_ceil(req.remain_len, self.page_size)
            for req in self.running_reqs.values()
        )

    def remove_req(self, req: Request) -> None:
        self.running_reqs.pop(req.uid, None)

    def abort_req(self, uid: int) -> Request | None:
        return self.running_reqs.pop(uid, None)

    def _free_req_resources(self, ctx: Context, req: Request) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def on_prefill_done(self, req: Request, full_hidden, mapping) -> None:
        """Prefill -> AR handoff hook. decode: no-op; verify (MTP): seed the carry."""
        return None

    def advance_for_next_schedule(self, forward_input: ForwardInput) -> None:
        """Advance the state of the manager after a forward pass, before scheduling the next batch.

        Note that only decode_manager needs to implement this,
        verify_manager disable a inflight req from being scheduled again.
        """

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
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                if finished and req not in self.finished_reqs:
                    self.remove_req(req)
                    self._free_req_resources(ctx, req)
                    new_finished.add(req)
                elif batch.is_prefill:
                    if req.can_decode:
                        self.running_reqs[req.uid] = req
                    if batch.full_hidden is not None:
                        self.on_prefill_done(req, batch.full_hidden, forward_input.input_tuple[0])
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished
        return reply
