from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from picosgl.core import Batch, Request
from picosgl.utils import align_ceil

@dataclass
class DecodeManager:
    page_size   : int
    running_reqs: dict[int, Request] = field(default_factory=dict)

    def filter_reqs(self, reqs: Iterable[Request]) -> None:
        self.running_reqs |= {req.uid: req for req in reqs if req.can_decode}

    def remove_req(self, req: Request) -> None:
        self.running_reqs.pop(req.uid, None)

    def abort_req(self, uid: int) -> Request | None:
        return self.running_reqs.pop(uid, None)

    @property
    def inflight_tokens(self) -> int:
        return sum(align_ceil(req.remain_len, self.page_size) for req in self.running_reqs.values())

    def schedule_next_batch(self) -> Batch | None:
        return Batch(reqs=sorted(self.running_reqs.values()), phase="decode") if self.runnable else None

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
