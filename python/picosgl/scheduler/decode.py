from __future__ import annotations

from typing import TYPE_CHECKING

from picosgl.core import Batch, ChunkedRequest

from .ar import ARManagerBase

if TYPE_CHECKING:
    from .ar import ForwardInput


class DecodeManager(ARManagerBase):
    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None

        return Batch(reqs=sorted(self.running_reqs.values()), phase="decode")

    def advance_for_next_schedule(self, forward_input: ForwardInput) -> None:
        batch = forward_input.batch
        for req in batch.reqs:
            if isinstance(req, ChunkedRequest):
                continue
            req.complete_n(1)