from __future__ import annotations

from typing import TYPE_CHECKING

from picosgl.core import Batch

from .ar import ARManagerBase

if TYPE_CHECKING:
    from .ar import ForwardInput


class DecodeManager(ARManagerBase):
    def schedule_next_batch(self) -> Batch | None:
        return Batch(reqs=sorted(self.running_reqs.values()), phase="decode") if self.runnable else None

    def after_forward(self, forward_input: ForwardInput, output) -> None:
        self.filter_reqs(forward_input.batch.reqs)
