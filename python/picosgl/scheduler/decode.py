from __future__ import annotations

from typing import TYPE_CHECKING

from picosgl.core import Batch

from .ar import ARManagerBase

if TYPE_CHECKING:
    from .ar import ForwardInput


class DecodeManager(ARManagerBase):
    """One-token-per-step AR manager (the non-MTP path). Pure local scheduling: a decode
    batch is just all running reqs sorted by uid; the engine's ``complete_one`` advances
    each req a single position per forward."""

    def schedule_next_batch(self) -> Batch | None:
        return Batch(reqs=sorted(self.running_reqs.values()), phase="decode") if self.runnable else None

    def after_forward(self, forward_input: ForwardInput, output) -> None:
        # Non-MTP prefill -> decode handoff happens here too: filter_reqs on every non-verify
        # batch adds the freshly-prefilled reqs to the decode loop.
        self.filter_reqs(forward_input.batch.reqs)

    # settle / on_prefill_done inherit the base no-ops; process inherits the non-verify emit.
