from __future__ import annotations

from typing import TYPE_CHECKING

from picosgl.core import Batch

from .ar import ARManagerBase


class DecodeManager(ARManagerBase):
    def schedule_next_batch(self) -> Batch | None:
        self.inflight_uids[0] = self.inflight_uids[1]
        self.inflight_uids[1] = set()
        
        if (0 < len(self.inflight_uids[0]) < self.token_budget 
            and len(self.running_reqs) > len(self.inflight_uids[0])):
            # skip one iteration to try achieving a larger batch size
            return None

        scheduled_token = 0
        reqs = []
        for uid, req in sorted(
            self.running_reqs.items(), 
            key=lambda x: x[0] in self.inflight_uids[0]
            # prioritize the requests that have not been scheduled in the last iteration
        ):
            if scheduled_token >= self.token_budget:
                break
            self.inflight_uids[1].add(uid)
            reqs.append(req)
            scheduled_token += 1

        if reqs:
            return Batch(reqs=reqs, phase="decode")
        else:
            return None

    def advance_for_overlap(self, batch: Batch) -> None:
        for req in batch.reqs:
            req.complete_n(1)