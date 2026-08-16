from __future__ import annotations

from typing import TYPE_CHECKING

from picosgl.core import Batch

from .ar import ARManagerBase


class DecodeManager(ARManagerBase):
    def schedule_next_batch(self) -> Batch | None:
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
                scheduled_token += 1

        if reqs:
            return Batch(reqs=reqs, phase="decode")
        else:
            return None