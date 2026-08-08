from __future__ import annotations

import os
from dataclasses import dataclass, field

from picosgl.engine import EngineConfig


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int  = 8192
    cache_type       : str  = "radix"
    offline_mode     : bool = False
    _unique_suffix   : str  = field(default_factory=lambda: f".pid={os.getpid()}")

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/picosgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/picosgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/picosgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True


