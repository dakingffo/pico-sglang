from __future__ import annotations

import json
import os
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from picosgl.distributed import get_tp_info
from picosgl.utils import init_logger

if TYPE_CHECKING:
    from picosgl.core import Batch


logger = init_logger(__name__)


def _env_bool(name: str) -> bool:
    return os.getenv(name, "0").lower() in ("1", "true", "yes")


def _env_int(name: str) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else 0


@dataclass
class _PhaseProfile:
    batches        : int = 0
    logical_rows   : int = 0
    padded_rows    : int = 0
    logical_tokens : int = 0
    padded_tokens  : int = 0
    graph_batches  : int = 0
    schedule_ns    : int = 0
    prepare_ns     : int = 0
    submit_ns      : int = 0
    wait_ns        : int = 0
    process_ns     : int = 0
    gpu_ns         : int = 0
    logical_bs     : Counter[int] = field(default_factory=Counter)
    padded_bs      : Counter[int] = field(default_factory=Counter)

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    def as_dict(self) -> dict:
        return {
            "batches": self.batches,
            "logical_rows": self.logical_rows,
            "padded_rows": self.padded_rows,
            "logical_tokens": self.logical_tokens,
            "padded_tokens": self.padded_tokens,
            "graph_batches": self.graph_batches,
            "avg_logical_bs": self._ratio(self.logical_rows, self.batches),
            "avg_padded_bs": self._ratio(self.padded_rows, self.batches),
            "row_padding_efficiency": self._ratio(self.logical_rows, self.padded_rows),
            "token_padding_efficiency": self._ratio(
                self.logical_tokens, self.padded_tokens
            ),
            "graph_hit_rate": self._ratio(self.graph_batches, self.batches),
            "schedule_ms": self.schedule_ns / 1e6,
            "prepare_ms": self.prepare_ns / 1e6,
            "submit_ms": self.submit_ns / 1e6,
            "wait_ms": self.wait_ns / 1e6,
            "process_ms": self.process_ns / 1e6,
            "gpu_ms": self.gpu_ns / 1e6,
            "avg_gpu_ms": self._ratio(self.gpu_ns / 1e6, self.batches),
            "gpu_ms_per_logical_token": self._ratio(
                self.gpu_ns / 1e6, self.logical_tokens
            ),
            "logical_bs": dict(sorted(self.logical_bs.items())),
            "padded_bs": dict(sorted(self.padded_bs.items())),
        }


class SchedulerProfiler:
    """Low-overhead scheduler profiling, disabled unless explicitly requested.

    Set ``PICOSGL_PROFILE_SCHEDULER=1`` to collect aggregate CPU, CUDA-event,
    batch-size, padding and graph-hit statistics.  CUDA timings are harvested
    asynchronously, so serving does not synchronize once per forward.

    ``PICOSGL_PROFILE_CUDA_START`` and ``PICOSGL_PROFILE_CUDA_STOP`` select an
    inclusive range of AR forwards for Nsight Systems capture with
    ``--capture-range=cudaProfilerApi``.  Stopping the capture synchronizes once
    after the final selected forward, which is acceptable in profiling mode.
    """

    def __init__(self, device: torch.device) -> None:
        self.enabled = _env_bool("PICOSGL_PROFILE_SCHEDULER")
        self.device = device
        self.path = os.getenv("PICOSGL_PROFILE_SCHEDULER_PATH")
        self.flush_interval = _env_int("PICOSGL_PROFILE_FLUSH_INTERVAL")
        self.cuda_capture_start = _env_int("PICOSGL_PROFILE_CUDA_START")
        self.cuda_capture_stop = _env_int("PICOSGL_PROFILE_CUDA_STOP")
        self.ar_forward_count = 0
        self.ar_no_batch = 0
        self.last_flushed_forward = 0
        self.phases = {
            phase: _PhaseProfile() for phase in ("prefill", "decode", "verify")
        }
        self.pending_gpu: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    @contextmanager
    def nvtx_range(self, name: str):
        if self.enabled:
            with torch.cuda.nvtx.range(name):
                yield
        else:
            yield

    def record_batch(
        self,
        batch: Batch,
        *,
        schedule_ns: int,
        prepare_ns: int,
        graph_used: bool,
    ) -> None:
        if not self.enabled:
            return
        profile = self.phases[batch.phase]
        profile.batches += 1
        profile.logical_rows += batch.size
        profile.padded_rows += batch.padded_size
        profile.logical_tokens += sum(req.extend_len for req in batch.reqs)
        profile.padded_tokens += sum(req.extend_len for req in batch.padded_reqs)
        profile.graph_batches += int(graph_used)
        profile.schedule_ns += schedule_ns
        profile.prepare_ns += prepare_ns
        profile.logical_bs[batch.size] += 1
        profile.padded_bs[batch.padded_size] += 1

    def begin_submit(
        self, phase: str
    ) -> tuple[torch.cuda.Event, torch.cuda.Event] | None:
        if not self.enabled:
            return None
        if phase in ("decode", "verify"):
            self.ar_forward_count += 1
            if self.ar_forward_count == self.cuda_capture_start:
                torch.cuda.profiler.start()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return start, end

    def finish_submit(
        self,
        phase    : str,
        events   : tuple[torch.cuda.Event, torch.cuda.Event] | None,
        submit_ns: int,
    ) -> None:
        if not self.enabled:
            return
        self.phases[phase].submit_ns += submit_ns
        assert events is not None
        start, end = events
        end.record()
        self.pending_gpu.append((phase, start, end))
        if (
            phase in ("decode", "verify")
            and self.ar_forward_count == self.cuda_capture_stop
        ):
            end.synchronize()
            torch.cuda.profiler.stop()

    def record_wait(self, phase: str, wait_ns: int) -> None:
        if self.enabled:
            self.phases[phase].wait_ns += wait_ns

    def record_process(self, phase: str, process_ns: int) -> None:
        if self.enabled:
            self.phases[phase].process_ns += process_ns

    def record_ar_no_batch(self) -> None:
        if self.enabled:
            self.ar_no_batch += 1

    def poll(self) -> None:
        if not self.enabled:
            return
        remaining = []
        for phase, start, end in self.pending_gpu:
            if end.query():
                self.phases[phase].gpu_ns += int(start.elapsed_time(end) * 1e6)
            else:
                remaining.append((phase, start, end))
        self.pending_gpu = remaining
        if (
            self.path
            and self.flush_interval > 0
            and self.ar_forward_count - self.last_flushed_forward
            >= self.flush_interval
        ):
            self._write_result()
            self.last_flushed_forward = self.ar_forward_count

    def _make_result(self) -> dict:
        tp_info = get_tp_info()
        return {
            "tp_rank": tp_info.rank,
            "ar_forward_count": self.ar_forward_count,
            "ar_no_batch": self.ar_no_batch,
            "pending_gpu_events": len(self.pending_gpu),
            "phases": {
                phase: profile.as_dict() for phase, profile in self.phases.items()
            },
        }

    def _write_result(self) -> None:
        assert self.path is not None
        tp_info = get_tp_info()
        stem, suffix = os.path.splitext(self.path)
        path = f"{stem}.rank{tp_info.rank}{suffix or '.json'}"
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self._make_result(), f, indent=2)
        os.replace(tmp_path, path)

    def dump(self) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize(self.device)
        self.poll()
        assert not self.pending_gpu

        result = self._make_result()
        logger.info("Scheduler profile:\n%s", json.dumps(result, indent=2))
        if self.path:
            self._write_result()
