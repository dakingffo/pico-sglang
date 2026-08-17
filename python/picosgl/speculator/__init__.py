"""Speculative-decoding drafter package.

Split-process architecture: the target runs the verify (acceptance) logic in
``scheduler.verify.VerifyManager``; the drafter (MTP head / standalone draft model) runs in
its own process behind ``DrafterRunner``, reached over a zmq control plane and an NCCL data
plane. ``EngineBase``/``DraftState`` are the interface contract; ``drafters.mtp`` provides
the Qwen3.5 MTP engine.
"""

from .base import DraftState, EngineBase
from .client import DraftBroadcastReceiver, DrafterClient
from .data_plane import DataPlane, DataPlaneSizes, NCCLDataPlane, PipeDataPlane
from .drafters import MTPEngine, MTPState
from .runner import DrafterRunner
from .server import launch_drafter_worker

__all__ = [
    "DraftState",
    "EngineBase",
    "DrafterClient",
    "DraftBroadcastReceiver",
    "DataPlane",
    "DataPlaneSizes",
    "NCCLDataPlane",
    "PipeDataPlane",
    "MTPEngine",
    "MTPState",
    "DrafterRunner",
    "launch_drafter_worker",
]
