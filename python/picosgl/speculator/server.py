from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING

import torch

from picosgl.distributed import DistributedInfo, tp_override
from picosgl.message import BaseDrafterMsg, DraftStepMsg
from picosgl.models.drafters import BaseDrafterModel
from picosgl.message.queue import ZmqPullQueue, ZmqPushQueue
from picosgl.utils import Registry

from .base import EngineBase
from .data_plane import NCCLDataPlane
from .drafters.mtp import MTPEngine
from .runner import SpeculatorRunner
from .client import (
    BroadcastSpeculatorClient,
    LocalSpeculatorClient,
    RemoteSpeculatorClient,
    SpeculatorClientBase,
)

if TYPE_CHECKING:
    from picosgl.scheduler.config import SchedulerConfig
    from picosgl.scheduler.scheduler import Scheduler


SUPPORTED_DRAFTER_ENGINE = Registry[type[EngineBase]]("Drafter Engine")
SUPPORTED_DRAFTER_ENGINE.register("MTP")(MTPEngine)


def make_drafter_engine(
    drafter: BaseDrafterModel | None,
    device : torch.device,
    config : SchedulerConfig,
) -> EngineBase:
    algorithm = config.speculative_algorithm
    assert algorithm is not None
    return SUPPORTED_DRAFTER_ENGINE[algorithm].from_config(drafter, device, config)


# independent speculator process when DT separation is enabled
@torch.inference_mode()
def speculator_worker(
    *,
    args      : SchedulerConfig,
    ack_queue : mp.Queue[str] | None = None,
) -> None:
    device = torch.device(
        f"cuda:{args.tp_info.size}" if args.dt_separation else "cuda:0"
    )
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    stream = torch.cuda.Stream(device=device)
    torch.cuda.set_stream(stream)

    # the drafter is always tp=1 (standalone rank 0): the split controls the DEVICE it
    # runs on (--enable-dt-separation), not the drafter's TP sharding. The layer builders
    # (LinearColumnParallel etc.) read the global TP info, which the main engine sets for
    # itself — this process must set it before constructing the model.
    with tp_override(DistributedInfo(0, 1)):
        runner = SpeculatorRunner(
            engine=make_drafter_engine(None, device, args),
            data_plane=NCCLDataPlane(device, rank=1, dtype=args.dtype),
            recv=ZmqPullQueue(args.zmq_drafter_addr, create=False, decoder=DraftStepMsg.decoder), 
            reply=ZmqPushQueue(args.zmq_drafter_reply_addr, create=False, encoder=BaseDrafterMsg.encoder)
        )

        if ack_queue is not None:
            ack_queue.put("Speculator is ready")
        runner.run_forever()


# called by scheduler
def make_speculator_client(
    sche: Scheduler, config: SchedulerConfig
) -> SpeculatorClientBase:
    mc = config.model_config
    if not config.tp_info.is_primary():
        return BroadcastSpeculatorClient(
            config, sche.device, mc.vocab_size, mc.hidden_size,
            scheduler_io=sche
        )
    if config.dt_separation:
        return RemoteSpeculatorClient(
            config, sche.device, mc.vocab_size, mc.hidden_size,
            scheduler_io=sche
        )
    else:
        engine = sche.engine.speculator
        assert engine is not None
        return LocalSpeculatorClient(
            config, sche.device, mc.vocab_size, mc.hidden_size,
            engine=engine, scheduler_io=sche
        )
