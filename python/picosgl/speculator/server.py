from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING

import torch

from picosgl.distributed import DistributedInfo, tp_override
from picosgl.message import BaseDrafterMsg, DraftStepMsg
from picosgl.models.drafters import Qwen3_5MTPDrafter
from picosgl.utils import ZmqPullQueue, ZmqPushQueue

from .data_plane import NCCLDataPlane
from .drafters.mtp import MTPEngine
from .runner import DrafterRunner
from .client import BroadcastDrafterClient, DrafterClientBase, LocalDrafterClient, RemoteDrafterClient

if TYPE_CHECKING:
    from picosgl.engine.config import EngineConfig
    from picosgl.scheduler.config import SchedulerConfig
    from picosgl.scheduler.scheduler import Scheduler


# independent drafter process
@torch.inference_mode()
def drafter_worker(
    *,
    args      : SchedulerConfig,
    ack_queue : mp.Queue[str] | None = None,
) -> None:
    device = torch.device(
        f"cuda:{args.tp_info.size}" if args.enable_dt_separation else "cuda:0"
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
        model = Qwen3_5MTPDrafter(args.model_config)
        model.load_weights(args.speculative_draft_model_path, device)
        engine = MTPEngine(
            model, device, args.model_config.vocab_size, args.speculative_num_draft_tokens
        )
        runner = DrafterRunner(
            engine,
            data_plane=NCCLDataPlane(device, rank=1, dtype=args.dtype),
            recv=ZmqPullQueue(args.zmq_drafter_addr, create=False, decoder=DraftStepMsg.decoder), 
            reply=ZmqPushQueue(args.zmq_drafter_reply_addr, create=False, encoder=BaseDrafterMsg.encoder)
        )

        if ack_queue is not None:
            ack_queue.put("Drafter is ready")
        runner.run_forever()


# called by engine
def make_local_drafter(device: torch.device, config: EngineConfig):
    with tp_override(DistributedInfo(0, 1)):
        mc = config.model_config
        drafter = Qwen3_5MTPDrafter(mc)
        drafter.load_weights(config.speculative_draft_model_path, device)
    return drafter


# called by scheduler
def make_drafter_client(sche: Scheduler, config: SchedulerConfig) -> DrafterClientBase:
    mc = config.model_config
    if not config.tp_info.is_primary():
        return BroadcastDrafterClient(
            config, sche.device, mc.vocab_size, mc.hidden_size,
            scheduler_io=sche
        )
    if config.enable_dt_separation:
        return RemoteDrafterClient(
            config, sche.device, mc.vocab_size, mc.hidden_size,
            scheduler_io=sche
        )
    else:
        engine = MTPEngine(
            sche.engine.drafter, sche.device, mc.vocab_size, config.speculative_num_draft_tokens
        )
        return LocalDrafterClient(
            config, sche.device, mc.vocab_size, mc.hidden_size,
            engine=engine, scheduler_io=sche
        )
