from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING

import torch

from picosgl.distributed import DistributedInfo, set_tp_info
from picosgl.message import BaseDrafterMsg, DraftStepMsg
from picosgl.models.drafters import Qwen3_5MTPDrafter
from picosgl.utils import ZmqPullQueue, ZmqPushQueue

from .data_plane import DataPlane, NCCLDataPlane
from .drafters.mtp import MTPEngine
from .runner import DrafterRunner

if TYPE_CHECKING:
    from picosgl.scheduler.config import SchedulerConfig


@torch.inference_mode()
def launch_drafter_worker(
    *,
    args       : SchedulerConfig,
    data_plane : DataPlane | None = None,
    ack_queue  : mp.Queue[str] | None = None,
) -> None:
    """Launch the drafter process (mirrors ``tokenize_worker``).

    Loads the standalone drafter on ``cuda:{tp_size}`` when DT separation is on (else
    ``cuda:0``, sharing the target rank0's device), builds the MTP engine, and runs the
    DrafterRunner loop over the zmq control plane. ``data_plane`` defaults to NCCL; local
    tests inject a PipeDataPlane end.
    """
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
    set_tp_info(DistributedInfo(0, 1))

    model = Qwen3_5MTPDrafter(args.model_config)
    model.load_weights(args.speculative_draft_model_path, device)
    engine = MTPEngine(
        model, device, args.model_config.vocab_size, args.speculative_num_draft_tokens
    )
    if data_plane is None:
        data_plane = NCCLDataPlane(device, rank=1, dtype=args.dtype)

    recv = ZmqPullQueue(args.zmq_drafter_addr, create=False, decoder=DraftStepMsg.decoder)
    reply = ZmqPushQueue(args.zmq_drafter_reply_addr, create=False, encoder=BaseDrafterMsg.encoder)
    runner = DrafterRunner(engine, data_plane, recv, reply)

    if ack_queue is not None:
        ack_queue.put("Drafter is ready")
    runner.run_forever()
