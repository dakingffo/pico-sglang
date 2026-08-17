from __future__ import annotations

import logging
import multiprocessing as mp
from typing import TYPE_CHECKING

import torch

from picosgl.utils import init_logger

from .config import SchedulerConfig
from .scheduler import Scheduler

if TYPE_CHECKING:
    from picosgl.speculator import DataPlane

@torch.inference_mode()
def schedule_worker(
    args: SchedulerConfig,
    ack_queue: mp.Queue[str],
    drafter_data_plane: "DataPlane | None" = None,
) -> None:
    scheduler = Scheduler(args, drafter_data_plane=drafter_data_plane)
    scheduler.sync_all_ranks()

    if args.tp_info.is_primary():
        ack_queue.put("Scheduler is ready")

    if args.silent_output:
        logging.disable(logging.INFO)

    try:
        scheduler.run_forever()
    except KeyboardInterrupt:
        logger = init_logger(__name__)
        if args.tp_info.is_primary():
            logger.info("\nScheduler exiting gracefully...")
        scheduler.shutdown()