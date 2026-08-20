from __future__ import annotations

import sys
import multiprocessing as mp
from dataclasses import replace

from picosgl.distributed import DistributedInfo
from picosgl.utils import init_logger
from picosgl.scheduler import schedule_worker
from picosgl.speculator import drafter_worker
from picosgl.tokenizer import tokenize_worker, detokenize_worker


def launch_server(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, "initializer")

    def start_subprocess() -> None:
        mp.set_start_method("spawn", force=True)

        world_size = server_args.tp_info.size
        ack_queue: mp.Queue[str] = mp.Queue()

        for i in range(world_size):
            args = replace(
                server_args,
                tp_info=DistributedInfo(i, world_size),
            )
            kwargs: dict = {
                "args": args,
                "ack_queue": ack_queue,
            }
            mp.Process(
                target=schedule_worker,
                kwargs=kwargs,
                daemon=False,
                name=f"picosgl-TP{i}-scheduler",
            ).start()

        enable_drafter_worker: bool = (
            server_args.enable_dt_separation and server_args.speculative_algorithm is not None
        )
        if enable_drafter_worker:
            mp.Process(
                target=drafter_worker,
                kwargs={
                    "args": server_args,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name="picosgl-drafter",
            ).start()

        for i in range(server_args.num_tokenizer):
            mp.Process(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "tokenizer_addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "local_bs": 1,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"picosgl-tokenizer-{i}",
            ).start()

        mp.Process(
            target=detokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "detokenizer_addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "detokenizer_id": 0,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="picosgl-detokenizer-0",
        ).start()

        num_workers = server_args.num_tokenizer + 2 + int(enable_drafter_worker)
        for _ in range(num_workers):
            logger.info(ack_queue.get())

    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
