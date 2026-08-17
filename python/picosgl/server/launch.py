from __future__ import annotations

import sys
import multiprocessing as mp
from dataclasses import replace

import torch

from picosgl.distributed import DistributedInfo
from picosgl.env import ENV
from picosgl.utils import init_logger
from picosgl.scheduler import schedule_worker
from picosgl.speculator import PipeDataPlane, launch_drafter_worker
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

        # Local single-GPU WSL2 cannot bootstrap a 2-rank NCCL communicator ("Duplicate
        # GPU detected" — GPU-PV topology, see wsl2-nccl-2rank), so setting
        # picosgl_DRAFTER_DATA_PLANE=pipe swaps the target<->drafter data plane for a
        # PipeDataPlane pair (multiprocessing.Queue round-trip). Production default is
        # NCCL, unchanged.
        use_pipe = (
            server_args.speculative_algorithm is not None
            and ENV.DRAFTER_DATA_PLANE.value == "pipe"
        )
        target_dp: PipeDataPlane | None = None
        drafter_dp: PipeDataPlane | None = None
        if use_pipe:
            to_drafter, to_target = mp.Queue(), mp.Queue()
            drafter_dev = (
                f"cuda:{world_size}" if server_args.enable_dt_separation else "cuda:0"
            )
            target_dp = PipeDataPlane(
                torch.device("cuda:0"), 0, to_drafter, to_target,
                dtype=server_args.dtype,
            )
            drafter_dp = PipeDataPlane(
                torch.device(drafter_dev), 1, to_drafter, to_target,
                dtype=server_args.dtype,
            )

        for i in range(world_size):
            args = replace(
                server_args,
                tp_info=DistributedInfo(i, world_size),
            )
            kwargs: dict = {
                "args": args,
                "ack_queue": ack_queue,
            }
            if i == 0 and target_dp is not None:
                kwargs["drafter_data_plane"] = target_dp
            mp.Process(
                target=schedule_worker,
                kwargs=kwargs,
                daemon=False,
                name=f"picosgl-TP{i}-scheduler",
            ).start()

        if server_args.speculative_algorithm is not None:
            kwargs = {
                "args": server_args,
                "ack_queue": ack_queue,
            }
            if drafter_dp is not None:
                kwargs["data_plane"] = drafter_dp
            mp.Process(
                target=launch_drafter_worker,
                kwargs=kwargs,
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

        num_workers = (
            server_args.num_tokenizer + 2
            + (1 if server_args.speculative_algorithm is not None else 0)
        )
        for _ in range(num_workers):
            logger.info(ack_queue.get())

    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
