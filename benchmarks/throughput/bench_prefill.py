"""Prefill-throughput benchmark: long prompts, tiny outputs.

Launches a pico-sglang HTTP server (tp-size from --tp-size), streams a batch of
``--input-len``-token prompts through ``/v1/chat/completions`` and reports prefill
throughput (input-tok/s), TTFT and req/s. Server flags are parsed by the library's
``picosgl.server.args.parse_args`` -- nothing is redefined here.

Examples:
  python benchmarks/throughput/bench_prefill.py --model-path <model> --tp-size 2 \
      --input-len 2048 --num-prompts 64
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bench_common import (
    launch_server,
    make_prompts,
    parse_full,
    parse_int_list,
    print_stats,
    resolve_port,
    run_online_bench,
    wait_server_ready,
    kill_server,
)


def main() -> int:
    server_args, bench, server_argv = parse_full(
        sys.argv[1:],
        extra={
            "--input-len": {"type": int, "default": 512, "help": "prompt length in tokens"},
            "--output-len": {"type": int, "default": 8, "help": "generation length in tokens"},
        },
    )
    port = resolve_port(server_argv, server_args)
    proc = launch_server(server_argv, port=port, enable_mtp=False, num_spec_tokens=0)
    try:
        base = wait_server_ready(port)
        for c in parse_int_list(bench.num_prompts, 32):
            prompts, in_lens = make_prompts(
                server_args.model_path, bench.input_len, c, bench.seed
            )
            out_lens = [bench.output_len] * c
            stats = asyncio.run(
                run_online_bench(
                    base,
                    server_args.model_path,
                    prompts,
                    in_lens,
                    out_lens,
                    mtp=False,
                    warmup=not bench.no_warmup,
                    pbar=not bench.no_pbar,
                )
            )
            assert stats["chunks_ok"], "chunk count != requested output tokens (SSE parse broken?)"
            print_stats(
                f"bench_prefill tp={server_args.tp_info.size} "
                f"in={bench.input_len} out={bench.output_len} conc={c}",
                stats,
            )
        return 0
    finally:
        kill_server(proc)


if __name__ == "__main__":
    sys.exit(main())
