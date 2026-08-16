"""Decode-throughput benchmark: short prompts, long generations.

Launches a pico-sglang HTTP server (tp-size from --tp-size), streams a batch of
``--output-len``-token generations through ``/v1/chat/completions`` and reports decode
throughput (output-tok/s), total throughput, TPOT and TTFT. Server flags are parsed by
the library's ``picosgl.server.args.parse_args`` -- nothing is redefined here.

Examples:
  python benchmarks/throughput/bench_decode.py --model-path <model> --tp-size 2 \
      --input-len 32 --output-len 256 --num-prompts 32
"""
import asyncio
import sys

from bench_common import (
    launch_server,
    make_prompts,
    parse_full,
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
            "--input-len": {"type": int, "default": 32, "help": "prompt length in tokens"},
            "--output-len": {"type": int, "default": 256, "help": "generation length in tokens"},
        },
    )
    prompts, in_lens = make_prompts(
        server_args.model_path, bench.input_len, bench.num_prompts, bench.seed
    )
    out_lens = [bench.output_len] * bench.num_prompts
    port = resolve_port(server_argv, server_args)
    proc = launch_server(server_argv, port=port, enable_mtp=False, num_spec_tokens=0)
    try:
        base = wait_server_ready(port)
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
            f"bench_decode tp={server_args.tp_info.size} in={bench.input_len} out={bench.output_len}",
            stats,
        )
        return 0
    finally:
        kill_server(proc)


if __name__ == "__main__":
    sys.exit(main())
