"""Speculative-decoding throughput benchmark: MTP vs non-MTP, controlled variables.

Both runs use the SAME seeded prompt batch, the SAME httpx load driver and the SAME
metrics -- the only thing that differs is the server's ``--enable-mtp`` flag
(``--num-spec-tokens`` comes from the library's ServerArgs, default 4). tp size comes
from ``--tp-size``. Server flags are parsed by the library's
``picosgl.server.args.parse_args`` -- nothing is redefined here.

Examples:
  python benchmarks/throughput/bench_spec_decode.py --model-path <model> --tp-size 2 \
      --input-len 64 --output-len 256 --num-prompts 32 --num-spec-tokens 4
  python benchmarks/throughput/bench_spec_decode.py --model-path <model> \
      --mode mtp --out /tmp/mtp.json   # single-mode run for longer data collection
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bench_common import (
    launch_server,
    make_prompts,
    parse_full,
    parse_int_list,
    print_compare,
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
            "--input-len": {"type": int, "default": 64, "help": "prompt length in tokens"},
            "--output-len": {"type": int, "default": 256, "help": "generation length in tokens"},
            "--mode": {"type": str, "choices": ["both", "nonmtp", "mtp"], "default": "both"},
            "--out": {"type": str, "default": None, "help": "write per-mode stats JSON"},
            "--dt": {"action": "store_true",
                     "help": "enable drafter/target separation for the MTP run"},
        },
    )

    if bench.mode == "both":
        modes = [("nonmtp", False), ("mtp", True)]
    elif bench.mode == "nonmtp":
        modes = [("nonmtp", False)]
    else:
        modes = [("mtp", True)]

    port = resolve_port(server_argv, server_args)

    results: dict[str, dict[str, dict]] = {}
    for label, enable_mtp in modes:
        proc = launch_server(
            server_argv,
            port=port,
            enable_mtp=enable_mtp,
            num_spec_tokens=server_args.speculative_num_draft_tokens,
            enable_dt=bench.dt and enable_mtp,
        )
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
                        mtp=enable_mtp,
                        warmup=not bench.no_warmup,
                        pbar=not bench.no_pbar,
                    )
                )
                assert stats["chunks_ok"], "SSE chunk count inconsistent (parse broken?)"
                note = f"avg_accept={stats['avg_accept']:.2f}" if enable_mtp else ""
                print_stats(
                    f"bench_spec_decode {label} tp={server_args.tp_info.size} "
                    f"in={bench.input_len} out={bench.output_len} conc={c}",
                    stats,
                    note=note,
                )
                results.setdefault(label, {})[str(c)] = stats
        finally:
            kill_server(proc)

    if bench.mode == "both":
        for c in parse_int_list(bench.num_prompts, 32):
            print(f"  concurrency = {c}")
            print_compare(results["nonmtp"][str(c)], results["mtp"][str(c)])
    if bench.out:
        with open(bench.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  wrote {bench.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
