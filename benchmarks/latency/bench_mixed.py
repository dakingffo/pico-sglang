"""Mixed short/long-text latency + throughput benchmark.

Launches one pico-sglang server and, for each concurrency level, drives a batch that
mixes a majority of short prompts with a few long ones (ratio configurable via
``--long-frac``), reporting:

  - latency  : TTFT / TPOT percentiles per group (short and long separately)
  - throughput : decode / total tok/s and req/s for the whole mixed batch
  - the long:short ratio, printed in the header and every summary row

The server's own flags (--tensor-parallel-size, --speculative-algorithm, ...) are
parsed by the library and passed through untouched; the bench adds ``--dt`` which maps
to the server's ``--enable-dt-separation``. Examples:

  # DT separation: target tp=1 on one card, drafter on the other
  python benchmarks/latency/bench_mixed.py --model-path <model> --tensor-parallel-size 1 \
      --dt --num-prompts 1,4,8,16,32,64,128,256 \
      --short-len 128 --short-out 64 --long-len 2048 --long-out 512 --long-frac 0.2

  # No DT separation: target tp=2 (drafter shares rank0's card)
  python benchmarks/latency/bench_mixed.py --model-path <model> --tensor-parallel-size 2 \
      --num-prompts 1,4,8,16,32,64,128,256
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bench_common import (
    kill_server,
    launch_server,
    make_mixed_prompts,
    parse_full,
    parse_int_list,
    print_stats,
    resolve_num_spec_tokens,
    resolve_port,
    run_online_bench_tics,
    summarize_group,
    wait_server_ready,
)


def _f(x: float) -> str:
    return "    —" if x == 0 else f"{x:>7.2f}"


def print_mixed_summary(rows: list[tuple]) -> None:
    print("=" * 112)
    print("  mixed short/long sweep summary")
    print(f"  {'conc':>5} {'short':>5} {'long':>5} {'long%':>6} "
          f"{'sTTFT':>7} {'lTTFT':>7} {'sTPOT':>7} {'lTPOT':>7} "
          f"{'dec tok/s':>10} {'tot tok/s':>10} {'req/s':>8}")
    for conc, n_short, n_long, s, s_short, s_long in rows:
        long_pct = 100.0 * n_long / conc if conc else 0.0
        print(f"  {conc:>5} {n_short:>5} {n_long:>5} {long_pct:>5.1f}% "
              f"{_f(s_short['ttft_ms'][1]) if n_short else '     —':>7} "
              f"{_f(s_long['ttft_ms'][1]):>7} "
              f"{_f(s_short['tpot_ms'][1]) if n_short else '     —':>7} "
              f"{_f(s_long['tpot_ms'][1]):>7} "
              f"{s['decode_tok_per_s']:>10.1f} {s['total_tok_per_s']:>10.1f} "
              f"{s['req_per_s']:>8.2f}")
    print("=" * 112)


def main() -> int:
    server_args, bench, server_argv = parse_full(
        sys.argv[1:],
        extra={
            "--short-len": {"type": int, "default": 128, "help": "short prompt length in tokens"},
            "--short-out": {"type": int, "default": 64, "help": "short generation length in tokens"},
            "--long-len": {"type": int, "default": 2048, "help": "long prompt length in tokens"},
            "--long-out": {"type": int, "default": 512, "help": "long generation length in tokens"},
            "--long-frac": {"type": float, "default": 0.2,
                            "help": "fraction of prompts that are long (default 0.2)"},
            "--dt": {"action": "store_true",
                     "help": "launch with --enable-dt-separation (drafter on its own device)"},
        },
    )
    frac = max(0.0, min(1.0, bench.long_frac))
    concs = parse_int_list(bench.num_prompts, 32)

    print(f"  mixed workload: {int(frac * 100)}% long (in {bench.long_len}, out {bench.long_out}) + "
          f"{int((1 - frac) * 100)}% short (in {bench.short_len}, out {bench.short_out})")

    port = resolve_port(server_argv, server_args)
    proc = launch_server(
        server_argv, port=port, enable_specualtive_decoding=True,
        num_spec_tokens=resolve_num_spec_tokens(server_argv),
        enable_dt=bench.dt,
    )
    try:
        base = wait_server_ready(port)
        rows: list[tuple] = []
        for c in concs:
            n_long = min(c, max(1, round(c * frac)))
            n_short = c - n_long
            segments = [(bench.short_len, n_short), (bench.long_len, n_long)]
            prompts, in_lens = make_mixed_prompts(server_args.model_path, segments, bench.seed + c)
            out_lens = [bench.short_out] * n_short + [bench.long_out] * n_long
            summary, tics = asyncio.run(
                run_online_bench_tics(
                    base, server_args.model_path, prompts, in_lens, out_lens,
                    mtp=True, warmup=not bench.no_warmup, pbar=not bench.no_pbar,
                )
            )
            assert summary["chunks_ok"], "SSE chunk count inconsistent (parse broken?)"
            short_idx = list(range(n_short))
            long_idx = list(range(n_short, c))
            s_short = summarize_group(tics, in_lens, out_lens, short_idx, mtp=True) if n_short else {}
            s_long = summarize_group(tics, in_lens, out_lens, long_idx, mtp=True) if n_long else {}
            print_stats(
                f"bench_mixed tp={server_args.tp_info.size} conc={c} "
                f"short={n_short} long={n_long}",
                summary,
                note=f"avg_accept={summary['avg_accept']:.2f}",
            )
            if n_short:
                print_stats(f"  short group (n={n_short})", s_short)
            if n_long:
                print_stats(f"  long group (n={n_long})", s_long)
            rows.append((c, n_short, n_long, summary, s_short, s_long))
        print_mixed_summary(rows)
        return 0
    finally:
        kill_server(proc)


if __name__ == "__main__":
    sys.exit(main())
