"""Shared online throughput-benchmark harness for pico-sglang.

bench_prefill / bench_decode / bench_spec_decode all launch a real pico-sglang HTTP
server as a subprocess, wait for it to be ready, then drive a streaming
``/v1/chat/completions`` workload against it over httpx. The workload code is
byte-identical across tp sizes and MTP on/off -- the ONLY thing that changes is the
server's launch flags (``--tensor-parallel-size`` / ``--speculative-algorithm``) -- so every run
is a controlled-variable comparison.

Server arguments are NOT redefined here: the library's
``picosgl.server.args.parse_args`` is the single source of truth. This module only
splits off the bench-only workload flags via ``parse_known_args`` and hands the rest
to the library unchanged.

Run (tp=1 on a single GPU, tp>1 needs N GPUs):
  python benchmarks/throughput/bench_decode.py --model-path <model> --tp-size 2
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import signal
import socket
import subprocess
import sys
import time
from typing import Any

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
if os.path.isdir("/home/daking/.conda/envs/daking"):
    os.environ.setdefault("CUDA_HOME", "/home/daking/.conda/envs/daking")


# -------------------------------------------------------------------------------------
# argument parsing: library parse_args is the single source of truth
# -------------------------------------------------------------------------------------

# Bench-only flags. Every other flag (--model-path, --tp-size, --dtype, --host/--port,
# --page-size, --num-pages, --max-running-requests, --memory-ratio,
# --speculative-num-draft-tokens, ...) is parsed by the library's
# picosgl.server.args.parse_args.
BENCH_ONLY_FLAGS: dict[str, dict[str, Any]] = {
    "--num-prompts": {"type": str, "default": None,
                      "help": "comma-separated concurrent request counts (default 32)"},
    "--input-len": {"type": int, "default": None, "help": "prompt length in tokens"},
    "--output-len": {"type": int, "default": None, "help": "generation length in tokens"},
    "--seed": {"type": int, "default": 0, "help": "RNG seed for prompt generation (A/B uses one batch)"},
    "--no-warmup": {"action": "store_true", "help": "skip the warmup request before the timed batch"},
    "--no-pbar": {"action": "store_true", "help": "disable the progress counter"},
}


def parse_full(argv: list[str], extra: dict[str, dict[str, Any]] | None = None):
    """Split bench-only flags from server flags, then let the library parse the rest.

    Returns (ServerArgs, bench namespace, leftover server argv). The leftover argv is
    what the library parsed -- the subprocess cmd is rebuilt from it so the bench never
    redefines a server flag.
    """
    flags = dict(BENCH_ONLY_FLAGS)
    flags.update(extra or {})  # scripts override --input-len/--output-len defaults
    p = argparse.ArgumentParser(add_help=False)
    for name, kw in flags.items():
        p.add_argument(name, **kw)
    bench_ns, server_argv = p.parse_known_args(argv)

    from picosgl.server.args import parse_args

    server_args, _ = parse_args(server_argv)
    return server_args, bench_ns, server_argv


def parse_int_list(s: str | None, default: int) -> list[int]:
    """Comma-separated bench flag value -> list of ints (None/empty -> [default])."""
    if s is None:
        return [default]
    vals = [int(x) for x in s.split(",") if x.strip()]
    return vals if vals else [default]


# -------------------------------------------------------------------------------------
# prompt generation (deterministic, seeded)
# -------------------------------------------------------------------------------------

def generate_prompt(tokenizer: Any, n: int) -> str:
    """Generate a prompt of approximately `n` tokens via decode/encode round-trip."""
    vocab_size = tokenizer.vocab_size // 2
    token_ids = [random.randint(0, vocab_size) for _ in range(n)]
    for _ in range(64):
        prompt = tokenizer.decode(token_ids)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) == n:
            return prompt
        if len(token_ids) < n:
            token_ids.extend([random.randint(0, vocab_size) for _ in range(n - len(token_ids))])
        else:
            token_ids = token_ids[:n]
    raise ValueError("Failed to generate a message of the desired length.")


def make_prompts(model_path: str, input_len: int, num_prompts: int, seed: int):
    """One seeded batch of prompts + their exact token lengths. Reused across A/B runs."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    random.seed(seed)
    prompts, lens = [], []
    for _ in range(num_prompts):
        p = generate_prompt(tokenizer, input_len)
        prompts.append(p)
        lens.append(len(tokenizer.encode(p, add_special_tokens=False)))
    return prompts, lens


def make_mixed_prompts(model_path: str, segments: list[tuple[int, int]], seed: int):
    """One seeded mixed-length batch. ``segments`` is [(input_len, count), ...]; returns
    (prompts, lens) in segment order so group indices are just running offsets."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    random.seed(seed)
    prompts, lens = [], []
    for in_len, count in segments:
        for _ in range(count):
            p = generate_prompt(tokenizer, in_len)
            prompts.append(p)
            lens.append(len(tokenizer.encode(p, add_special_tokens=False)))
    return prompts, lens


# -------------------------------------------------------------------------------------
# server lifecycle: launch / ready / kill
# -------------------------------------------------------------------------------------

def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def resolve_port(server_argv: list[str], server_args: Any) -> int:
    """Honor an explicit --port; otherwise pick a free one."""
    if any(a == "--port" or a.startswith("--port=") for a in server_argv):
        return server_args.server_port
    return get_free_port()


_MTP_VALUE_FLAGS = (
    "--port", "--speculative-algorithm",
    "--speculative-num-draft-tokens", "--speculative-draft-model-path",
)
_MTP_BOOL_FLAGS = ("--enable-dt-separation",)


def _flag_value(argv: list[str], name: str) -> str | None:
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def resolve_num_spec_tokens(server_argv: list[str]) -> int:
    from picosgl.speculator import MTPSpeculatorConfig

    value = _flag_value(server_argv, "--speculative-num-draft-tokens")
    return int(value) if value is not None else MTPSpeculatorConfig.num_draft_tokens


def make_server_cmd(server_argv: list[str], *, port: int, enable_specualtive_decoding: bool, num_spec_tokens: int, enable_dt: bool = False) -> list[str]:
    """Rebuild the child cmd from the library-parsed argv, pinning what the bench controls.

    --port and every speculative/MTP flag are stripped and set deterministically so tp /
    MTP / DT comparisons differ by exactly the flag under test.
    """
    argv: list[str] = []
    i = 0
    while i < len(server_argv):
        a = server_argv[i]
        if a in _MTP_VALUE_FLAGS:
            i += 2  # skip flag + value
            continue
        if a in _MTP_BOOL_FLAGS:
            i += 1  # store_true
            continue
        if any(a.startswith(f + "=") for f in _MTP_VALUE_FLAGS + _MTP_BOOL_FLAGS):
            i += 1
            continue
        argv.append(a)
        i += 1
    argv += ["--port", str(port)]
    if enable_specualtive_decoding:
        model_path = _flag_value(server_argv, "--model-path")
        assert model_path, "MTP bench needs --model-path"
        argv += ["--speculative-algorithm", "MTP",
                 "--speculative-num-draft-tokens", str(num_spec_tokens),
                 "--speculative-draft-model-path", model_path]
        if enable_dt:
            argv += ["--enable-dt-separation"]
    return [sys.executable, "-m", "picosgl"] + argv


def launch_server(server_argv: list[str], *, port: int, enable_specualtive_decoding: bool, num_spec_tokens: int, enable_dt: bool = False):
    """Spawn the pico-sglang server in its own process group. Returns (Popen, log path)."""
    cmd = make_server_cmd(
        server_argv,
        port=port,
        enable_specualtive_decoding=enable_specualtive_decoding,
        num_spec_tokens=num_spec_tokens,
        enable_dt=enable_dt,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(_REPO, "python") + os.pathsep + env.get("PYTHONPATH", "")
    log_path = f"/tmp/picosgl_bench_{port}.log"
    print(f"  launching: {' '.join(cmd)}", flush=True)
    print(f"  server log: {log_path}", flush=True)
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            cmd,
            env=env,
            start_new_session=True,  # own process group -> killpg teardown hits every worker
            stdout=log,
            stderr=log,
        )
    return proc


def wait_server_ready(port: int, timeout: float = 900.0) -> str:
    """Poll /v1/models. The server binds its port only after every worker has acked, so
    a 200 means the model is loaded and all ranks are synced."""
    import httpx

    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + timeout
    last_report = 0.0
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=5) as c:
                if c.get(f"{base}/v1/models").status_code == 200:
                    return base
        except Exception:  # noqa: BLE001 -- server not up yet
            pass
        if time.time() - last_report >= 10:
            print(f"  waiting for server on :{port} ... {int(time.time() - deadline + timeout)}s", flush=True)
            last_report = time.time()
        time.sleep(2)
    raise TimeoutError(f"server on :{port} not ready within {timeout}s")


def kill_server(proc: subprocess.Popen) -> None:
    if proc is None or proc.poll() is not None:
        return
    pgid = os.getpgid(proc.pid)
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


# -------------------------------------------------------------------------------------
# load driver (httpx streaming) + metrics
# -------------------------------------------------------------------------------------

class _Progress:
    def __init__(self, total: int, on: bool):
        self.total, self.done, self.on = total, 0, on

    def tick(self) -> None:
        self.done += 1
        if self.on:
            print(f"\r  requests done: {self.done}/{self.total}", end="", file=sys.stderr, flush=True)


async def stream_one(client: Any, base: str, model: str, prompt: str, output_len: int, progress: _Progress | None = None):
    """Stream one chat-completions request; tic on every SSE chunk.

    Every ``data: {json}`` line is exactly one detokenizer ack (one token per ack on the
    non-MTP path, one verify round per ack on the MTP path). ``data: [DONE]`` is the
    terminator. ``ignore_eos + top_k=1 + temperature=0`` makes every request greedy and
    deterministic, and the server delivers the full ``max_tokens`` budget -- see
    ``summarize``.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_len,
        "temperature": 0.0,
        "top_k": 1,
        "stream": True,
        "ignore_eos": True,
    }
    tics = [time.perf_counter()]
    async with client.stream("POST", f"{base}/v1/chat/completions", json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            tics.append(time.perf_counter())
    if progress is not None:
        progress.tick()
    return tics


async def run_online_bench_tics(
    base: str,
    model: str,
    prompts: list[str],
    input_lengths: list[int],
    output_lengths: list[int],
    *,
    mtp: bool = False,
    warmup: bool = True,
    pbar: bool = True,
) -> tuple[dict, list[list[float]]]:
    """Drive the workload, returning (summary, per-request tics) for split reporting."""
    import httpx

    # bounded connect so a dead server fails fast; unbounded read for long streams
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30.0, read=None, write=None, pool=None),
        limits=httpx.Limits(max_connections=1024, max_keepalive_connections=128),
    )
    try:
        if warmup and prompts:
            # flush Triton autotune / CUDA graph capture / lazy detokenizer init
            await stream_one(client, base, model, prompts[0], min(8, output_lengths[0]))
        progress = _Progress(len(prompts), on=pbar and len(prompts) > 1)
        tics_all = await asyncio.gather(*[
            stream_one(client, base, model, p, o, progress)
            for p, o in zip(prompts, output_lengths, strict=True)
        ])
    finally:
        await client.aclose()
    if progress.on:
        print(file=sys.stderr, flush=True)
    return summarize(tics_all, input_lengths, output_lengths, mtp=mtp), tics_all


async def run_online_bench(
    base: str,
    model: str,
    prompts: list[str],
    input_lengths: list[int],
    output_lengths: list[int],
    *,
    mtp: bool = False,
    warmup: bool = True,
    pbar: bool = True,
) -> dict:
    summary, _ = await run_online_bench_tics(
        base, model, prompts, input_lengths, output_lengths,
        mtp=mtp, warmup=warmup, pbar=pbar,
    )
    return summary


def summarize_group(
    tics_all: list[list[float]],
    input_lengths: list[int],
    output_lengths: list[int],
    indices: list[int],
    *,
    mtp: bool = False,
) -> dict:
    """Summarize a subset of requests (by index) -- short/long split reporting."""
    sub = [tics_all[i] for i in indices]
    return summarize(
        sub,
        [input_lengths[i] for i in indices],
        [output_lengths[i] for i in indices],
        mtp=mtp,
    )


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def _stats(xs: list[float], scale: float = 1000.0) -> list[float]:
    """[avg, p50, p90, p99, max] * scale (ms by default)."""
    if not xs:
        return [0.0] * 5
    return [
        scale * sum(xs) / len(xs),
        scale * _pct(xs, 0.5),
        scale * _pct(xs, 0.9),
        scale * _pct(xs, 0.99),
        scale * max(xs),
    ]


def summarize(
    tics_all: list[list[float]],
    input_lengths: list[int],
    output_lengths: list[int],
    *,
    mtp: bool = False,
) -> dict:
    n = len(tics_all)
    in_tokens = sum(input_lengths)
    requested = sum(output_lengths)
    # len(tics)-1 == number of JSON chunks (incl. the trailing finish_reason chunk).
    # rounds == chunks - 1 == content chunks: per-token (non-MTP) or per-verify-round (MTP).
    rounds = [max(len(t) - 2, 0) for t in tics_all]
    sum_rounds = sum(rounds)
    # Actual generated token count:
    #  - MTP: each verify round commits num_sampled tokens; the server delivers the full
    #    max_tokens budget. Verified on Qwen3.5-0.8B: max_tokens=8/16 -> 8/16 tokens.
    #  - non-MTP: one content chunk == one token, so ground truth == content chunks. The
    #    scheduler delivers the full budget (DecodeManager snapshots `finished` at
    #    complete_n time instead of reading the pipeline-advanced device_len).
    if mtp:
        out_tokens = requested
        avg_accept = out_tokens / sum_rounds if sum_rounds > 0 else 0.0
        # sanity: spec-decode rounds are never more than the tokens they carry
        chunks_ok = 0 < sum_rounds <= requested
    else:
        out_tokens = sum_rounds
        avg_accept = 0.0  # one chunk == one token; avg_accept is MTP-only
        chunks_ok = sum_rounds == requested  # full budget delivered

    start = min(t[0] for t in tics_all)
    end = max(t[-1] for t in tics_all)
    dur = end - start
    first_times = [t[1] - t[0] for t in tics_all]
    accum_times = [t[i + 1] - t[i] for t in tics_all for i in range(1, len(t) - 1)]
    e2e_times = [t[-1] - t[0] for t in tics_all]
    return {
        "n": n,
        "in_tokens": in_tokens,
        "requested_out": requested,
        "out_tokens": out_tokens,
        "duration_s": dur,
        "prefill_tok_per_s": in_tokens / dur if dur > 0 else 0.0,
        "decode_tok_per_s": out_tokens / dur if dur > 0 else 0.0,
        "total_tok_per_s": (in_tokens + out_tokens) / dur if dur > 0 else 0.0,
        "req_per_s": n / dur if dur > 0 else 0.0,
        "ttft_ms": _stats(first_times),
        "tpot_ms": _stats(accum_times),
        "e2e_s": _stats(e2e_times, scale=1.0),
        "rounds": sum_rounds,
        "avg_accept": avg_accept,
        "chunks_ok": chunks_ok,
    }


# -------------------------------------------------------------------------------------
# reporting
# -------------------------------------------------------------------------------------

def _fmt_ms(x: float) -> str:
    if x >= 1000:
        return f"{int(x):>7}"
    if x >= 10:
        return f"{x:>7.2f}"
    return f"{x:>7.3f}"


def print_stats(title: str, s: dict, *, note: str = "") -> None:
    t, p, e = s["ttft_ms"], s["tpot_ms"], s["e2e_s"]
    short = "" if s["out_tokens"] == s["requested_out"] else f"  (delivered {s['out_tokens']} of {s['requested_out']})"
    print("=" * 64)
    print(f"  {title}{'  [' + note + ']' if note else ''}")
    print(f"  prompts: {s['n']}  input: {s['in_tokens']} tok  output: {s['out_tokens']} tok"
          f"{short}  duration: {s['duration_s']:.2f}s")
    print(f"  prefill: {s['prefill_tok_per_s']:10.1f} input-tok/s")
    print(f"  decode:  {s['decode_tok_per_s']:10.1f} output-tok/s")
    print(f"  total:   {s['total_tok_per_s']:10.1f} tok/s   {s['req_per_s']:8.2f} req/s")
    print(f"  TTFT(ms): avg {_fmt_ms(t[0])}  p50 {_fmt_ms(t[1])}  p90 {_fmt_ms(t[2])}  "
          f"p99 {_fmt_ms(t[3])}  max {_fmt_ms(t[4])}")
    print(f"  TPOT(ms): avg {_fmt_ms(p[0])}  p50 {_fmt_ms(p[1])}  p90 {_fmt_ms(p[2])}  "
          f"p99 {_fmt_ms(p[3])}  max {_fmt_ms(p[4])}")
    print(f"  E2E(s):   avg {e[0]:>7.3f}  p50 {e[1]:>7.3f}  p90 {e[2]:>7.3f}  "
          f"p99 {e[3]:>7.3f}  max {e[4]:>7.3f}")
    if s.get("avg_accept"):
        print(f"  verify rounds: {s['rounds']}  avg_accept: {s['avg_accept']:.2f} tok/round")
    print("=" * 64)


def print_conc_summary(rows: list[tuple[int, dict]]) -> None:
    print("=" * 66)
    print("  concurrency sweep")
    print(f"  {'conc':>6} {'decode':>10} {'total':>10} {'req/s':>8} "
          f"{'TTFT p50':>10} {'TPOT p50':>10}")
    for conc, s in rows:
        print(f"  {conc:>6} {s['decode_tok_per_s']:>10.1f} {s['total_tok_per_s']:>10.1f} "
              f"{s['req_per_s']:>8.2f} {s['ttft_ms'][1]:>10.2f} {s['tpot_ms'][1]:>10.2f}")
    print("=" * 66)


def print_compare(a: dict, b: dict) -> None:
    print("=" * 64)
    print(f"  {'':>18} {'non-mtp':>12} {'mtp':>12} {'speedup':>10}")
    for key, name in [
        ("decode_tok_per_s", "decode tok/s"),
        ("total_tok_per_s", "total tok/s"),
        ("req_per_s", "req/s"),
    ]:
        va, vb = a[key], b[key]
        print(f"  {name:>18} {va:12.1f} {vb:12.1f} {(vb / va if va > 0 else 0):>10.2f}")
    print(f"  {'avg_accept':>18} {'—':>12} {b['avg_accept']:12.2f}")
    print(f"  {'TTFT p50(ms)':>18} {a['ttft_ms'][1]:12.2f} {b['ttft_ms'][1]:12.2f}")
    print(f"  {'TPOT p50(ms)':>18} {a['tpot_ms'][1]:12.2f} {b['tpot_ms'][1]:12.2f}")
    print(f"  {'E2E p50(s)':>18} {a['e2e_s'][1]:12.3f} {b['e2e_s'][1]:12.3f}")
    print("=" * 64)
