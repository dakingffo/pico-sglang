"""Step 8e: EOS / max_tokens / ignore_eos termination logic.

The Qwen3.5-0.8B model never emits its real EOS within 40 tokens, so natural-EOS
generation cannot be observed. Instead this drives the EOS code path deterministically:
the scheduler's ``eos_token_id`` is overridden to a token the model DOES emit at a known
verify-committed position, so the round that commits that token must stop the stream
there (EOS) -- or, with ignore_eos, keep generating past it.

One scheduler per OS process (Engine asserts torch.cuda is not already initialized), so
each config runs separately and the results are compared by a driver.

Usage:
  python tests/mtp/test_mtp_eos.py --mtp 1 --max-tokens 8 \
      --eos-token 220 [--ignore-eos] --out /tmp/eos_mtp.json
  python tests/mtp/test_mtp_eos.py --mtp 0 --max-tokens 8 \
      --eos-token 220 [--ignore-eos] --out /tmp/eos_nonmtp.json
"""
import argparse
import json
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("CUDA_HOME", "/home/daking/.conda/envs/daking")
_REPO = "/home/daking/PROJECT/pico-sglang"
sys.path.insert(0, os.path.join(_REPO, "python"))
sys.path.insert(0, _REPO)

import torch

from picosgl.core import SamplingParams
from picosgl.message import UserMsg
from test_mtp_e2e import OfflineScheduler, make_config
from transformers import AutoTokenizer

MODEL = os.environ.get("QWEN35_MODEL", "/home/daking/models/huggingface/Qwen3.5-0.8B")
PROMPT = "1+1="


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtp", type=int, choices=[0, 1], required=True)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--eos-token", type=int, default=None)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids[0].to(torch.int32)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
    )
    msgs = [UserMsg(uid=0, input_ids=ids, sampling_params=sampling)]
    sched = OfflineScheduler(make_config(bool(args.mtp)), msgs)
    if args.eos_token is not None:
        sched.eos_token_id = args.eos_token  # override: deterministic EOS trigger
        # the AR manager caches eos_token_id at construction (dependency injection), so
        # the override must reach the manager the process() path actually reads.
        sched.verify_manager.eos_token_id = args.eos_token
        sched.decode_manager.eos_token_id = args.eos_token

    data = None
    idle = 0
    for _ in range(5000):
        if data is None and not (
            sched.prefill_manager.runnable
            or sched.decode_manager.runnable
            or (sched.verify_manager is not None and sched.verify_manager.runnable)
        ):
            idle += 1
            if idle >= 2:
                break
        else:
            idle = 0
        data = sched.overlap_loop(data)

    stream = []
    fin_at = []
    pos = 0
    for r in sched.results:
        toks = r.next_token if isinstance(r.next_token, list) else [r.next_token]
        stream.extend(toks)
        pos += len(toks)
        if r.finished:
            fin_at.append(pos - 1)  # token index where the stream ended
    real_eos = tokenizer.eos_token_id
    eos_used = args.eos_token if args.eos_token is not None else real_eos
    sched.shutdown()

    result = {
        "mtp": bool(args.mtp),
        "max_tokens": args.max_tokens,
        "eos_used": eos_used,
        "ignore_eos": args.ignore_eos,
        "n_emitted": len(stream),
        "stream": stream,
        "finished_at": fin_at,
        "stopped_at_eos": eos_used in stream and len(stream) == stream.index(eos_used) + 1,
    }
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
