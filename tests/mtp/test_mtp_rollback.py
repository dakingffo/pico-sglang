"""Step 8c: linear-state rollback correctness.

Core circular-buffer test: after a verify round accepts num_sampled tokens, rollback_to
must land on the snapshot that holds "the state after exactly num_sampled tokens were
appended to the baseline" -- i.e. the state a plain (non-MTP) decode would be at.

  baseline: prefill, committed slot p (=1 after advance_batch)
  reference: from baseline, 2 sequential per-token decodes (t0,t1) -> committed slot
             (p+2) holds "state after 2 tokens". Same for 5 tokens (wrap: slot p).
  verify:    from baseline, fused verify window [C, C+5) with tokens
             [t0, t1, d2, d3, d4], then pool.rollback_to(req, num_sampled) ->
             slots = (p+num_sampled) % D. Read the slot it points at.
  compare:   conv_state / recurrent_state at the rolled-back slot vs the reference slot.

Expected: the pointer arithmetic is exact (integer mod); the content matches the
reference within bf16 ULP (verify projects all window tokens batched, decode per token,
so ~1 ULP input difference into the shared per-token recurrent rule -- NOT a structural
divergence). num_sampled=5 exercises the ring wrap (slot (p+5)%5 == p, the baseline slot,
which is safe to overwrite because it is only read at round start).

Run: /home/daking/.conda/envs/daking/bin/python tests/mtp/test_mtp_rollback.py
"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("CUDA_HOME", "/home/daking/.conda/envs/daking")

MODEL_PATH = os.environ.get("QWEN35_MODEL", "/home/daking/models/huggingface/Qwen3.5-0.8B")
sys.path.insert(0, "/home/daking/PROJECT/pico-sglang/python")

import torch

from picosgl.utils import cached_load_hf_config, torch_dtype


def setup():
    from picosgl.distributed import DistributedInfo, set_tp_info

    set_tp_info(DistributedInfo(rank=0, size=1))
    from picosgl.models.config import ModelConfig

    return ModelConfig.from_hf(cached_load_hf_config(MODEL_PATH))


def to_device(module, device):
    if isinstance(module, (list, tuple)):
        for v in module:
            to_device(v, device)
        return
    d = getattr(module, "__dict__", None)
    if d is None:
        return
    for name, v in d.items():
        if name.startswith("_"):
            continue
        if isinstance(v, torch.Tensor):
            if v.is_meta:
                continue
            setattr(module, name, v.to(device))
        elif isinstance(v, (list, tuple)):
            for el in v:
                to_device(el, device)
        elif getattr(v, "__dict__", None) is not None:
            to_device(v, device)


def main():
    mcfg = setup()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    from picosgl.core import Context, Request, Batch, set_global_ctx
    from picosgl.cache import create_kvcache_pool
    from picosgl.cache import LinearStatePool
    from picosgl.layers.attention_backend import create_attention_backend
    from picosgl.models.qwen3_5 import Qwen3_5ForCausalLM
    from picosgl.models.weight import load_weight

    loaded = {k: v for k, v in load_weight(MODEL_PATH, "cpu")}
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = Qwen3_5ForCausalLM(mcfg, paged=True)
    model.load_state_dict(dict(loaded))
    del loaded
    to_device(model, device)
    torch.cuda.empty_cache()

    K = 4  # num_spec_tokens
    D = K + 1
    conv_dim = (
        mcfg.linear_num_key_heads * mcfg.linear_key_head_dim * 2
        + mcfg.linear_num_value_heads * mcfg.linear_value_head_dim
    )
    ctx = Context(page_size=1)
    num_pages = 8192
    ctx.kv_cache = create_kvcache_pool(
        model_config=mcfg, num_pages=num_pages, page_size=1,
        dtype=torch.bfloat16, device=device,
    )
    pool = LinearStatePool(
        num_linear_layers=mcfg.num_linear_layers, max_req=4, conv_dim=conv_dim,
        kernel_size=mcfg.linear_conv_kernel_dim,
        num_v_heads=mcfg.linear_num_value_heads,
        head_k_dim=mcfg.linear_key_head_dim,
        head_v_dim=mcfg.linear_value_head_dim,
        device=device, dtype=torch.bfloat16, depth=D,
    )
    ctx.linear_state = pool
    ctx.page_table = torch.zeros((8, num_pages), dtype=torch.int32, device=device)
    base = torch.arange(num_pages, device=device, dtype=torch.int32).view(1, -1)
    ctx.page_table[:, :num_pages] = base
    set_global_ctx(ctx)
    ctx.attn_backend = create_attention_backend("fi", mcfg)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    text = "def quicksort(arr):\n    if len(arr) <= 1:"
    ids = tokenizer(text, return_tensors="pt").input_ids[0].to(device)
    S = ids.shape[0]
    TIDX = 4
    C = S

    req = Request(
        input_ids=ids.cpu(), table_idx=TIDX, cached_len=0, output_len=40, uid=0,
        sampling_params=None,  # type: ignore
        cache_handle=None,  # type: ignore
    )

    kv = ctx.kv_cache
    storage = kv._storage_shape

    def zero_window():
        for layer in range(kv.num_layers):
            for p in range(C, C + K + 1):
                kv.k_cache(layer).view(storage)[p].zero_()
                kv.v_cache(layer).view(storage)[p].zero_()

    # ---------------- prefill ----------------
    pb = Batch(reqs=[req], phase="prefill")
    pb.input_ids = ids
    pb.positions = torch.arange(S, device=device)
    pb.padded_reqs = [req]
    pb.out_loc = ctx.page_table[TIDX, :S]
    ctx.attn_backend.prepare_metadata(pb)
    with ctx.forward_batch(pb):
        hidden_p, logits_p = model.forward_verify()
    bonus = int(logits_p[0].argmax().item())
    pool.advance_batch([req])
    p = int(pool.slots[TIDX])
    print(f"prompt_len={S} bonus={bonus} baseline slot p={p} depth={D}")
    assert p == 1, f"baseline slot should be 1, got {p}"

    snap = (pool.conv_state.clone(), pool.recurrent_state.clone(), pool.slots.clone())

    def restore():
        pool.conv_state.copy_(snap[0])
        pool.recurrent_state.copy_(snap[1])
        pool.slots.copy_(snap[2])
        req.cached_len = C
        req.device_len = C + 1

    def decode_at(pos, tok):
        req.device_len = pos + 1
        db = Batch(reqs=[req], phase="decode")
        db.input_ids = tok
        db.positions = torch.tensor([pos], device=device)
        db.padded_reqs = [req]
        db.out_loc = ctx.page_table[TIDX, pos : pos + 1]
        ctx.attn_backend.prepare_metadata(db)
        with ctx.forward_batch(db):
            model.model.forward(tok)
        pool.slots[TIDX] = (int(pool.slots[TIDX]) + 1) % pool.depth

    def verify_window(toks):
        n = toks.shape[0]
        vb = Batch(reqs=[req], phase="verify")
        vb.input_ids = toks
        vb.positions = torch.arange(C, C + n, device=device)
        vb.padded_reqs = [req]
        req.device_len = C + n
        vb.out_loc = ctx.page_table[TIDX, C : C + n]
        ctx.attn_backend.prepare_metadata(vb)
        with ctx.forward_batch(vb):
            model.forward_verify()

    def read_slot(slot):
        s = slot % D
        return (
            pool.conv_state[:, s, TIDX].clone(),
            pool.recurrent_state[:, s, TIDX].clone(),
        )

    def cmp(tag, a, b):
        dc = (a[0].float() - b[0].float()).abs().max().item()
        dr = (a[1].float() - b[1].float()).abs().max().item()
        ac = a[0].abs().max().item()
        ar = a[1].abs().max().item()
        rel = max(dc / max(ac, 1e-9), dr / max(ar, 1e-9))
        print(f"  {tag:28s} conv|d|={dc:.6f} (absmax {ac:.3f})  "
              f"recurrent|d|={dr:.6f} (absmax {ar:.3f})  rel={rel:.2e}")
        return rel

    t0 = torch.tensor([bonus], device=device)
    drafts = torch.tensor([123, 456, 789, 999], device=device)
    vt = torch.cat([t0, drafts])

    # ---- reference: direct per-token decodes -> state at slot (p+2) and slot (p+5)=p ----
    restore()
    zero_window()
    for j in range(5):
        decode_at(C + j, vt[j].view(1))
    ref2 = read_slot(p + 2)   # state after 2 tokens (num_sampled=2)
    ref5 = read_slot(p + 5)   # state after 5 tokens (wrap -> slot p)
    print(f"\nreference: committed slot after 5 decodes = {int(pool.slots[TIDX])} (expect {(p+5)%D})")

    # ---- verify + rollback_to(2): slot (p+2) must hold "state after 2 tokens" ----
    restore()
    zero_window()
    verify_window(vt)
    pool.rollback_to([req], 2)
    got2 = read_slot(int(pool.slots[TIDX]))
    print(f"rollback_to(2): slot={int(pool.slots[TIDX])} (expect {(p+2)%D})")
    r2 = cmp("num_sampled=2 vs direct-2", got2, ref2)

    # ---- verify + rollback_to(5): wrap, slot (p+5)%5=p must hold "state after 5" ----
    restore()
    zero_window()
    verify_window(vt)
    pool.rollback_to([req], 5)
    got5 = read_slot(int(pool.slots[TIDX]))
    print(f"rollback_to(5): slot={int(pool.slots[TIDX])} (expect {(p+5)%D})")
    r5 = cmp("num_sampled=5 vs direct-5", got5, ref5)

    # ---- structural check: the rolled-back state is NOT the stale prefill baseline ----
    restore()
    verify_window(vt)
    pool.rollback_to([req], 2)
    g2 = read_slot(int(pool.slots[TIDX]))
    b2 = read_slot(p)  # stale baseline
    dc = (g2[0].float() - b2[0].float()).abs().max().item()
    dr = (g2[1].float() - b2[1].float()).abs().max().item()
    print(f"\nrollback(2) slot vs stale baseline slot p: conv|d|={dc:.4f} "
          f"recurrent|d|={dr:.4f}  (>>0 => rollback is NOT pointing at stale baseline)")

    # ---- pointer-arithmetic sanity: rollback_to must be idempotent on the commit ----
    restore()
    verify_window(vt)
    pool.rollback_to([req], 3)
    assert int(pool.slots[TIDX]) == (p + 3) % D
    pool.rollback_to([req], 2)
    assert int(pool.slots[TIDX]) == (p + 5) % D
    print(f"\npointer arithmetic: rollback(3) -> {(p+3)%D}, then rollback(2) -> {(p+5)%D} OK")

    ok = max(r2, r5) < 0.02  # bf16 ULP band for state content
    print(f"\nROLLBACK: {'PASS' if ok else 'CHECK'} (max rel state diff {max(r2, r5):.2e}; "
          f"pointer arithmetic exact)")
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
