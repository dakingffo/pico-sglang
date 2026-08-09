"""Dump pico-sglang Qwen3.5-2B prefill logits (all positions) for comparison with the
transformers-v5 reference (ref_dump_logits.py).

Run with the main env: /home/daking/.conda/envs/daking/bin/python experiments/pico_dump_logits.py
"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("CUDA_HOME", "/home/daking/.conda/envs/daking")
sys.path.insert(0, "/home/daking/PROJECT/pico-sglang/python")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

import verify_qwen35 as vq

TEXT = "The quick brown fox jumps over the lazy dog while the sun sets over the mountain lake."
OUT = "/tmp/pico_logits.pt"


def main():
    mcfg = vq.setup()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"pico rotary: rotary_dim={mcfg.rotary_config.rotary_dim} base={mcfg.rotary_config.base}")

    model, loaded = vq.load_model(mcfg, paged=True)
    vq.to_device(model, device)  # in-place move, avoids the 2x-model double-copy peak
    del loaded

    from picosgl.core import Batch, Context, Request, clear_global_ctx, set_global_ctx
    from picosgl.cache import create_kvcache_pool, LinearStatePool
    from picosgl.layers.attention_backend import create_attention_backend

    conv_dim = (
        mcfg.linear_num_key_heads * mcfg.linear_key_head_dim * 2
        + mcfg.linear_num_value_heads * mcfg.linear_value_head_dim
    )
    ctx = Context(page_size=1)
    num_pages = 2048
    ctx.kv_cache = create_kvcache_pool(
        model_config=mcfg, num_pages=num_pages, page_size=1,
        dtype=torch.bfloat16, device=device,
    )
    ctx.linear_state = LinearStatePool(
        num_linear_layers=mcfg.num_linear_layers, max_req=4, conv_dim=conv_dim,
        kernel_size=mcfg.linear_conv_kernel_dim,
        num_v_heads=mcfg.linear_num_value_heads,
        head_k_dim=mcfg.linear_key_head_dim,
        head_v_dim=mcfg.linear_value_head_dim,
        device=device, dtype=torch.bfloat16,
    )
    ctx.page_table = torch.zeros((8, 8192), dtype=torch.int32, device=device)
    base = torch.arange(num_pages, device=device, dtype=torch.int32).view(1, -1)
    ctx.page_table[:, :num_pages] = base
    set_global_ctx(ctx)  # create_attention_backend reads ctx.kv_cache via get_global_ctx()
    ctx.attn_backend = create_attention_backend("fi", mcfg)

    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(vq.MODEL_PATH)
    ids = tokenizer(TEXT, return_tensors="pt").input_ids[0].to(device)  # (S,)
    S = ids.shape[0]

    req = Request(
        input_ids=ids.cpu(), table_idx=0, cached_len=0, output_len=1,
        uid=0, sampling_params=None,  # type: ignore
        cache_handle=None,  # type: ignore
    )
    batch = Batch(reqs=[req], phase="prefill")
    batch.input_ids = ids
    batch.positions = torch.arange(S, device=device)
    batch.padded_reqs = [req]
    batch.out_loc = ctx.page_table[0, :S]
    ctx.attn_backend.prepare_metadata(batch)

    with ctx.forward_batch(batch):
        hidden = model.model.forward(ids)  # (S, hidden)
        lmw = model.lm_head.tied_embedding.weight
        logits = F.linear(hidden, lmw)  # (S, vocab), bypass prefill last-only gather

    torch.save({"ids": ids.cpu(), "text": TEXT, "logits": logits.detach().float().cpu()}, OUT)
    print(f"saved {OUT}: logits {tuple(logits.shape)} S={S}")

    clear_global_ctx()


if __name__ == "__main__":
    main()
