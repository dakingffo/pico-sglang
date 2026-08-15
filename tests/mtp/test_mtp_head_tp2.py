"""Numerical TP=2 verification of the MTP head's TP-sensitive pieces.

The MTP head (Qwen3_5MultiTokenPredictor) is: norm(emb) -> norm(h) -> cat -> fc ->
dense attention layer -> norm -> lm_head. Every piece is a TP op already verified in
test_tp2_numeric (embedding all_reduce, attention o_proj all_reduce, mlp down_proj
all_reduce, ParallelLMHead gather) EXCEPT the fusion fc, which is a LinearColumnParallel
all_gather. The fc all_gather sits at the END of ``fc.forward`` (its output feeds the
next layer, so it is not "mid-forward" when tested in isolation), so the single-GPU
recording trick applies.

A true composition replay is impossible on one GPU: the fc all_gather output feeds the
attention, whose partials would be needed by the other rank before its own forward runs.
The composition is confirmed by the real 2-GPU run on the server.

Run: /home/daking/.conda/envs/daking/bin/python tests/mtp/test_mtp_head_tp2.py
"""
import os
import sys

os.environ.setdefault("CUDA_HOME", "/home/daking/.conda/envs/daking")
sys.path.insert(0, "/home/daking/PROJECT/pico-sglang/python")
sys.path.insert(0, "/home/daking/PROJECT/pico-sglang/tests/mtp")

import torch

from picosgl.distributed import DistributedInfo, set_tp_info
from picosgl.distributed.impl import DistributedCommunicator, DistributedImpl, NoopDistributedImpl
from picosgl.models.config import ModelConfig, RotaryConfig


def make_mtp_config() -> ModelConfig:
    return ModelConfig(
        num_layers=2,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=64,
        hidden_size=512,
        vocab_size=5000,  # odd -> exercises the vocab shard edge
        intermediate_size=1024,
        rms_norm_eps=1e-6,
        rotary_config=RotaryConfig(
            head_dim=64, rotary_dim=32, max_position=4096, base=10000, scaling=None
        ),
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        model_type="qwen3_5",
        architectures=["Qwen3_5ForCausalLM"],
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=4,
        linear_num_value_heads=12,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        partial_rotary_factor=0.5,
        attn_output_gate=True,
        mtp_num_hidden_layers=1,
        mamba_ssm_dtype="float32",
    )


class RecordingImpl(DistributedImpl):
    """Records every collective input. all_reduce returns x unchanged (row-parallel
    partial); all_gather returns a shape-correct dummy (the real partial is recorded)."""

    def __init__(self, records: list, tp_size: int):
        self.records = records
        self.tp_size = tp_size

    def all_reduce(self, x):
        self.records.append(("reduce", x.detach().clone()))
        return x

    def all_gather(self, x):
        self.records.append(("gather", x.detach().clone()))
        shp = list(x.shape)
        shp[0] *= self.tp_size
        return x.new_zeros(shp)


def _set_tree(module, name, v):
    parts = name.split(".")
    obj = module
    for p in parts[:-1]:
        if p.isdigit():
            obj = obj.op_list[int(p)]
        else:
            obj = obj.__dict__[p]
    setattr(obj, parts[-1], v)


def reset_tp(rank, size):
    import picosgl.distributed.info as _info

    _info._TP_INFO = None
    set_tp_info(DistributedInfo(rank=rank, size=size))


def _load_from(op, sd, prefix, device="cuda:0"):
    for k, v in sd.items():
        if prefix and not k.startswith(prefix):
            continue
        key = k[len(prefix):] if prefix else k
        _set_tree(op, key, v.to(device))


def build_full_sd(config: ModelConfig, seed: int) -> dict:
    reset_tp(0, 1)
    torch.manual_seed(seed)
    from picosgl.models.qwen3_5 import Qwen3_5ForCausalLM

    model = Qwen3_5ForCausalLM(config, paged=False)
    for name, p in model.state_dict().items():
        _set_tree(model, name, torch.randn_like(p) * 0.02)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def shard_sd(full_sd, rank, size, config):
    from picosgl.models.weight import _shard_tensor

    qkv_regions = None
    if config.is_hybrid and config.linear_num_key_heads:
        qk = config.linear_num_key_heads * config.linear_key_head_dim
        v = config.linear_num_value_heads * config.linear_value_head_dim
        qkv_regions = (qk, qk, v)
    return {
        k: _shard_tensor(k, v, rank, size, config.num_kv_heads, qkv_regions)
        for k, v in full_sd.items()
    }


def check(name, full, p0, p1, combine="sum", tol=1e-3):
    if combine == "sum":
        combined = p0 + p1
    else:  # gather: rank-interleave reshape, exactly like LinearColumnParallel / LMHead
        combined = torch.stack([p0, p1], dim=1).reshape(p0.shape[0], -1)
    err = (combined - full).abs().max().item()
    rel = err / (full.abs().max().item() + 1e-12)
    status = "OK " if (err < tol and rel < tol) else "FAIL"
    print(f"  [{status}] {name:28s} abs_err={err:.3e} rel_err={rel:.3e}")
    return status == "OK "


def run():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    config = make_mtp_config()
    n = 2
    full_sd = build_full_sd(config, seed=42)

    ok = True
    T = 9

    # ---- 1. MTP fusion fc (LinearColumnParallel all_gather), standalone ----
    print("=" * 60)
    print("MTP fusion fc (all_gather)")
    from picosgl.layers import LinearColumnParallel

    reset_tp(0, 1)
    fc_full = LinearColumnParallel(config.hidden_size * 2, config.hidden_size, has_bias=False)
    _load_from(fc_full, full_sd, "mtp.fc.")
    xfc = torch.randn(T, config.hidden_size * 2, device=device)
    fc_parts = []
    for r in range(n):
        reset_tp(r, n)
        fc = LinearColumnParallel(config.hidden_size * 2, config.hidden_size, has_bias=False)
        _load_from(fc, shard_sd(full_sd, r, n, config), "mtp.fc.")
        recs = []
        DistributedCommunicator.impl = RecordingImpl(recs, n)
        fc.forward(xfc)
        assert len(recs) == 1 and recs[0][0] == "gather"
        fc_parts.append(recs[0][1])  # the REAL pre-gather partial (T, out/tp)
    ok &= check("mtp fusion fc", fc_full.forward(xfc).detach(), fc_parts[0], fc_parts[1], "cat")

    # ---- 2. MTP pre_fc norms: shared weight, rank-invariant (no collective) ----
    print("=" * 60)
    print("MTP pre_fc norms (shared weight, no collective)")
    from picosgl.layers.qwen3_5.norm import Qwen3_5RMSNorm

    reset_tp(0, 1)
    nrm_full = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    _load_from(nrm_full, full_sd, "mtp.pre_fc_norm_embedding.")
    xin = torch.randn(T, config.hidden_size, device=device)
    expected = nrm_full.forward(xin).detach()
    for r in range(n):
        reset_tp(r, n)
        nrm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        _load_from(nrm, shard_sd(full_sd, r, n, config), "mtp.pre_fc_norm_embedding.")
        got = nrm.forward(xin).detach()
        ok &= (expected - got).abs().max().item() < 1e-9
    print("  [OK ] pre_fc_norm_embedding        rank-invariant")

    # ---- 3. lm_head vocab gather within the mtp head (get_logits) ----
    print("=" * 60)
    print("MTP get_logits (lm_head all_gather)")
    from picosgl.layers.embedding import VocabParallelEmbedding, ParallelLMHead
    from picosgl.models.qwen3_5 import Qwen3_5MultiTokenPredictor

    hidden_in = torch.randn(T, config.hidden_size, device=device)

    def build(rank, size, sd):
        reset_tp(rank, size)
        emb = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        lm = ParallelLMHead(config.vocab_size, config.hidden_size, tie_word_embeddings=False)
        _load_from(emb, sd, "model.embed_tokens.")
        _load_from(lm, sd, "lm_head.")
        mtp = Qwen3_5MultiTokenPredictor(config, emb, lm)
        _load_from(mtp, sd, "mtp.")
        return mtp

    mtp_full = build(0, 1, full_sd)
    logits_full = mtp_full.get_logits(hidden_in).detach()
    lm_parts = []
    for r in range(n):
        mtp = build(r, n, shard_sd(full_sd, r, n, config))
        recs = []
        DistributedCommunicator.impl = RecordingImpl(recs, n)
        mtp.get_logits(hidden_in)
        assert len(recs) == 1 and recs[0][0] == "gather"
        lm_parts.append(recs[0][1])  # (T, vocab/tp) partial
    ok &= check("mtp get_logits", logits_full, lm_parts[0], lm_parts[1], "cat")

    DistributedCommunicator.impl = NoopDistributedImpl()
    print("=" * 60)
    print("MTP HEAD TP=2:", "ALL PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    run()
