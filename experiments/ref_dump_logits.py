"""Gold-standard reference logits: transformers v5 Qwen3_5ForCausalLM on the 2B text weights.

The checkpoint is multimodal (Qwen3_5ForConditionalGeneration) with text weights under
``model.language_model.*``. We build the pure-text ``Qwen3_5ForCausalLM`` and load those keys
renamed to ``model.*`` (dropping ``model.visual.*`` and ``mtp.*``). A text-only prefill of the
same prompt that pico-sglang uses gives per-position logits that pico must match.

Run with the transformers-v5 venv: /tmp/refq35/bin/python experiments/ref_dump_logits.py
"""
import glob
import json
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

MODEL_PATH = "/home/daking/models/huggingface/Qwen3.5-2B"
TEXT = "The quick brown fox jumps over the lazy dog while the sun sets over the mountain lake."
OUT = "/tmp/ref_logits.pt"


def main():
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    # Pure-text config (unwrap the multimodal wrapper's text_config).
    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    text_cfg = cfg.text_config
    print(f"text config: model_type={text_cfg.model_type} "
          f"layers={text_cfg.num_hidden_layers} "
          f"rotary_dim={int((text_cfg.head_dim) * text_cfg.rope_parameters.get('partial_rotary_factor', 1.0))} "
          f"rope_theta={text_cfg.rope_parameters.get('rope_theta')}")

    # Every text weight is loaded from the checkpoint, so random init is wasted work
    # (the 508M-element embedding's CPU normal_ is the dominant cost). Skip it, mirroring
    # from_pretrained(low_cpu_mem_usage=True).
    def _noop_reset(self, *a, **k):
        return None

    torch.nn.Embedding.reset_parameters = _noop_reset
    torch.nn.Linear.reset_parameters = _noop_reset

    model = Qwen3_5ForCausalLM(text_cfg).to(dtype=torch.bfloat16, device=device)
    model.eval()

    # Load text weights: model.language_model.* -> model.*
    index = json.load(open(os.path.join(MODEL_PATH, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    sd = {}
    for shard in shards:
        path = os.path.join(MODEL_PATH, shard)
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("model.language_model."):
                    sd[k.replace("model.language_model.", "model.", 1)] = f.get_tensor(k)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # lm_head.weight is tied to model.embed_tokens.weight -> legitimately "missing" from sd
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert set(missing) <= {"lm_head.weight"}, f"unexpected missing: {missing}"
    print(f"loaded {len(sd)} text tensors; missing={missing} (tied lm_head ok)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    ids = tokenizer(TEXT, return_tensors="pt").input_ids.to(device)
    S = ids.shape[1]

    with torch.no_grad():
        out = model(input_ids=ids, logits_to_keep=S)
    logits = out.logits[0].float().cpu()  # (S, vocab)

    torch.save({"ids": ids.cpu(), "text": TEXT, "logits": logits}, OUT)
    print(f"saved {OUT}: logits {tuple(logits.shape)} S={S}")


if __name__ == "__main__":
    main()
