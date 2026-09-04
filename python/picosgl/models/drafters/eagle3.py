from __future__ import annotations

import glob
import os

import safetensors
import torch

from picosgl.layers import (
    BaseOP,
    GatedMLP,
    LinearColParallelPartitioned,
    LinearReplicated,
    RMSNorm,
    RMSNormFused,
    RotaryAttention,
    VocabParallelEmbedding,
)

from .base import BaseDrafterModel


class Eagle3DecoderLayer(BaseOP):
    def __init__(
        self,
        hidden_size     : int,
        head_dim        : int,
        num_qo_heads    : int,
        num_kv_heads    : int,
        intermediate_size: int,
        hidden_act      : str,
        rms_norm_eps    : float,
        max_position    : int,
        rope_base       : float,
    ) -> None:
        self.self_attn = RotaryAttention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            layer_id=0,
            rotary_dim=head_dim,
            max_position=max_position,
            rope_base=rope_base,
            rope_scaling=None,
        )
        self.self_attn.qkv_proj = LinearColParallelPartitioned(
            input_size=hidden_size * 2,
            partition_size=head_dim,
            partitions=[(num_qo_heads, False)] + [(num_kv_heads, True)] * 2,
            has_bias=False,
        )
        self.mlp = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
        )
        self.hidden_norm = RMSNorm(hidden_size, rms_norm_eps)
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(hidden_size, rms_norm_eps)


class Eagle3Drafter(BaseDrafterModel):
    def __init__(
        self,
        *,
        hidden_size      : int,
        head_dim         : int,
        num_qo_heads     : int,
        num_kv_heads     : int,
        intermediate_size: int,
        hidden_act       : str,
        rms_norm_eps     : float,
        max_position     : int,
        rope_base        : float,
        vocab_size       : int,
        draft_vocab_size : int,
    ) -> None:
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.draft_vocab_size = draft_vocab_size
        self.midlayer = Eagle3DecoderLayer(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            rms_norm_eps=rms_norm_eps,
            max_position=max_position,
            rope_base=rope_base,
        )
        self.norm = RMSNormFused(hidden_size, rms_norm_eps)
        self.fc = LinearReplicated(hidden_size * 3, hidden_size, has_bias=False)
        self.lm_head = LinearReplicated(hidden_size, draft_vocab_size, has_bias=False)
        self._embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)
        self._hot_token_id: torch.Tensor | None = None

    @property
    def hot_token_id(self) -> torch.Tensor:
        assert self._hot_token_id is not None
        return self._hot_token_id

    def load_weights(self, model_path: str, device: torch.device) -> None:
        checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        state_dict: dict[str, torch.Tensor] = {}
        merge_groups = {
            "midlayer.self_attn.qkv_proj.weight": (
                "midlayer.self_attn.q_proj.weight",
                "midlayer.self_attn.k_proj.weight",
                "midlayer.self_attn.v_proj.weight",
            ),
            "midlayer.mlp.gate_up_proj.weight": (
                "midlayer.mlp.gate_proj.weight",
                "midlayer.mlp.up_proj.weight",
            ),
        }
        consumed = {name for names in merge_groups.values() for name in names}
        for target_name, source_names in merge_groups.items():
            state_dict[target_name] = torch.cat(
                [checkpoint[name] for name in source_names], dim=0
            ).to(device)
        for name, tensor in checkpoint.items():
            if name in consumed or name in ("d2t", "t2d"):
                continue
            state_dict[name] = tensor.to(device)

        d2t = checkpoint["d2t"]
        self._hot_token_id = (
            d2t + torch.arange(d2t.shape[0], dtype=d2t.dtype)
        ).to(device)
        del checkpoint
        self.load_state_dict(state_dict)

    def load_target_embedding(
        self,
        model_path: str,
        device    : torch.device,
    ) -> None:
        key = "model.embed_tokens.weight"
        for path in glob.glob(os.path.join(model_path, "*.safetensors")):
            with safetensors.safe_open(path, framework="pt", device="cpu") as file:
                if key in file.keys():
                    weight = file.get_tensor(key)
                    assert weight.shape == self._embed_tokens.weight.shape
                    self._embed_tokens.weight = weight.to(
                        device=device,
                        dtype=self._embed_tokens.weight.dtype,
                    )
                    return
        raise KeyError(f"Target checkpoint does not contain {key}")

    def prepare_cache_rows(
        self,
        input_ids    : torch.Tensor,
        positions    : torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.shape[-1] != self.hidden_size:
            assert hidden_states.shape[-1] == self.hidden_size * 3
            hidden_states = self.fc.forward(hidden_states)

        residual = hidden_states
        hidden_states = self.midlayer.hidden_norm.forward(hidden_states)
        embeddings = self.midlayer.input_layernorm.forward(
            self._embed_tokens.forward(input_ids)
        )
        hidden_states = torch.cat([embeddings, hidden_states], dim=-1)
        query, key, value = self.midlayer.self_attn.proj(hidden_states, positions)
        return residual, query, key, value

    def finish_cache_rows(
        self,
        residual        : torch.Tensor,
        attention_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.midlayer.self_attn.o_proj.forward(
            attention_output.flatten(-2)
        )
        hidden_states, residual = self.midlayer.post_attention_layernorm.forward(
            hidden_states, residual
        )
        hidden_states = self.midlayer.mlp.forward(hidden_states)
        logits_hidden, recurrent_hidden = self.norm.forward(hidden_states, residual)
        return self.lm_head.forward(logits_hidden), recurrent_hidden


__all__ = ["Eagle3Drafter"]
