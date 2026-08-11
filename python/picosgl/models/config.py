from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]
    # ---- Qwen3.5 hybrid (linear_attention + full_attention) ----
    layer_types: list[str] | None = None
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_conv_kernel_dim: int = 0
    partial_rotary_factor: float = 1.0
    attn_output_gate: bool = False
    # Qwen3.5 attention output gate activation: "sigmoid" (default) or "swish"
    # (Qwen3.6, gate * sigmoid(gate)). Default preserves Qwen3.5 byte-identity.
    output_gate_type: str = "sigmoid"
    mtp_num_hidden_layers: int = 0
    mamba_ssm_dtype: str = "float32"

    @property
    def is_moe(self) -> bool:
        return "moe" in self.model_type

    @property
    def is_hybrid(self) -> bool:
        return bool(self.layer_types)

    @property
    def num_attention_layers(self) -> int:
        """Number of full-attention layers (the only ones that use paged KV cache)."""
        if self.is_hybrid:
            return sum(1 for t in self.layer_types if t == "full_attention")
        return self.num_layers

    @property
    def num_linear_layers(self) -> int:
        if self.is_hybrid:
            return sum(1 for t in self.layer_types if t == "linear_attention")
        return 0

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        if hasattr(config, "text_config") and config.text_config is not None:
            top = config
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "llama")
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])

        # Llama/Qwen: rope_theta is a direct attr; Mistral: it's inside rope_scaling dict;
        # Qwen3.5: inside rope_parameters dict.
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_params = getattr(config, "rope_parameters", None) or {}
        rope_theta = (
            getattr(config, "rope_theta", None)
            or rope_params.get("rope_theta")
            or (rope_scaling["rope_theta"] if rope_scaling else None)
        )

        # ---- Qwen3.5 hybrid ----
        # transformers populates ``layer_types`` for ALL Qwen3 configs (dense ones get an
        # all-"full_attention" array), so "non-empty" is NOT hybrid. Only a real mix of
        # linear_attention layers makes the model hybrid.
        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None and "linear_attention" not in layer_types:
            layer_types = None
        if layer_types is None:
            interval = getattr(config, "full_attention_interval", None)
            if interval is not None:
                layer_types = [
                    "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                    for i in range(config.num_hidden_layers)
                ]
        partial_rotary_factor = float(
            rope_params.get(
                "partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0)
            )
        )

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=int(head_dim * partial_rotary_factor),
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
            layer_types=layer_types,
            linear_num_key_heads=getattr(config, "linear_num_key_heads", 0),
            linear_num_value_heads=getattr(config, "linear_num_value_heads", 0),
            linear_key_head_dim=getattr(config, "linear_key_head_dim", 0),
            linear_value_head_dim=getattr(config, "linear_value_head_dim", 0),
            linear_conv_kernel_dim=getattr(config, "linear_conv_kernel_dim", 0),
            partial_rotary_factor=partial_rotary_factor,
            attn_output_gate=getattr(config, "attn_output_gate", False),
            output_gate_type=getattr(config, "output_gate_type", "sigmoid"),
            mtp_num_hidden_layers=getattr(config, "mtp_num_hidden_layers", 0),
            mamba_ssm_dtype=getattr(config, "mamba_ssm_dtype", "float32"),
        )