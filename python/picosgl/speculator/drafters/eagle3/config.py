from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from picosgl.utils import load_model_config

from ...base import BaseSpeculatorConfig, SpeculatorReserve

if TYPE_CHECKING:
    from picosgl.engine.config import EngineConfig


@dataclass(frozen=True)
class Eagle3SpeculatorConfig(BaseSpeculatorConfig):
    algorithm: ClassVar[str] = "EAGLE3"

    num_draft_tokens: int = 3
    window_size     : int = 128
    target_layer_ids: tuple[int, int, int] = (2, 14, 25)

    @property
    def hidden_size_multiplier(self) -> int:
        return len(self.target_layer_ids)

    @property
    def max_init_hidden_rows(self) -> int:
        return self.window_size

    def make_reserve(self, max_running_req: int) -> SpeculatorReserve:
        return SpeculatorReserve()

    def validate(self, config: EngineConfig) -> None:
        from picosgl.models.qwen3 import Qwen3Config

        assert isinstance(config.model_config, Qwen3Config), (
            "EAGLE3 adaptation currently supports a dense Qwen3 target"
        )
        assert config.speculative_draft_model_path is not None
        draft_config = load_model_config(config.speculative_draft_model_path)
        assert getattr(draft_config, "architectures", None) == [
            "LlamaForCausalLMEagle3"
        ], "Expected an AngelSlim LlamaForCausalLMEagle3 checkpoint"
        assert draft_config.hidden_size == config.model_config.hidden_size
        assert draft_config.vocab_size == config.model_config.vocab_size
        assert draft_config.num_hidden_layers == 1
        assert len(set(self.target_layer_ids)) == 3
        assert min(self.target_layer_ids) >= 0
        assert max(self.target_layer_ids) < config.model_config.num_layers
        assert self.num_draft_tokens >= 1
        assert config.decode_batch_budget >= self.num_draft_tokens


__all__ = ["Eagle3SpeculatorConfig"]
