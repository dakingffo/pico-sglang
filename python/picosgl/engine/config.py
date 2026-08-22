from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import torch

from picosgl.distributed import DistributedInfo
from picosgl.utils import load_model_config
from picosgl.models import ModelConfig

if TYPE_CHECKING:
    from picosgl.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path          : str
    tp_info             : DistributedInfo
    dtype               : torch.dtype
    max_running_req     : int              = 128
    attention_backend   : str              = "auto"
    moe_backend         : str              = "auto"
    cuda_graph_bs       : list[int] | None = None
    cuda_graph_max_bs   : int | None       = None
    page_size           : int              = 128
    memory_ratio        : float            = 0.85
    distributed_timeout : float            = 60.0
    use_dummy_weight    : bool             = False
    max_seq_len_override: int | None       = None
    num_page_override   : int | None       = None  # if not None, will override the number of pages
    speculative_algorithm        : str | None = None  # speculative decoding: "MTP" / "DFLASH" / None
    speculative_draft_model_path : str | None = None  # drafter weights path (must equal model_path for MTP)
    speculative_num_draft_tokens : int        = 4     # draft length K per verify round
    speculator_window_size       : int        = 128   # rolling carry window rows the drafter keeps per request
    enable_dt_separation         : bool       = False # run the drafter on its own device (needs tp+1 GPUs)

    @property
    def enable_mtp(self) -> bool:
        return self.speculative_algorithm is not None

    @cached_property
    def pretrained_config(self):
        return load_model_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        return ModelConfig.from_pretrained(self.pretrained_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        else:
            return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
