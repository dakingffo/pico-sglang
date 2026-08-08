from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from picosgl.kvcache import BaseCacheHandle, BaseKVCachePool
    from picosgl.layers.attention_backend import BaseAttnBackend, BaseAttnMetadata
    from picosgl.layers.moe_backend import BaseMoeBackend


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k      : int   = -1
    top_p      : float = 1.0
    ignore_eos : bool  = False
    max_tokens : int   = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Request:
    input_ids      : torch.Tensor  # assert is_cpu
    table_idx      : int
    cached_len     : int
    output_len     : int
    uid            : int
    sampling_params: SamplingParams
    cache_handle   : BaseCacheHandle
    device_len     : int = field(init=False)
    max_device_len : int = field(init=False)

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)
        self.max_device_len = self.device_len + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    def __lt__(self, other: Request) -> bool:
        return self.uid < other.uid
    
    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        self.input_ids = torch.cat([self.input_ids, next_token])

    @property
    def can_decode(self) -> bool:
        return self.remain_len > 0

    def __repr__(self) -> str:
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )

class ChunkedRequest(Request):
    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedRequest should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False  # avoid being added to decode manager


@dataclass
class Batch:
    reqs     : list[Request]
    phase    : Literal["prefill", "decode"]
    # these fields should be set by scheduler
    input_ids  : torch.Tensor  = field(init=False)
    positions  : torch.Tensor  = field(init=False)
    out_loc    : torch.Tensor  = field(init=False)
    padded_reqs: list[Request] = field(init=False)
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)
    

@dataclass
class Context:
    page_size: int
    # NOTE: this table always treat page_size = 1
    page_table  : torch.Tensor    = field(init=False)
    attn_backend: BaseAttnBackend = field(init=False)
    moe_backend : BaseMoeBackend  = field(init=False)
    kv_cache    : BaseKVCachePool = field(init=False)
    _batch      : Batch | None    = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None | dict[str, Context | None]  = None


def set_global_ctx(ctx: Context, key: str | None = None) -> None:
    global _GLOBAL_CTX
    if key is not None:
        assert isinstance(_GLOBAL_CTX, dict), "Global context is not a dict"
        value = _GLOBAL_CTX.get(key)
        assert value is None, f"Global context for '{key=}' is already set"
        _GLOBAL_CTX[key] = ctx
    else:
        assert _GLOBAL_CTX is None, "Global context is already set"
        _GLOBAL_CTX = ctx


def get_global_ctx(key: str | None = None) -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    if key is not None:
        assert isinstance(_GLOBAL_CTX, dict), "Global context is not a dict"
        value = _GLOBAL_CTX.get(key)
        assert value is not None, f"Global context for '{key=}' is not set"
        return value
    else:
        return _GLOBAL_CTX


def clear_global_ctx(key: str | None = None) -> None:
    global _GLOBAL_CTX
    if key is not None:
        assert isinstance(_GLOBAL_CTX, dict), "Global context is not a dict"
        value = _GLOBAL_CTX.get(key, -1)
        assert value is not -1, f"Global context does not have '{key=}'"
        _GLOBAL_CTX[key] = None
    else:
        _GLOBAL_CTX = None


__all__ = [
    "SamplingParams",
    "Request",
    "Batch",
    "Context",
    "set_global_ctx",
    "get_global_ctx",
    "clear_global_ctx",
]