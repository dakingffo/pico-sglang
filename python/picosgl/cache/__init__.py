from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

from picosgl.utils import Registry

if TYPE_CHECKING:    
    from picosgl.models import ModelConfig

from .kv.base import (
    BaseCacheHandle,
    BaseKVCachePool,
    BasePrefixCache,
    MatchResult,
    SizeInfo,
)
from .linear.state_pool import LinearStatePool

class CacheManagerCreator(Protocol):
    def __call__(self, device: torch.device) -> BasePrefixCache: ...


SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")


def create_kvcache_pool(
    model_config: ModelConfig,
    num_pages   : int,
    page_size   : int,
    dtype       : torch.dtype,
    device      : torch.device,
) -> BaseKVCachePool:
    from .kv.pool import MHAKVCachePool  # TODO: support other variants (e.g. MLA)

    return MHAKVCachePool(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_attention_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )

def create_linear_state_pool(
    model_config   : ModelConfig,
    max_req        : int,
    device         : torch.device,
    dtype          : torch.dtype,
    enable_mtp     : bool = False,
    num_spec_tokens: int = 4,
) -> LinearStatePool:
    conv_dim = (
        model_config.linear_num_key_heads * model_config.linear_key_head_dim * 2
        + model_config.linear_num_value_heads * model_config.linear_value_head_dim
    )
    depth = num_spec_tokens + 1 if enable_mtp else 1
    return LinearStatePool(
        num_linear_layers=model_config.num_linear_layers,
        max_req=max_req,
        conv_dim=conv_dim,
        kernel_size=model_config.linear_conv_kernel_dim,
        num_v_heads=model_config.linear_num_value_heads,
        head_k_dim=model_config.linear_key_head_dim,
        head_v_dim=model_config.linear_value_head_dim,
        device=device,
        dtype=dtype,
        depth=depth,
    )


@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache(device: torch.device):
    from .kv.prefix_cache import NaivePrefixCache

    return NaivePrefixCache(device=device)


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(device: torch.device):
    from .kv.prefix_cache import RadixPrefixCache

    return RadixPrefixCache(device=device)


def create_prefix_cache(device: torch.device, type: str) -> BasePrefixCache:
    return SUPPORTED_CACHE_MANAGER[type](device)


__all__ = [
    "create_kvcache_pool",
    "create_linear_state_pool",
    "create_prefix_cache",
    "BaseKVCachePool",
    "BaseCacheHandle",
    "BasePrefixCache",
    "SizeInfo",
    "MatchResult",
    "SUPPORTED_CACHE_MANAGER",
]
