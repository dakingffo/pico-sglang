from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

from picosgl.distributed import DistributedInfo, try_get_tp_info
from picosgl.utils import Registry, div_even

if TYPE_CHECKING:    
    from picosgl.models import ModelConfig

from .kv.base import BaseKVCachePool
from .linear.state_pool import LinearStatePool
from .base import (
    BaseCacheHandle,
    BasePrefixCache,
    MatchResult,
    SizeInfo,
)

# Fallback when TP info is unset (unit tests build pools without an engine).
_TP_DEFAULT = DistributedInfo(rank=0, size=1)


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
    from .kv.mha_pool import MHAKVCachePool  # TODO: support other variants (e.g. MLA)

    return MHAKVCachePool(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_attention_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )

def _local_linear_head_counts(model_config: ModelConfig) -> tuple[int, int]:
    """Per-TP-rank linear-attention head counts. The loader shards in_proj/conv1d/
    A_log/dt_bias column-parallel, so each rank's state pool and budget use the
    LOCAL heads. Falls back to tp=1 when unset (unit tests)."""
    tp_size = (try_get_tp_info() or _TP_DEFAULT).size
    return (
        div_even(model_config.linear_num_key_heads, tp_size),
        div_even(model_config.linear_num_value_heads, tp_size),
    )


def linear_state_slot_bytes_for_config(model_config: ModelConfig, dtype: torch.dtype) -> int:
    if not model_config.is_hybrid:
        return 0
    num_k_heads, num_v_heads = _local_linear_head_counts(model_config)
    conv_dim = (
        num_k_heads * model_config.linear_key_head_dim * 2
        + num_v_heads * model_config.linear_value_head_dim
    )
    per_layer = (
        conv_dim * (model_config.linear_conv_kernel_dim - 1)
        + num_v_heads * model_config.linear_key_head_dim * model_config.linear_value_head_dim
    )
    return model_config.num_linear_layers * per_layer * dtype.itemsize


def create_linear_state_pool(
    model_config   : ModelConfig,
    num_slots      : int,
    device         : torch.device,
    dtype          : torch.dtype,
) -> LinearStatePool:
    num_k_heads, num_v_heads = _local_linear_head_counts(model_config)
    conv_dim = (
        num_k_heads * model_config.linear_key_head_dim * 2
        + num_v_heads * model_config.linear_value_head_dim
    )
    return LinearStatePool(
        num_slots=num_slots,
        num_linear_layers=model_config.num_linear_layers,
        conv_dim=conv_dim,
        kernel_size=model_config.linear_conv_kernel_dim,
        num_v_heads=num_v_heads,
        head_k_dim=model_config.linear_key_head_dim,
        head_v_dim=model_config.linear_value_head_dim,
        device=device,
        dtype=dtype,
    )


@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache(device: torch.device):
    from .naive_prefix_cache import NaivePrefixCache

    return NaivePrefixCache(device=device)


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(device: torch.device):
    from .radix_prefix_cache import RadixTreeNode, RadixPrefixCache

    return RadixPrefixCache(device=device, node_type=RadixTreeNode)


@SUPPORTED_CACHE_MANAGER.register("hybrid_radix")
def create_hybrid_radix_cache(device: torch.device):
    from .radix_prefix_cache import HybridRadixTreeNode, RadixPrefixCache

    return RadixPrefixCache(device=device, node_type=HybridRadixTreeNode)


def create_prefix_cache(device: torch.device, type: str) -> BasePrefixCache:
    return SUPPORTED_CACHE_MANAGER[type](device)


__all__ = [
    "create_kvcache_pool",
    "create_linear_state_pool",
    "create_prefix_cache",
    "linear_state_slot_bytes_for_config",
    "BaseKVCachePool",
    "BaseCacheHandle",
    "BasePrefixCache",
    "SizeInfo",
    "MatchResult",
    "SUPPORTED_CACHE_MANAGER",
]
