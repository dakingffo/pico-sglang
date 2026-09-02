from __future__ import annotations

from typing import Protocol

from picosgl.utils import Registry

from .base import (
    BaseLinearAttentionBackend,
    GatedDeltaConfig,
    GatedDeltaForwardInput,
    GatedDeltaInput,
)


class LinearAttentionBackendCreator(Protocol):
    def __call__(self) -> BaseLinearAttentionBackend: ...


SUPPORTED_LINEAR_ATTENTION_BACKENDS = Registry[LinearAttentionBackendCreator](
    "Linear Attention Backend"
)


@SUPPORTED_LINEAR_ATTENTION_BACKENDS.register("native")
def make_native_backend() -> BaseLinearAttentionBackend:
    from .native import NativeLinearAttentionBackend

    return NativeLinearAttentionBackend()


@SUPPORTED_LINEAR_ATTENTION_BACKENDS.register("fla")
def make_fla_backend() -> BaseLinearAttentionBackend:
    from .fla import FlashLinearAttentionBackend

    return FlashLinearAttentionBackend()


def make_linear_attention_backend(backend: str) -> BaseLinearAttentionBackend:
    return SUPPORTED_LINEAR_ATTENTION_BACKENDS[backend]()


__all__ = [
    "BaseLinearAttentionBackend",
    "GatedDeltaConfig",
    "GatedDeltaForwardInput",
    "GatedDeltaInput",
    "SUPPORTED_LINEAR_ATTENTION_BACKENDS",
    "make_linear_attention_backend",
]
