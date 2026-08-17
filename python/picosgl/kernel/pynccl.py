from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal

from picosgl.env import ENV

from .utils import load_aot

if TYPE_CHECKING:
    from abc import abstractmethod

    import torch
    from tvm_ffi import Module

    class PyNCCLCommunicator:
        @abstractmethod
        def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None: ...
        @abstractmethod
        def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...
        @abstractmethod
        def get_buffer(self) -> int: ...

else:
    PyNCCLCommunicator = Any


@functools.cache
def _load_nccl_module() -> Module:
    return load_aot("pynccl", cuda_files=["pynccl.cu"], extra_ldflags=["-lnccl"])


@functools.cache
def _get_pynccl_wrapper_cls():
    import tvm_ffi

    @tvm_ffi.register_object("picosgl.NCCLWrapper")
    class PyNCCLImpl(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)

    return PyNCCLImpl


def init_pynccl(
    *,
    tp_rank: int,
    tp_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
) -> PyNCCLCommunicator:
    import torch

    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)

    module = _load_nccl_module()
    cls = _get_pynccl_wrapper_cls()

    if tp_rank == 0:
        id_list = [module.create_nccl_uid()]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )
    else:
        id_list = [None]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )

    nccl_id = id_list[0]
    assert not nccl_id is None, f"Failed to get NCCL unique ID on {tp_rank = }"

    # bypass type checking for the FFI object
    return cls(tp_rank, tp_size, max_size_bytes, nccl_id)  # type: ignore


def create_nccl_uid_bytes() -> bytes:
    """Create a fresh ncclUniqueId and return its 128 raw bytes (for the zmq handshake)."""
    return bytes((v & 0xFF for v in _load_nccl_module().create_nccl_uid()))


def _nccl_uid_to_ffi(nccl_uid: bytes) -> list[int]:
    """Decode the 128 raw uid bytes into the signed-char ``Array<char>`` the FFI expects.

    ``create_nccl_uid()`` returns ``tvm_ffi.container.Array`` of signed chars (some values
    negative), and the FFI init round-trips them as a plain Python ``list[int]``. The
    control-plane message carries the masked (0..255) raw bytes, so convert back to the
    signed-char range here. No masking is needed for int8 bytes since NCCL_UNIQUE_ID_BYTES
    is exactly 128.
    """
    assert len(nccl_uid) == 128, f"ncclUniqueId must be 128 bytes, got {len(nccl_uid)}"
    return [v if v < 128 else v - 256 for v in nccl_uid]


def init_pynccl_p2p(
    *,
    rank          : int,
    world_size    : int,
    nccl_uid      : bytes,
    max_size_bytes: int = 0,
) -> PyNCCLCommunicator:
    """Init a standalone NCCL communicator from a uid shipped over the control plane.

    Used for the 2-rank target↔drafter data plane, which has no gloo group to broadcast
    the uid through (unlike the TP group's ``init_pynccl``). ``nccl_uid`` is the 128 raw
    bytes of the ncclUniqueId created on rank 0 and carried by DraftHandshakeMsg.
    """
    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)

    module = _load_nccl_module()
    cls = _get_pynccl_wrapper_cls()
    # bypass type checking for the FFI object
    return cls(rank, world_size, max_size_bytes, _nccl_uid_to_ffi(nccl_uid))  # type: ignore
