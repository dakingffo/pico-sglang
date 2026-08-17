from __future__ import annotations

import os
from functools import partial
from typing import Callable, Generic, TypeVar
from abc import ABC, abstractmethod

class BaseEnv(ABC):
    @abstractmethod
    def get_value_by(self, name: str) -> None: ...

    @abstractmethod
    def __bool__(self): ...

    @abstractmethod
    def __str__(self): ...


T = TypeVar("T")
class EnvVar(BaseEnv, Generic[T]):
    def __init__(self, default_value: T, fn: Callable[[str], T]):
        self.value = default_value
        self.fn = fn
        super().__init__()

    def get_value_by(self, name: str) -> None:
        env_value = os.getenv(name)
        if env_value is not None:
            try:
                self.value = self.fn(env_value)
            except Exception:
                pass

    def __bool__(self):
        return self.value

    def __str__(self):
        return str(self.value)


_TO_BOOL  = lambda x: x.lower() in ("1", "true", "yes")
_UNIT_MAP = {"K": 1024, "M": 1024**2, "G": 1024**3}

def _PARSE_MEM_BYTES(mem: str) -> int:
    mem = mem.strip().upper()
    if not mem[-1].isalpha():
        return int(mem)
    if mem.endswith("B"):
        mem = mem[:-1]
    
    return int(float(mem[:-1]) * _UNIT_MAP[mem[-1]])


picosgl_ENV_PREFIX = "picosgl_"

EnvInt    = partial(EnvVar[int], fn=int)
EnvFloat  = partial(EnvVar[float], fn=float)
EnvBool   = partial(EnvVar[bool], fn=_TO_BOOL)
EnvOption = partial(EnvVar[bool | None], fn=_TO_BOOL, default_value=None)
EnvMem    = partial(EnvVar[int], fn=_PARSE_MEM_BYTES)


class EnvClassSingleton:
    _instance: EnvClassSingleton | None = None

    # shell
    SHELL_MAX_TOKENS  = EnvInt(2048)
    SHELL_TOP_K       = EnvInt(-1)
    SHELL_TOP_P       = EnvFloat(1.0)
    SHELL_TEMPERATURE = EnvFloat(0.6)

    # backend runtime
    FLASHINFER_USE_TENSOR_CORES = EnvOption()
    PYNCCL_MAX_BUFFER_SIZE      = EnvMem(_UNIT_MAP["G"])
    # drafter target<->drafter data plane transport: "nccl" (production) or "pipe"
    # (local single-GPU WSL2, which cannot bootstrap a 2-rank NCCL communicator)
    DRAFTER_DATA_PLANE          = EnvVar[str]("nccl", str)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self, attr_name)
            assert isinstance(attr_value, BaseEnv)
            attr_value.get_value_by(f"{picosgl_ENV_PREFIX}{attr_name}")


ENV = EnvClassSingleton()
