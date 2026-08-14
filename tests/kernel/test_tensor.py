from __future__ import annotations

from picosgl.kernel import test_tensor as _run_test_tensor
from picosgl.utils import call_if_main
import torch


@call_if_main(__name__)
def main():
    x = torch.empty((12, 2048), dtype=torch.int32, device="cpu")[:, :1024]
    y = torch.empty((12, 1024), dtype=torch.int64, device="cuda:0")
    _run_test_tensor(x, y)
