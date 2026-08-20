from __future__ import annotations

import torch

from picosgl.core import SamplingParams

from ...base import DraftState


class MTPState(DraftState):
    def __init__(
        self,
        sampling_params : SamplingParams,
        window_positions: list[int],
        window_tokens   : list[int],
        window_hidden   : torch.Tensor,
        window_size     : int,
    ) -> None:
        super().__init__()
        self.sampling_params  = sampling_params
        self.window_positions = list(window_positions)
        self.window_tokens    = list(window_tokens)
        self.window_hidden    = window_hidden
        self.window_size      = window_size
        self.mtp_kv           = None
        self.n_drafts         = 0

    def update_window(self, positions: list[int], tokens: list[int], hidden: torch.Tensor) -> None:
        self.window_positions.extend(positions)
        self.window_tokens.extend(tokens)
        self.window_hidden = (
            torch.cat([self.window_hidden, hidden], dim=0)
            if self.window_hidden is not None else hidden
        )
        if reserved_len := max(0, len(self.window_positions) - self.window_size):
            self.window_positions = self.window_positions[reserved_len:]
            self.window_tokens    = self.window_tokens[reserved_len:]
            self.window_hidden    = self.window_hidden[reserved_len:]
            if self.mtp_kv is not None:
                k, v = self.mtp_kv
                self.mtp_kv = (k[:, reserved_len:].contiguous(), v[:, reserved_len:].contiguous())
