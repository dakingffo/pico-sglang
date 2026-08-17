from __future__ import annotations

import torch

from picosgl.core import SamplingParams

from ...base import DraftState


class MTPState(DraftState):
    """Drafter-side persistent per-request MTP state.

    Mirror of the old in-model ``VerifyState``. The ``carry_*`` fields describe the rolling
    window (last ``window_size`` rows) of the *target's* committed tokens + main-model hidden
    states. ``carry_tokens`` replaces the target-side token_pool reads: the target ships the
    window token ids in the control message because the drafter process has no token pool.
    ``mtp_kv`` is this drafter's OWN KV computed by ``MTPEngine.draft`` (never the target's).
    """

    def __init__(
        self,
        sampling_params: SamplingParams,
        carry_positions: list[int],
        carry_tokens   : list[int],
        carry_hidden   : torch.Tensor,
        window_size    : int,
    ) -> None:
        super().__init__()
        self.sampling_params = sampling_params
        self.carry_positions = list(carry_positions)
        self.carry_tokens    = list(carry_tokens)
        self.carry_hidden    = carry_hidden
        self.window_size     = window_size
        self.mtp_kv          = None
        self.n_drafts        = 0

    def update_carry(self, positions: list[int], tokens: list[int], hidden: torch.Tensor) -> None:
        """Append the target's newly committed rows, then front-trim to ``window_size``.

        Verbatim port of ``VerifyManager._update_carry`` (positions / hidden, plus the new
        ``carry_tokens`` list since the drafter has no token_pool). ``mtp_kv`` is trimmed in
        lockstep so ``start = mtp_kv[0].shape[1]`` stays the count of carried KV rows and the
        next step-0 only re-processes the freshly committed tail.
        """
        self.carry_positions.extend(positions)
        self.carry_tokens.extend(tokens)
        self.carry_hidden = (
            torch.cat([self.carry_hidden, hidden], dim=0)
            if self.carry_hidden is not None else hidden
        )
        reserved_len = max(0, len(self.carry_positions) - self.window_size)
        if reserved_len:
            self.carry_positions = self.carry_positions[reserved_len:]
            self.carry_tokens    = self.carry_tokens[reserved_len:]
            self.carry_hidden    = self.carry_hidden[reserved_len:]
            if self.mtp_kv is not None:
                k, v = self.mtp_kv
                self.mtp_kv = (k[:, reserved_len:].contiguous(), v[:, reserved_len:].contiguous())
