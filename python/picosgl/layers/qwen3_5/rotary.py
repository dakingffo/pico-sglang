from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from picosgl.layers import StateLessOP

if TYPE_CHECKING:
    from picosgl.models.config import ModelConfig


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Qwen3_5RotaryEmbedding(StateLessOP):
    """Partial / interleaved RoPE (Qwen3.5 mrope).

    For pure text the T/H/W positions are identical, so the interleaved layout collapses to
    the standard RoPE applied on the first ``rotary_dim`` dims, with the rest passed through.
    """

    def __init__(self, head_dim: int, rotary_dim: int, max_position: int, base: float):
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self._max_position = max_position
        self._base = base
        self._cos_sin_cache: torch.Tensor | None = None

    def _get_cache(self, device: torch.device) -> torch.Tensor:
        cache = self._cos_sin_cache
        if cache is None or cache.device != device:
            # Inv-freqs are computed on the target device (the module is built on meta).
            inv_freq = 1.0 / (
                self._base
                ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=device) / self.rotary_dim)
            )
            t = torch.arange(self._max_position, device=device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)  # (max_pos, rotary_dim)
            cache = torch.cat([emb.cos(), emb.sin()], dim=-1)  # (max_pos, 2*rotary_dim)
            self._cos_sin_cache = cache
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self._get_cache(query.device)
        cos = cache[positions, : self.rotary_dim].unsqueeze(1)  # (T, 1, rotary_dim)
        sin = cache[positions, self.rotary_dim :].unsqueeze(1)
        q_rot, q_pass = query[..., : self.rotary_dim], query[..., self.rotary_dim :]
        k_rot, k_pass = key[..., : self.rotary_dim], key[..., self.rotary_dim :]
        q_embed = (q_rot * cos + _rotate_half(q_rot) * sin).to(query.dtype)
        k_embed = (k_rot * cos + _rotate_half(k_rot) * sin).to(key.dtype)
        return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)
