from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from picosgl.layers import (
    BaseOP,
    LinearColumnParallel,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from picosgl.layers.qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm
from picosgl.models import load_weight

if TYPE_CHECKING:
    from picosgl.models.config import ModelConfig


class Qwen3_5MTPDrafter(BaseOP):
    """Standalone MTP drafter with its own embedding and LM head.

    The checkpoint stores no ``mtp.*`` embed/lm_head keys; those are the target's
    ``model.embed_tokens.weight``, loaded by ``load_weights``.

    Weight keys keep the ``mtp.*`` prefix so the drafter loads exactly the ``mtp.*``
    slice of a Qwen3.5 checkpoint (``load_state_dict(..., prefix="mtp")``). Runs at tp=1
    in its own process; ``draft`` never reads ``ctx.batch``.
    """

    def __init__(self, config: ModelConfig):
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Input is the cat([embedding; hidden]) — full width on every rank — so the fc
        # must be output-split (LinearColumnParallel). tp=1 drafter, kept as-is.
        self.fc = LinearColumnParallel(config.hidden_size * 2, config.hidden_size, has_bias=False)
        self.layers = OPList(
            [
                Qwen3_5DecoderLayer(
                    config, 0, block_type="full_attention", paged=False
                )
            ]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Own embedding + LM head. Underscore-named: excluded from BaseOP.state_dict, so
        # the drafter's state_dict is exactly the mtp.* checkpoint slice; embed/lm_head
        # are filled by load_weights (a tied lm_head aliases the embedding).
        self._embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self._lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=(
                self._embed_tokens if config.tie_word_embeddings else None
            ),
        )

    def load_weights(self, model_path: str, device: torch.device) -> None:
        """Load the ``mtp.*`` weights plus the target's embed/lm_head into this drafter."""
        mtp_sd: dict[str, torch.Tensor] = {}
        embed: torch.Tensor | None = None
        lm_head: torch.Tensor | None = None
        for key, value in load_weight(model_path, device):
            key = key.removeprefix("model.")
            if key.startswith("mtp."):
                mtp_sd[key] = value
            elif key == "embed_tokens.weight":
                embed = value
            elif key == "lm_head.weight":
                lm_head = value
        assert embed is not None, f"checkpoint {model_path} missing embed_tokens.weight"
        self.load_state_dict(mtp_sd, prefix="mtp")
        self._embed_tokens.weight = embed.to(self._embed_tokens.weight.dtype)
        if not self._lm_head.tied_embedding:
            assert lm_head is not None, (
                f"untied lm_head needs checkpoint lm_head.weight ({model_path})"
            )
            self._lm_head.weight = lm_head.to(self._lm_head.weight.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        emb = self._embed_tokens.forward(input_ids)
        emb = self.pre_fc_norm_embedding.forward(emb)
        h = self.pre_fc_norm_hidden.forward(hidden_states)
        h = torch.cat([emb, h], dim=-1)
        h = self.fc.forward(h)
        h = self.layers.op_list[0].forward(h, positions)
        return self.norm.forward(h)

    def get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project MTP output to vocab logits via the drafter's own lm_head.

        The lm_head is vocab-parallel (each rank holds vocab/tp rows), so the per-rank
        logits must be all-gathered into full-vocab logits before the draft argmax /
        p_draft (draft_probs is sized to the full vocab). Mirrors ParallelLMHead.forward
        but standalone: draft runs outside a forward batch, so it must not read ctx.batch.
        """
        module = self._lm_head.tied_embedding or self._lm_head
        logits = F.linear(hidden_states, module.weight, self._lm_head.bias)
        if self._lm_head.tp_size > 1:
            input_shape = logits.shape
            gathered = self._lm_head._comm.all_gather(logits)
            gathered = gathered.view((self._lm_head.tp_size,) + input_shape)
            gathered = gathered.permute(1, 0, 2).contiguous()
            logits = gathered.reshape(
                input_shape[:1] + (self._lm_head.tp_size * input_shape[1],)
            )[:, : module.num_embeddings]
        return logits

    def draft(self, input_ids, positions, hidden_states, past_kv=None):
        """Single MTP draft step, returns (next_token, logits, hidden, kv).

        Step-0 (past_kv=None): feed the carry window (last W accepted tokens + their target
        hidden); the last row's logits predict the next token -> draft_0, and the window's
        KV is returned for carry. Step-j (j>=1): feed [draft_{j-1}] + this predictor's own
        hidden from step j-1, attending to the carried KV -> draft_j.

        next_token is greedy (argmax); p_draft is recovered from ``logits`` by the caller.
        """
        emb = self._embed_tokens.forward(input_ids)
        emb = self.pre_fc_norm_embedding.forward(emb)
        h = self.pre_fc_norm_hidden.forward(hidden_states)
        h = torch.cat([emb, h], dim=-1)
        h = self.fc.forward(h)

        layer = self.layers.op_list[0]
        residual = h
        h = layer.input_layernorm.forward(h)
        h, kv = layer.self_attn.forward_with_kv(h, positions, past_kv)
        h = residual + h
        residual = h
        h = layer.post_attention_layernorm.forward(h)
        h = layer.mlp.forward(h)
        h = residual + h
        h = self.norm.forward(h)

        logits = self.get_logits(h)
        return logits.argmax(dim=-1), logits, h, kv

    def draft_batch(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        valid_mask: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Run one dense batched MTP step without consulting the global engine context."""
        assert input_ids.ndim == 2
        assert hidden_states.shape[:2] == input_ids.shape
        batch_size, seq_len = input_ids.shape
        # The custom embedding kernel consumes a flat token vector.
        emb = self._embed_tokens.forward(input_ids.reshape(-1)).view(
            batch_size, seq_len, -1
        )
        emb = self.pre_fc_norm_embedding.forward(emb)
        h = self.pre_fc_norm_hidden.forward(hidden_states)
        h = torch.cat([emb, h], dim=-1)
        h = self.fc.forward(h)

        layer = self.layers.op_list[0]
        residual = h
        h = layer.input_layernorm.forward(h)
        h, kv, key_valid = layer.self_attn.forward_with_kv_batch(
            h, positions, valid_mask, past_kv, past_valid_mask
        )
        h = residual + h
        residual = h
        h = layer.post_attention_layernorm.forward(h)
        h = layer.mlp.forward(h)
        h = self.norm.forward(residual + h)

        # Every carry window is left-padded, so the predicting row is always the last
        # one. Avoid materializing the prohibitively large (B, T, vocab) tensor.
        logits = self.get_logits(h[:, -1])
        return logits, h, kv, key_valid
