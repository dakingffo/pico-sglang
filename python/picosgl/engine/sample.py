from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from picosgl.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from picosgl.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k       : torch.Tensor | None = None
    top_p       : torch.Tensor | None = None

    @property
    def is_greedy(self) -> bool:
        return self.temperatures is None


def make_device_tensor(data: list, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits      : torch.Tensor,
    temperatures: torch.Tensor,
    top_k       : torch.Tensor | int | None,
    top_p       : torch.Tensor | float | None,
) -> torch.Tensor:
    import flashinfer.sampling as sampling
    
    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)
    elif top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)
    elif top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)
    else:
        assert top_k is not None and top_p is not None
        return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)

        return BatchSamplingArgs(
            temperatures,
            top_k=(
                make_device_tensor(top_ks, torch.int32, self.device) 
                if any(k != self.vocab_size for k in top_ks) else None
            ),
            top_p=(
                make_device_tensor(top_ps, torch.float32, self.device) 
                if any(p < 1.0 for p in top_ps) else None
            )
        )

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        if args.is_greedy:  # greedy sampling
            return torch.argmax(logits, dim=-1)
        else:
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)

    def draft_token(self, logits_row: torch.Tensor, params) -> int:
        """Sample one MTP draft token. Greedy -> argmax; sampling -> draw from the request's
        target distribution (the SAME ``_target_dist`` used by ``reject_sample``, so the
        ``p_draft`` in the residual is exactly the distribution the draft came from)."""
        if params.is_greedy:
            return int(logits_row.argmax().item())
        dist = self._target_dist(logits_row, params)
        return int(torch.multinomial(dist, 1).item())


    def _target_dist(self, logits: torch.Tensor, params) -> torch.Tensor:
        """Request's target sampling distribution: softmax(logits/temp) with top_k / top_p
        masking. Mirrors flashinfer ``sample_impl`` for the common case (top_k=-1, top_p=1);
        top_k/top_p use the standard semantics (keep top-k tokens; smallest set whose cumsum
        >= top_p, always keeping the argmax). Logits is (..., vocab), returns same shape."""
        scaled = logits.float() / max(params.temperature, 1e-6)
        probs = torch.softmax(scaled, dim=-1)
        if params.top_k >= 1 and params.top_k < self.vocab_size:
            k = min(params.top_k, self.vocab_size)
            keep = torch.zeros_like(probs, dtype=torch.bool)
            keep.scatter_(-1, torch.topk(probs, k, dim=-1).indices, True)
            scaled = scaled.masked_fill(~keep, float("-inf"))
        if params.top_p < 1.0:
            sorted_probs, sort_idx = torch.sort(torch.softmax(scaled, dim=-1), -1, True)
            keep = torch.cumsum(sorted_probs, dim=-1) <= params.top_p
            keep[..., 0] = True  # always keep the argmax
            keep_mask = torch.zeros_like(scaled, dtype=torch.bool)
            keep_mask.scatter_(-1, sort_idx, keep)
            scaled = scaled.masked_fill(~keep_mask, float("-inf"))
        return torch.softmax(scaled, dim=-1)

    @nvtx_annotate("Sampler")
    def reject_sample(self, logits, batch, sample_args) -> torch.Tensor:
        """Spec-decode rejection sampling. Returns extend_token (bs, K+1) int32.

        Row layout per request: [draft_0, ..., draft_{num_sampled-2}, new_bonus, PAD...]
        (PAD = -1). VerifyManager derives num_sampled by comparing extend_token against its
        own draft_tokens: the first differing index i gives num_sampled = i+1 (if every draft
        matched, num_sampled = K+1 and the bonus is extend_token[K]).

        greedy  : drafts are argmax (delta distribution); accept draft_i iff
                  argmax(logits[C+i]) == draft_i; bonus = argmax there (never equals the
                  rejected draft).
        sampling: drafts are sampled from the MTP head; accept draft_i with prob
                  min(1, p_target(d_i)/p_draft(d_i)); on rejection the bonus is drawn via
                  residual-Gumbel from max(0, p_target - p_draft) -- the residual at the
                  rejected draft is 0 (rejection implies p_target(d) < p_draft(d)), so the
                  bonus never equals it, which keeps the count-by-comparison safe. When all
                  drafts are accepted the bonus is sampled from p_target at the last window
                  position.

        ``logits`` is (T, vocab) full-position logits; each request's K_r+1 rows are sliced
        by ``req.extend_len`` in batch.reqs order (same segmentation as the input tuple).
        """
        bs = batch.size
        K = 0 if batch.draft_tokens is None else batch.draft_tokens.shape[1]
        extend_token = torch.full((bs, K + 1), -1, dtype=torch.int32, device=self.device)
        if K == 0:
            return extend_token

        offset = 0
        for i, req in enumerate(batch.reqs):
            num_drafts = req.extend_len - 1  # K_r
            seg = logits[offset : offset + num_drafts + 1]
            offset += num_drafts + 1
            seg_float = seg.float()
            drafts = batch.draft_tokens[i]  # (K,) int32
            params = req.sampling_params

            if params.is_greedy:
                argmax = torch.argmax(seg_float, dim=-1)  # (K_r+1,)
                accepted = 0
                if num_drafts > 0:
                    # one host sync for the whole row; extend_token writes stay on device
                    argmax_cpu = argmax[:num_drafts].tolist()
                    drafts_cpu = drafts[:num_drafts].tolist()
                    for j in range(num_drafts):
                        if argmax_cpu[j] == drafts_cpu[j]:
                            accepted = j + 1
                            extend_token[i, j] = drafts[j]
                        else:
                            break
                extend_token[i, accepted] = argmax[accepted]
                continue

            # ---- sampling path ----
            target_probs = self._target_dist(seg_float, params)  # (K_r+1, vocab)
            assert batch.draft_probs is not None, "sampling verify batch needs draft_probs"
            draft_probs = batch.draft_probs[i]  # (K, vocab) fp32
            accepted = 0
            for j in range(num_drafts):
                d = int(drafts[j])
                p_t = target_probs[j, d].item()
                p_d = max(draft_probs[j, d].item(), 1e-6)
                if torch.rand((), device=self.device).item() <= min(1.0, p_t / p_d):
                    accepted = j + 1
                    extend_token[i, j] = drafts[j]
                else:
                    break
            if accepted < num_drafts:
                # rejected at draft `accepted`: residual-Gumbel over max(0, p_target - p_draft).
                # residual(draft) = 0 since rejection implies p_target(d) < p_draft(d).
                residual = torch.clamp(target_probs[accepted] - draft_probs[accepted], min=0.0)
                total = residual.sum()
                if total <= 0:  # degenerate; should not happen (see docstring)
                    bonus = int(torch.argmax(seg_float[accepted]).item())
                else:
                    bonus = int(torch.multinomial(residual / total, 1).item())
            else:
                # all drafts accepted: bonus from p_target at the last window position
                bonus = int(torch.multinomial(target_probs[num_drafts], 1).item())
            extend_token[i, accepted] = bonus

        return extend_token
