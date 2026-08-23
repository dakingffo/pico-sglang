from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from picosgl.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from picosgl.core import Batch, Request


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k       : torch.Tensor | None = None
    top_k_rows  : torch.Tensor | None = None
    top_p       : torch.Tensor | None = None
    top_p_rows  : torch.Tensor | None = None

    @property
    def is_greedy(self) -> bool:
        return self.temperatures is None


def make_device_tensor(data: list, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def normalize_probs(
    logits      : torch.Tensor,
    temperatures: torch.Tensor,
    top_k       : torch.Tensor | None,
    top_k_rows  : torch.Tensor | None,
    top_p       : torch.Tensor | None,
    top_p_rows  : torch.Tensor | None,
) -> torch.Tensor:
    import flashinfer.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is not None:
        assert top_k_rows is not None
        filtered = sampling.top_k_renorm_probs(
            probs.index_select(0, top_k_rows), top_k
        )
        probs.index_copy_(0, top_k_rows, filtered)
    if top_p is not None:
        assert top_p_rows is not None
        filtered = sampling.top_p_renorm_probs(
            probs.index_select(0, top_p_rows), top_p
        )
        probs.index_copy_(0, top_p_rows, filtered)
    return probs


def sample_impl(
    logits: torch.Tensor,
    args  : BatchSamplingArgs,
) -> torch.Tensor:
    import flashinfer.sampling as sampling

    assert args.temperatures is not None
    probs = normalize_probs(
        logits,
        args.temperatures,
        args.top_k,
        args.top_k_rows,
        args.top_p,
        args.top_p_rows,
    )
    return sampling.sampling_from_probs(probs)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        return self.prepare_params([r.sampling_params for r in batch.reqs])

    def prepare_params(self, params: list) -> BatchSamplingArgs:
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_k_rows = [i for i, p in enumerate(params) if 1 <= p.top_k < self.vocab_size]
        top_p_rows = [i for i, p in enumerate(params) if p.top_p < 1.0]
        temperatures = make_device_tensor(ts, torch.float32, self.device)

        return BatchSamplingArgs(
            temperatures,
            top_k=(
                make_device_tensor(
                    [params[i].top_k for i in top_k_rows], torch.int32, self.device
                ) if top_k_rows else None
            ),
            top_k_rows=(
                make_device_tensor(top_k_rows, torch.int64, self.device)
                if top_k_rows else None
            ),
            top_p=(
                make_device_tensor(
                    [max(params[i].top_p, MIN_P) for i in top_p_rows],
                    torch.float32,
                    self.device,
                ) if top_p_rows else None
            ),
            top_p_rows=(
                make_device_tensor(top_p_rows, torch.int64, self.device)
                if top_p_rows else None
            ),
        )

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        if args.is_greedy:  # greedy sampling
            return torch.argmax(logits, dim=-1)
        else:
            return sample_impl(logits.float(), args)

    def draft_token(self, logits_row: torch.Tensor, params) -> int:
        """Sample one MTP draft token from the same distribution used during verification."""
        if params.is_greedy:
            return int(logits_row.argmax().item())
        args = self.prepare_params([params])
        return int(self.sample(logits_row.unsqueeze(0), args)[0].item())

    def probabilities(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        assert args.temperatures is not None
        return normalize_probs(
            logits.float(),
            args.temperatures,
            args.top_k,
            args.top_k_rows,
            args.top_p,
            args.top_p_rows,
        )

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

        ``logits`` is (T, vocab) full-position logits; each request's n_drafts+1 rows are
        sliced by ``req.extend_len`` in batch.reqs order (same segmentation as the input tuple).
        """
        bs = batch.size
        K = 0 if batch.draft_tokens is None else batch.draft_tokens.shape[1]
        extend_token = torch.full((bs, K + 1), -1, dtype=torch.int32, device=self.device)
        if K == 0:
            return extend_token

        groups: dict[tuple[bool, int], list[tuple[int, int, Request]]] = {}
        offset = 0
        for batch_idx, req in enumerate(batch.reqs):
            num_drafts = req.extend_len - 1
            groups.setdefault((req.sampling_params.is_greedy, num_drafts), []).append(
                (batch_idx, offset, req)
            )
            offset += num_drafts + 1
        assert offset == logits.shape[0]

        for (is_greedy, num_drafts), entries in groups.items():
            seq_len = num_drafts + 1
            batch_indices = torch.tensor(
                [batch_idx for batch_idx, _offset, _req in entries],
                dtype=torch.int64,
                device=self.device,
            )
            row_indices = torch.tensor(
                [row for _batch_idx, begin, _req in entries
                 for row in range(begin, begin + seq_len)],
                dtype=torch.int64,
                device=self.device,
            )
            target_logits = logits.index_select(0, row_indices).view(
                len(entries), seq_len, self.vocab_size
            )
            drafts = batch.draft_tokens.index_select(
                0, batch_indices
            )[:, :num_drafts].contiguous()

            if is_greedy:
                target_tokens = target_logits.argmax(dim=-1).to(torch.int32)
                if num_drafts == 0:
                    accepted = torch.zeros(len(entries), dtype=torch.int64, device=self.device)
                else:
                    matches = target_tokens[:, :num_drafts] == drafts
                    accepted = matches.to(torch.int64).cumprod(dim=1).sum(dim=1)
                output = torch.full(
                    (len(entries), seq_len), -1, dtype=torch.int32, device=self.device
                )
                if num_drafts > 0:
                    positions = torch.arange(num_drafts, device=self.device)
                    output[:, :num_drafts] = torch.where(
                        positions[None, :] < accepted[:, None], drafts, -1
                    )
                bonus = target_tokens.gather(1, accepted[:, None])
                output.scatter_(1, accepted[:, None], bonus)
            else:
                params = [
                    req.sampling_params
                    for _batch_idx, _offset, req in entries
                    for _ in range(seq_len)
                ]
                args = self.prepare_params(params)
                target_probs = self.probabilities(
                    target_logits.flatten(0, 1), args
                ).view(len(entries), seq_len, self.vocab_size)
                if num_drafts == 0:
                    import flashinfer.sampling as sampling

                    output = sampling.sampling_from_probs(target_probs[:, 0]).unsqueeze(1)
                else:
                    import flashinfer.sampling as sampling

                    assert batch.draft_probs is not None, (
                        "sampling verify batch needs draft_probs"
                    )
                    draft_probs = batch.draft_probs.index_select(
                        0, batch_indices
                    )[:, :num_drafts].contiguous()
                    output, _, _ = sampling.chain_speculative_sampling(
                        draft_probs, drafts, target_probs
                    )
            extend_token[batch_indices, :seq_len] = output

        return extend_token
