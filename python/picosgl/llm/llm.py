from __future__ import annotations

from dataclasses import dataclass

import torch

from picosgl.core import SamplingParams
from picosgl.distributed import DistributedInfo
from picosgl.message import (
    BaseBackendMsg,
    DetokenizeMsg,
    UserMsg,
)
from picosgl.scheduler import Scheduler, SchedulerConfig


class RequestAllFinished(Exception):
    pass


@dataclass
class RequestStatus:
    uid       : int
    input_ids : list[int]
    output_ids: list[int]


class LLM(Scheduler):
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        super().__init__(SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(0, 1),
            dtype=dtype,
            offline_mode=True,
            **kwargs,
        ))
        self.pending_requests: list[tuple[list[int] | str, SamplingParams]] = []
        self.status_map: dict[int, RequestStatus] = {}
        self.counter = 0

    def _tokenize_one(self, prompt: list[int] | str) -> torch.Tensor:
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        else:
            return torch.tensor(prompt, dtype=torch.int32, device="cpu")

    def offline_receive_msg(self, blocking: bool = False) -> list[BaseBackendMsg]:
        if blocking and len(self.pending_requests) == 0:
            raise RequestAllFinished()
        results: list[BaseBackendMsg] = []
        added, sum_input_len = 0, 0
        for tokens_or_prompt, sampling_params in self.pending_requests:
            if sum_input_len >= self.prefill_budget:
                break
            input_ids = self._tokenize_one(tokens_or_prompt)
            sum_input_len += len(input_ids)
            uid, added = self.counter + added, added + 1
            results.append(UserMsg(uid=uid, input_ids=input_ids, sampling_params=sampling_params))
            self.status_map[uid] = RequestStatus(
                uid=uid,
                input_ids=(
                    input_ids.tolist() if isinstance(tokens_or_prompt, str) else tokens_or_prompt
                ),
                output_ids=[],
            )
        self.counter += added
        self.pending_requests = self.pending_requests[added:]
        return results

    def offline_send_result(self, reply: list[DetokenizeMsg]) -> None:
        for msg in reply:
            status = self.status_map[msg.uid]
            if not (msg.finished and msg.next_token == self.eos_token_id):
                status.output_ids.append(msg.next_token)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: list[SamplingParams] | SamplingParams,
    ) -> list[dict[str, str | list[int]]]:
        self.pending_requests = []
        self.status_map = {}
        self.counter = 0
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.pending_requests.append((prompt, sp))
        try:
            self.run_forever()
        except RequestAllFinished:
            pass
        results: list[dict[str, str | list[int]]] = []
        for i in range(len(prompts)):
            status = self.status_map[i]
            output_text = self.tokenizer.decode(status.output_ids)
            results.append({"text": output_text, "token_ids": status.output_ids})
        return results
