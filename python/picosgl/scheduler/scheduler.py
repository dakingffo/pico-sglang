from __future__ import annotations

from typing import NoReturn

import torch

from picosgl.core import Request
from picosgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    ExitMsg,
    UserMsg,
)
from picosgl.speculator import (
    BroadcastDrafterClient,
    DrafterClientBase,
    LocalDrafterClient,
    RemoteDrafterClient,
)
from picosgl.utils import init_logger, load_tokenizer, div_ceil
from picosgl.engine import Engine, ForwardOutput, ForwardData

from .ar import ForwardInput
from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import PrefillAdder, PrefillManager
from .table import TableManager
from .verify import VerifyManager

logger = init_logger(__name__)


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        self.engine = Engine(config)
        super().__init__(config, self.engine.tp_cpu_group)
        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.config = config
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type,
            num_states=getattr(self.engine, "num_states", 0),
            state_table=getattr(self.engine, "state_table", None),
            state_pool=getattr(self.engine, "linear_state", None),
            draft_state=getattr(self.engine, "draft_state", None),
        )
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.drafter_client = None
        if config.enable_mtp:
            self.drafter_client = self._make_drafter_client()
            self.ar_manager = VerifyManager(
                config, self.device,
                self.cache_manager, self.table_manager,
                self.eos_token_id, self.drafter_client,
                config.model_config.vocab_size,
            )
        else:
            self.ar_manager = DecodeManager(
                config, self.device,
                self.cache_manager, self.table_manager,
                self.eos_token_id
            )

        self.decode_manager = self.ar_manager
        self.verify_manager = self.ar_manager
        self.prefill_manager = PrefillManager(self.token_pool)
        self.prefill_budget = config.max_prefill_length

    def _make_drafter_client(self) -> DrafterClientBase:
        mc = self.engine.config.model_config
        if not self.config.tp_info.is_primary():
            return BroadcastDrafterClient(
                self.config, self.device, mc.vocab_size, mc.hidden_size,
                scheduler_io=self
            )
        if self.config.enable_dt_separation:
            return RemoteDrafterClient(
                self.config, self.device, mc.vocab_size, mc.hidden_size,
                scheduler_io=self
            )
        else:
            from picosgl.distributed import DistributedInfo, tp_override
            from picosgl.models.drafters import Qwen3_5MTPDrafter
            from picosgl.speculator import MTPEngine
            with tp_override(DistributedInfo(0, 1)):
                drafter = Qwen3_5MTPDrafter(mc)
                drafter.load_weights(self.config.speculative_draft_model_path, self.device)
                engine = MTPEngine(
                    drafter, self.device, mc.vocab_size, self.config.speculative_num_draft_tokens
                )
            return LocalDrafterClient(
                self.config, self.device, mc.vocab_size, mc.hidden_size,
                engine=engine, scheduler_io=self
            )

    def run_when_idle(self) -> None:
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.ar_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        return ongoing_data

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        assert torch.cuda.current_stream() == self.stream
        data = None
        while True:
            data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        if self.drafter_client is not None:
            self.drafter_client.destroy()
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return
        last_input, last_output = last_data
        last_output.copy_done_event.synchronize()
        reply, finished_reqs = self.ar_manager.process(self.engine.ctx, last_input, last_output)
        self.send_result(reply)
        for req in finished_reqs:
            self._free_req_resources(req)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req: Request
            inflight: bool
            req, inflight = self.prefill_manager.abort_req(msg.uid)
            if req is None:
                req, inflight = self.ar_manager.abort_req(msg.uid)
            if req is not None:
                self._free_req_resources(req, inflight)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Request, inflight: bool = False) -> None:
        if inflight: # The inflight aborted req has been allocated at [cached_len, device_len)
            C, D = req.cached_len, req.device_len
            ps = self.cache_manager.page_size
            for page in range(div_ceil(C, ps), div_ceil(D, ps)):
                p_start = page * ps
                tid = req.table_idx
                self.cache_manager._free(
                    self.table_manager.page_table[tid, p_start: p_start + ps]
                )
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def _schedule_next_batch(self) -> ForwardInput | None:
        batch = (
            self.prefill_manager.schedule_next_batch(
                PrefillAdder(
                    token_budget=self.prefill_budget,
                    reserved_size=self.ar_manager.need_tokens,
                    cache_manager=self.cache_manager,
                    table_manager=self.table_manager,
                )
            ) or
            self.ar_manager.schedule_next_batch()
        )
        return self.engine.prepare_batch(batch, self.cache_manager) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, _ = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.prefill_manager.advance_for_next_schedule(
            self.engine.ctx, forward_input, forward_output
        )
        return forward_output
