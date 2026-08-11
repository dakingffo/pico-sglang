from __future__ import annotations

from typing import NoReturn

import torch

from picosgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    ExitMsg,
    UserMsg,
)
from picosgl.utils import init_logger, load_tokenizer
from picosgl.engine import Engine, ForwardOutput

from .ar import ForwardData, ForwardInput
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

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.config = config
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        # The single dispatch point: exactly one AR manager is instantiated by config and
        # the loop never branches on decode vs verify again.
        if config.enable_mtp:
            self.ar_manager = VerifyManager(
                config, self.engine, self.cache_manager, self.table_manager,
                self.eos_token_id, config.num_spec_tokens,
            )
        else:
            self.ar_manager = DecodeManager(
                config, self.engine, self.cache_manager, self.table_manager, self.eos_token_id
            )
        # test-compat aliases: both point at the single AR manager.
        self.decode_manager = self.ar_manager
        self.verify_manager = self.ar_manager
        self.prefill_manager = PrefillManager()
        self.prefill_budget = config.max_extend_tokens

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.

        The AR manager's ``settle`` runs at the start of every schedule (its first line in
        ``_schedule_next_batch``), committing the previous round's verify forward -- pure
        internal accounting. ``process`` still runs at the end of the iteration in
        ``_process_last_data`` and is the only place user-facing DetokenizeMsgs are emitted.
        """
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
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        """User-facing emit stays here, in position, for every phase. The AR manager's
        ``process`` handles verify (reads the settle-stored commit) and non-verify
        (single-token emit / prefill commit) alike."""
        if last_data is None:
            return
        last_data[1].copy_done_event.synchronize()  # verify: same event settle already synced
        reply = self.ar_manager.process(last_data[0], last_data[1])
        self.send_result(reply)

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
            req_to_free = (
                self.prefill_manager.abort_req(msg.uid)
                or self.ar_manager.abort_req(msg.uid)  # verify: also frees mid-round window pages
            )
            if req_to_free is not None:
                self.ar_manager._free_req_resources(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _schedule_next_batch(self) -> ForwardInput | None:
        # 1) First settle the previous verify round's in-flight accounting (decode: no-op).
        #    Must precede drafting: the next round's drafts and page allocation both derive
        #    from this commit. Pure internal -- no user-facing output here.
        self.ar_manager.settle()
        # 2) Prefill has priority, then the AR manager's own batch; reserved leaves room for
        #    the AR manager's in-flight tokens.
        batch = (
            self.prefill_manager.schedule_next_batch(PrefillAdder(
                token_budget=self.prefill_budget,
                reserved_size=self.ar_manager.inflight_tokens,
                cache_manager=self.cache_manager,
                table_manager=self.table_manager,
            ))
            or self.ar_manager.schedule_next_batch()
        )
        # 3) Uniform prep: pad -> page allocation (verify skips: it allocated in schedule) ->
        #    positions -> input tuple -> write tuple (empty for verify) -> metadata -> sampler.
        return self.ar_manager.prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)
        # The next round's input write, explicit here (same as before): decode -> token_pool
        # at each req's committed slot; MTP prefill -> the sampled bonus at token_pool[C].
        # Verify writes committed tokens itself (settle/process), so it is skipped. Writing
        # here -- not in a base after_forward -- means there is no super() chain and no path
        # that calls an empty method and drops the prefill bonus.
        if not batch.is_verify:
            self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        # Pure manager hook: decode -> filter_reqs (incl. the non-MTP prefill->decode
        # handoff); verify -> filter + stash the round's output for the next settle.
        self.ar_manager.after_forward(forward_input, forward_output)
        return forward_output
