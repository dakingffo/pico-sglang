from __future__ import annotations

from datetime import timedelta
from typing import Any, NamedTuple, TypeAlias

import torch

from picosgl.layers.attention_backend import create_attention_backend
from picosgl.layers.moe_backend import create_moe_backend
from picosgl.core import Batch, Context, Request, set_global_ctx
from picosgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from picosgl.cache import (
    create_kvcache_pool,
    create_linear_state_pool,
    linear_state_slot_bytes_for_config,
)
from picosgl.layers import set_rope_device
from picosgl.models import create_model, load_weight
from picosgl.utils import (
    align_ceil, div_ceil, div_even, init_logger, is_sm90_supported, is_sm100_supported, torch_dtype
)

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)

# Smallest KV cache we will ever build. When the worst-case MTP state reserve is too
# large to leave room for this many pages, the reserve is capped instead of refusing
# to start (runtime eviction absorbs the deficit).
_MIN_KV_PAGES = 2


Indice2D : TypeAlias = tuple[torch.Tensor, torch.Tensor]

class ForwardInput(NamedTuple):
    batch      : Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)

class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


class VerifyOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event
    full_hidden    : torch.Tensor

ForwardData: TypeAlias = tuple[ForwardInput, ForwardOutput]

class Engine:
    def __init__(self, config: EngineConfig):
        assert not torch.cuda.is_initialized()
        set_tp_info(config.tp_info)
        _adjust_config(config)
        self.config = config

        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        self.model.load_state_dict(self._load_weight_state_dict(config))

        self.num_pages, num_tokens, num_states = self._determine_num_pages(init_free_memory, config)
        # ======================= KV cache initialization ========================
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
        )

        # ======================= Linear attention state pool ========================
        if config.model_config.is_hybrid:
            self.num_states = num_states
            self.ctx.linear_state = self.linear_state = create_linear_state_pool(
                model_config=config.model_config,
                num_slots=self.num_states,
                device=self.device,
                dtype=self.dtype,
            )


        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = align_ceil(self.max_seq_len, 32)
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )
        # ======================= Linear state table ========================
        if config.model_config.is_hybrid:
            page_cols = div_ceil(self.max_seq_len, config.page_size)
            reserve_cols = config.speculative_num_draft_tokens + 1 if config.enable_mtp else 0
            self.ctx.state_table = self.state_table = torch.full(
                (config.max_running_req + 1, page_cols + reserve_cols),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            self.ctx.draft_state = self.draft_state = (
                page_cols if reserve_cols else None
            )

        # ======================= Attention & MoE backend initialization ========================
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        if config.model_config.is_moe:
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

        # ======================= Sampler initialization ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        self.dummy_req = Request(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page

        mtp_skip_graphs = getattr(config, "enable_mtp", False)
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=[] if mtp_skip_graphs else config.cuda_graph_bs,
            cuda_graph_max_bs=0 if mtp_skip_graphs else config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        torch.distributed.init_process_group(
            backend="gloo",
            rank=config.tp_info.rank,
            world_size=config.tp_info.size,
            timeout=timedelta(seconds=config.distributed_timeout),
            init_method=config.distributed_addr,
        )
        tp_cpu_group = torch.distributed.group.WORLD
        assert tp_cpu_group is not None
        max_bytes = (
            config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
        )
        if config.tp_info.size > 1:
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            # Cast each tensor to its parameter's dtype (not blanket self.dtype) so that
            # fp32 params (e.g. GatedDeltaNet A_log / gated norm weight) stay fp32.
            param_dtypes = {k: v.dtype for k, v in self.model.state_dict().items()}
            return {
                k: v.to(param_dtypes.get(k, self.dtype))
                for k, v in load_weight(config.model_path, self.device)
            }

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> tuple[int, int, int | None]:
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = (
            2  # key + value
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
            * config.page_size
            * self.dtype.itemsize
            * config.model_config.num_attention_layers
        )
        # For hybrid models every KV page also owns one linear-state slot, and each MTP
        # request additionally owns a K+1 slot reserve. Both are inside the memory budget.
        reserve_bytes = 0
        reserve_state = 0
        cache_per_state = 0
        if config.model_config.is_hybrid:
            cache_per_state = linear_state_slot_bytes_for_config(config.model_config, self.dtype)
            reserve_per_req = config.speculative_num_draft_tokens + 1 if config.enable_mtp else 0
            reserve_state = config.max_running_req * reserve_per_req
            reserve_bytes = reserve_state * cache_per_state

        num_pages = config.num_page_override
        if num_pages is None:
            model_memory = old_free_memory - new_free_memory
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            if config.model_config.is_hybrid:
                # The K+1 verify reserve is a worst case (max_running_req concurrent MTP
                # requests). Cap it so it can never starve the KV cache below MIN_KV_PAGES;
                # if more requests verify concurrently than slots were reserved, runtime
                # eviction frees page state slots to absorb the deficit.
                min_pages_bytes = _MIN_KV_PAGES * (cache_per_page + cache_per_state)
                reserve_state = min(
                    reserve_state,
                    max(0, (available_memory - min_pages_bytes) // cache_per_state),
                )
                reserve_bytes = reserve_state * cache_per_state
                available_memory -= reserve_bytes
                num_pages = max(
                    _MIN_KV_PAGES,
                    available_memory // (cache_per_page + cache_per_state),
                )
            else:
                num_pages = available_memory // cache_per_page

        assert num_pages > 1, (
            "Not enough memory for KV cache, try reducing --num-pages. "
            f"(hybrid: cache_per_page={mem_GB(cache_per_page)}, cache_per_state="
            f"{mem_GB(cache_per_state)}, reserve_bytes={mem_GB(reserve_bytes)})"
        )
        num_tokens = num_pages * config.page_size
        logger.info(f"Allocating {num_tokens} tokens for KV cache,\
                     K + V = {mem_GB(num_pages * cache_per_page)}")
        if config.model_config.is_hybrid:
            reserve_per_req = config.speculative_num_draft_tokens + 1 if config.enable_mtp else 0
            logger.info(
                f"Allocating {num_pages + reserve_state}"
                f" linear state slots = {mem_GB(num_pages * cache_per_state + reserve_bytes)}"
            )
            return num_pages, num_tokens, num_pages + reserve_state
        else:
            return num_pages, num_tokens, None

    def _sync_get_memory(self) -> tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        return min_free_memory, max_free_memory

    def prepare_batch(self, batch: Batch, cache_manager) -> ForwardInput:
        """Uniform prep for decode / prefill / verify batches -> ForwardInput.

        Order matches the old scheduler._prepare_batch exactly: pad -> page allocation
        (verify skips: its schedule already allocated exactly the missing region) ->
        positions -> input tuple -> write tuple (empty for verify) -> out_loc -> metadata
        -> sampler.prepare.
        """
        self.graph_runner.pad_batch(batch)
        cache_manager.allocate_paged(batch.reqs)
        if not batch.is_verify:  # verify writes to the MTP reserve, not page slots
            cache_manager.allocate_state(batch.reqs)
        batch.positions = self._make_positions(batch, self.device)
        input_mapping = self._make_input_tuple(batch, self.device)
        write_mapping = self._make_write_tuple(batch, self.device)
        batch.out_loc = self.page_table[input_mapping]
        self.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def forward_batch(
        self, batch: Batch, args: BatchSamplingArgs
    ) -> ForwardOutput | VerifyOutput:
        assert torch.cuda.current_stream() == self.stream
        with self.ctx.forward_batch(batch):
            if batch.is_verify or (self.config.enable_mtp and batch.is_prefill):
                hidden, logits = self.model.forward_verify()
                batch.full_hidden = hidden
            elif self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()

        if batch.is_verify:
            next_tokens_gpu = self.sampler.reject_sample(logits, batch, args).to(torch.int32)
            next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
            copy_done_event = torch.cuda.Event()
            copy_done_event.record(self.stream)
            return VerifyOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event, hidden)
        else: # prefill / decode
            next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
            next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
            copy_done_event = torch.cuda.Event()
            copy_done_event.record(self.stream)
            return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()

    @staticmethod
    def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
        needed_size = sum(r.extend_len for r in batch.padded_reqs)
        indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
        offset = 0
        for req in batch.padded_reqs:
            length = req.extend_len
            torch.arange(
                req.cached_len,
                req.device_len,
                dtype=torch.int32,
                out=indices_host[offset: offset + length],
            )
            offset += length
        return indices_host.to(device, non_blocking=True)

    @staticmethod
    def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
        mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
        offset = 0
        for req in batch.padded_reqs:
            length = req.extend_len
            mapping_host[offset: offset + length].fill_(req.table_idx)
            offset += length
        return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)

    @staticmethod
    def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
        # verify writes committed tokens itself (settle/process); the write tuple is unused.
        if batch.is_verify:
            return (
                torch.tensor([], dtype=torch.int64, device=device),
                torch.tensor([], dtype=torch.int64, device=device),
            )
        mapping_list = [req.table_idx for req in batch.reqs]
        mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
        write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
        write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
        return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)

def _adjust_config(config: EngineConfig):
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        object.__setattr__(config, attr, value)

    if config.attention_backend == "auto":
        backend = "trtllm" if is_sm100_supported() else "fa,fi" if is_sm90_supported() else "fi"
        override("attention_backend", backend)
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")

    if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64, 128]:
        override("page_size", 64)
        logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")

    if config.model_config.is_moe and config.moe_backend == "auto":
        override("moe_backend", "fused")
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")

    if config.model_config.is_hybrid:
        if getattr(config, "cache_type", "radix") != "hybrid_radix":
            override("cache_type", "hybrid_radix")
            logger.info_rank0("Prefix cache set to hybrid_radix for hybrid (linear attention) model.")
        if config.page_size % 64 != 0:
            original_page_size = config.page_size
            override("page_size", 64)
            logger.warning_rank0(
                f"Page size overridden to 64 for hybrid model (must be a multiple of 64, got {original_page_size})."
            )
        if config.cuda_graph_max_bs is None:
            override("cuda_graph_max_bs", 0)
            logger.warning_rank0("CUDA graph disabled for hybrid (linear attention) model.")

    if config.enable_mtp:
        assert config.speculative_algorithm == "MTP", (
            "only the MTP algorithm is implemented (got --speculative-algorithm "
            f"{config.speculative_algorithm!r})"
        )
        assert config.speculative_draft_model_path == config.model_path, (
            "--speculative-draft-model-path must equal --model-path under MTP "
            f"(got {config.speculative_draft_model_path!r} vs {config.model_path!r})."
        )
        assert config.model_config.mtp_num_hidden_layers > 0, (
            "MTP speculative decoding requires a model with an MTP head "
            "(mtp_num_hidden_layers > 0)."
        )
        assert config.speculative_num_draft_tokens >= 1, (
            "--speculative-num-draft-tokens must be >= 1"
        )
        if config.enable_dt_separation:
            import torch as _torch

            assert _torch.cuda.device_count() > config.tp_info.size, (
                "--enable-dt-separation requires at least tp_size + 1 GPUs "
                f"(tp_size={config.tp_info.size})."
            )
    elif config.speculative_algorithm == "DFLASH":
        raise NotImplementedError(
            "DFLASH speculative decoding is not implemented yet; "
            "only --speculative-algorithm MTP is supported."
        )
    elif config.enable_dt_separation:
        raise ValueError(
            "--enable-dt-separation requires --speculative-algorithm."
        )
