# Pico-SGLang

A minimal, lightweight inference framework for Large Language Models, inspired by [mini-sglang](https://github.com/sgl-project/mini-sglang).

---

## 架构总览

### 进程模型

```
Main Process (FastAPI uvicorn)
    │
    ├── Scheduler Process × world_size (每 GPU 一个)
    │       rank 0 → cuda:0
    │       rank 1 → cuda:1
    │       ...
    │
    ├── Tokenizer Process × N (CPU)
    └── Detokenizer Process × 1 (CPU)
```

三大组件通过 ZMQ (IPC socket) 通信，msgpack 序列化消息。

---

### 请求生命周期

```
HTTP Client
    │ POST /v1/chat/completions
    v
API Server (FastAPI)                  Scheduler (主循环)              Engine (GPU)
    │                                       │                           │
    │ TokenizeMsg(uid, text, params)        │                           │
    +───> Tokenizer                         │                           │
              │                              │                           │
              │ UserMsg(uid, input_ids)      │                           │
              +─── ZMQ PUSH ──────────────> receive_msg()               │
                                            │                           │
                                            │ _process_one_msg():       │
                                            │   → PrefillManager        │
                                            │                           │
                                            │ _schedule_next_batch()    │
                                            │ _prepare_batch()          │
                                            │   → allocate pages        │
                                            │   → build positions       │
                                            │                           │
                                            │ _forward() ──────────────>│
                                            │                     model.forward()
                                            │                     sample()
                                            │ <──── ForwardOutput ──────│
                                            │                           │
                                            │ _process_last_data():     │
                                            │   → append next_token     │
                                            │   → DetokenizeMsg         │
              DetokenizeMsg(uid, token, finished)                       │
              <─── ZMQ PUSH ────────────────+                           │
    Detokenizer                               │                           │
        │                                     │                           │
        │ UserReply(uid, text, finished)      │                           │
        v                                     │                           │
    StreamingResponse ───────────────────> HTTP Client
```

---

### 模块依赖图

```
server/  (CLI, API, 进程启动)
    │
    ├── scheduler/  (调度逻辑)
    │       ├── cache.py     CacheManager: 页面分配 + 前缀缓存
    │       ├── table.py     TableManager: 请求槽管理 + token_pool
    │       ├── prefill.py   PrefillManager: 预填充批构建 + chunked prefill
    │       ├── decode.py    DecodeManager: 解码请求集合
    │       └── io.py        SchedulerIOMixin: ZMQ 收发消息
    │
    ├── engine/  (前向传播 + CUDA graph)
    │       ├── engine.py    Engine: 模型持有、forward、KV cache 创建
    │       ├── graph.py     GraphRunner: CUDA graph capture & replay
    │       ├── sample.py    Sampler: greedy / top-k / top-p 采样
    │       └── config.py    EngineConfig
    │
    ├── models/  (模型定义)
    │       ├── qwen3.py, llama.py, ...   HuggingFace 结构等价体
    │       ├── base.py                   BaseLLMModel
    │       ├── config.py                 ModelConfig
    │       ├── weight.py                 权重加载 (safetensors)
    │       └── utils.py                  MLP / MHA 子模块
    │
    ├── layers/  (可学习层)
    │       ├── linear.py     ColumnParallel / RowParallel 线性层
    │       ├── embedding.py  VocabParallelEmbedding + ParallelLMHead
    │       ├── attention.py  MultiHeadAttention
    │       ├── moe.py        MoE 层
    │       ├── norm.py       RMSNorm
    │       ├── rotary.py     RoPE
    │       └── activation.py SiLU, GeLU
    │
    ├── attention/  (注意力后端)
    │       ├── fi.py     FlashInfer 后端 (decode + prefill)
    │       ├── fa.py     FlashAttention 后端 (prefill only)
    │       └── trtllm.py TensorRT-LLM 后端 (Blackwell)
    │
    ├── kvcache/  (KV 缓存管理)
    │       ├── pool.py          MHAKVCache: 物理 KV buffer
    │       ├── base.py          BasePrefixCache 接口 + NamedTuple 类型
    │       └── prefix_cache.py  NaivePrefixCache / RadixPrefixCache
    │
    ├── kernel/  (CUDA/Triton 底层算子)
    │       ├── index.py + csrc/jit/index.cu      Page table gather
    │       ├── store.py + csrc/jit/store.cu      KV cache scatter
    │       ├── radix.py + csrc/src/radix.cpp     Token 前缀比对
    │       ├── moe_impl.py + triton/fused_moe.py Fused MoE kernel
    │       └── utils.py                          TVM JIT/AOT 编译框架
    │
    ├── message/  (进程间通信协议)
    │       ├── backend.py    BaseBackendMsg, UserMsg, ExitMsg
    │       ├── tokenizer.py  TokenizeMsg, DetokenizeMsg
    │       ├── frontend.py   UserReply
    │       └── utils.py      serialize/deserialize (msgpack + __type__ 标记)
    │
    ├── moe/  (MoE 策略层)
    │       ├── base.py   BaseMoeBackend
    │       └── fused.py  fused top-k + moe_align + Triton kernel 调用
    │
    ├── distributed/  (张量并行通信)
    │       ├── info.py   DistributedInfo, TP rank/size
    │       └── impl.py   TorchDistributedImpl / PyNCCLDistributedImpl
    │
    ├── tokenizer/  (tokenize/detokenize 服务)
    │       ├── server.py      ZMQ worker 主循环
    │       ├── tokenize.py    TokenizeManager
    │       └── detokenize.py  DetokenizeManager + 流式输出缓冲
    │
    └── utils/  (工具)
            ├── mp.py          ZmqPushQueue / ZmqPullQueue / ... (ZMQ 封装)
            ├── registry.py    name → factory 注册表 (插件系统)
            ├── logger.py      彩色日志 + rank0 过滤
            ├── hf.py          HuggingFace 配置/权重加载
            ├── torch_utils.py nvtx_annotate 装饰器
            └── misc.py        div_even, align_down 等数学工具
```

---

### 关键数据结构

```python
# 单个请求
Request:
    input_ids: Tensor[int32]    # CPU, 完整 token 序列
    table_idx: int              # page_table / token_pool 的行
    cached_len: int             # KV cache 已缓存的长度
    device_len: int             # 当前 GPU 上的序列长度 (= len(input_ids))
    output_len: int             # 允许生成的 token 数
    cache_handle: BaseCacheHandle  # 前缀缓存句柄
    sampling_params: SamplingParams

# 一个 forward batch
Batch:
    reqs: list[Request]
    phase: "prefill" | "decode"
    input_ids: Tensor[int32]    # GPU, [total_extend_tokens]
    positions: Tensor[int32]    # GPU, [total_extend_tokens]
    out_loc: Tensor[int32]      # GPU, [total_extend_tokens], page table 查出的物理位置
    padded_reqs: list[Request]  # CUDA graph padding 用
```

### 三种内存区域

```
token_pool:  [max_running_req][max_seq_len]  GPU int32   — 每个请求的 token ID
page_table:  [max_running_req][max_seq_len]  GPU int32   — 逻辑位置 → 物理 page 映射
KV Cache:    [2][num_layers][num_pages][page_size][kv_heads][head_dim]  GPU  — K/V 值
```

- **TableManager** 管理 token_pool 的行分配（`allocate()`/`free()`，CPU 端空闲列表）
- **CacheManager** 管理 page_table 的列写入 + 前缀缓存匹配
- **MHAKVCachePool** 管理物理 KV buffer，attention kernel 通过 `out_loc` 索引读写
- **RadixPrefixCache** 构建 token 序列 → page indices 的 radix tree，按 page 边界对齐

---

### 调度循环

```
run_forever():
  loop:
    1. receive_msg()         — 从 tokenizer ZMQ 拉 UserMsg
    2. _schedule_next_batch()  — prefill 优先，否则 decode
    3. _prepare_batch()         — 分配 page、构建 positions/out_loc、attention metadata
    4. _forward()               — engine.forward_batch() → logits → sample
    5. _process_last_data()    — 处理上轮结果：append token、detokenize、发送回复
```

CUDA graph 优化：decode 时 replay 预录的 graph，跳过 Python kernel launch 开销。`pad_batch()` 把实际 batch 补到最近的 graph batch size（如 [1,2,4,8,...,160]）。

---

### Python ↔ C++ 互操作方案（计划中）

调度层全部迁 C++ 的推荐方案：

- **pybind11** — Python ↔ C++ 对象绑定（Scheduler 类、Manager 类）
- **DLPack** — torch tensor 与 C++ GPU buffer 之间零拷贝共享
- **模型 forward 回调** — C++ Scheduler 通过 `std::function` 回调 Python 的 `engine.forward_batch()`
- **GPU 内存由 C++ 管** — cudaMalloc page_table / token_pool / kv_buffer，DLPack 暴露给 torch

参考：
- [pybind11 文档](https://pybind11.readthedocs.io/)
- [DLPack 规范](https://dmlc.github.io/dlpack/latest/)
