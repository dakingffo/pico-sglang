# 阶段计划:linear pool 预算 / speculator 模块抽取 / CLI 参数替换

## Context

三个相互独立又相关的问题,一次规划、分阶段落地:

1. **linear state pool 完全在 memory_ratio 预算之外**。`Engine._determine_num_pages`([engine.py:177-197](python/picosgl/engine/engine.py#L177-L197))只按 `memory_ratio*old_free - model_memory` 定 KV 页数;linear pool 在其后**无条件**分配,不看可用内存。27B 上 `max_running_req=256`、depth=5 → recurrent 约 97GB,直接在 `torch.zeros` 里 OOM(seetacloud 已踩过)。
2. **draft 完全寄生在 VerifyManager 里**。`_draft`([verify.py:222-262](python/picosgl/scheduler/verify.py#L222-L262))是逐 request 的顺序循环;MTP 头挂在主模型上共享 embed/lm_head;无任何 drafter 抽象,DFlash 无法落座,也没有 DT 分离的概念。
3. **CLI 是 `--enable-mtp`/`--num-spec-tokens`**,与算法无关,塞不下 "DFLASH + 独立 draft 模型路径" 的形态。

**用户决定(已确认)**:
- DT 分离:本期只做**抽象支持**(`Drafter.device` 一等概念 + 跨卡传输 helper),MTPDrafter 默认主卡零拷贝;真正的跨卡跑由后续 DFlash 演示。
- **硬门**:批量 draft 重构后,MTP 输出必须与现状**逐字节一致**(`test_mtp_e2e --mtp 1` 前后 JSON diff:tokens / verify_margins / accept_hist / avg_accept / full_commit / one_tok / integrity_ok 全部一致)。非 MTP 保持既有逐字节一致约束。

## 现状事实(已核实)

- `VerifyState`([verify.py:20-27](python/picosgl/scheduler/verify.py#L20-L27))持有 draft_tokens / draft_probs / carry_positions / carry_hidden / mtp_kv / scheduable。
- **`st.mtp_kv` 只在 `_draft` step-0 更新**(`st.mtp_kv = self.mtp.draft(...)[3]`);K-1 步自回归用**局部** `mtp_kv`,`_draft` 返回即弃。所以每轮结束后 `st.mtp_kv` == step-0 整窗 KV,`_update_carry`([verify.py:200-220](python/picosgl/scheduler/verify.py#L200-L220))每轮从**最前端**裁掉 `num_sampled` 行 → 下一轮 step-0 tail 恒为最近 1..K+1 行。
- 批量风险不在 attention(零填充 softmax 数学上可吸收),而在 **GEMM 的 M 从 1/L_i 变成 bs·L_max → cublas 换 kernel/tiling → bf16 低尾位可能翻转 argmax**。因此推荐 **batched feed + per-req attention** 混合方案(见 Part 2.2)。
- `create_linear_state_pool`([cache/__init__.py:47-71](python/picosgl/cache/__init__.py#L47-L71))的 shape 是 linear pool 字节的唯一事实来源。
- 测试复用 `test_mtp_e2e.make_config`(test_mtp_eos.py / test_mtp_abort.py import 它);`test_mtp_detokenize.py` 直接测 DetokenizeManager 不受影响;`test_mtp_rollback.py` 裸建 `LinearStatePool(depth=D)` 不受影响。

## Phase 0 — baseline(先做,作为字节一致门参考)

在 HEAD 上跑:
- `python tests/mtp/test_mtp_e2e.py --mtp 1 --out /tmp/mtp_before.json` 和 `--mtp 0 --out /tmp/non_mtp_before.json`。
- `test_mtp_eos.py`、`test_mtp_abort.py`、`test_mtp_rollback.py`、`test_nonmtp_byte_identity.py` 记录 PASS。
- 新增 `tests/mtp/test_mtp_byte_gate.py`:逐 key 严格 diff 两个 MTP JSON(tokens / verify_margins / accept_hist / avg_accept / full_commit / one_tok / integrity_ok / rounds / num_pages / free_pages / missing),任一不同 exit 1。**这是批量 draft 取舍的裁判**(engine 固定 `torch.manual_seed(42)`、greedy draft 是 argmax,确定性可复现)。

## Part 1 — linear pool 进入 memory_ratio 预算

### [cache/linear/state_pool.py](python/picosgl/cache/linear/state_pool.py)
新增单一事实来源的字节函数(模块级):
```python
def linear_state_pool_size_bytes(
    num_linear_layers, max_req, num_key_heads, key_head_dim,
    num_value_heads, value_head_dim, conv_kernel_dim, depth, dtype,
) -> int:
    conv_dim = num_key_heads * key_head_dim * 2 + num_value_heads * value_head_dim
    slots = max_req + 1
    conv = num_linear_layers * depth * slots * conv_dim * (conv_kernel_dim - 1)
    recurrent = num_linear_layers * depth * slots * num_value_heads * key_head_dim * value_head_dim
    return (conv + recurrent) * dtype.itemsize
```
`create_linear_state_pool` 的 `conv_dim` 公式保持现状(复用同一公式)。`cache/__init__.py` 导出该函数。

### [engine/engine.py:177-197](python/picosgl/engine/engine.py#L177-L197)
`num_pages is None` 分支里先扣 linear pool:
```python
linear_pool_bytes = 0
if config.model_config.is_hybrid:
    depth = (config.speculative_num_draft_tokens + 1) if config.enable_mtp else 1
    linear_pool_bytes = linear_state_pool_size_bytes(
        config.model_config.num_linear_layers, config.max_running_req,
        config.model_config.linear_num_key_heads, config.model_config.linear_key_head_dim,
        config.model_config.linear_num_value_heads, config.model_config.linear_value_head_dim,
        config.model_config.linear_conv_kernel_dim, depth, self.dtype)
available_memory = int(config.memory_ratio * old_free_memory) - model_memory - linear_pool_bytes
```
`_determine_num_pages`(line 75)先于 pool 分配(line 85)执行 → 预算正确。line 193 的 assert 改提示:`"Not enough memory for KV cache (after reserving the linear-state pool). Reduce --max-running-requests (the linear pool is sized depth x max_req) or --num-pages."`。**27B 上 max_req=256 会因预算为负而在启动期 assert(而非 torch.zeros OOM)——这正是期望行为,提示用户降 max_req。**

## Part 3 — CLI/config 改名(Part 2 factory 的前置)

### [engine/config.py:17-34](python/picosgl/engine/config.py#L17-L34)
删两个字段,加三个:
```python
speculative_algorithm        : str | None   = None   # "MTP" / "DFLASH" / None
speculative_draft_model_path : str | None   = None   # DFLASH 独立 draft 模型路径(MTP 不用)
speculative_num_draft_tokens : int          = 4
```
派生 property(frozen dataclass 可用):
```python
@property
def enable_mtp(self) -> bool:
    return self.speculative_algorithm is not None
```
这样 engine.py:91/131/247/340 和 scheduler.py:48 的 `config.enable_mtp` 读取零改动。

### [server/args.py:190-202](python/picosgl/server/args.py#L190-L202)
删 `--enable-mtp`、`--num-spec-tokens`,加:
```python
parser.add_argument("--speculative-algorithm", type=str, default=None,
    choices=["MTP", "DFLASH"], help="Speculative decoding algorithm (MTP / DFLASH).")
parser.add_argument("--speculative-draft-model-path", type=str, default=None,
    help="Path to the draft model weights (DFLASH).")
parser.add_argument("--speculative-num-draft-tokens", type=int,
    default=ServerArgs.speculative_num_draft_tokens,
    help="Number of speculative draft tokens (K) per verify round.")
```

### [engine/engine.py](python/picosgl/engine/engine.py)
- line 92:`num_spec_tokens=config.speculative_num_draft_tokens`(line 91 `enable_mtp=config.enable_mtp` 走 property 不变)。
- `_adjust_config`(line 340-344)改成 MTP-only + DFLASH stub:
```python
if config.enable_mtp:
    if config.speculative_algorithm == "MTP":
        assert config.model_config.mtp_num_hidden_layers > 0, (
            "--speculative-algorithm MTP requires a model with an MTP head (mtp_num_hidden_layers > 0).")
        assert config.speculative_num_draft_tokens >= 1, "--speculative-num-draft-tokens must be >= 1"
    elif config.speculative_algorithm == "DFLASH":
        raise NotImplementedError("DFLASH speculative decoding is not implemented yet")
    else:
        raise ValueError(f"Unknown speculative algorithm: {config.speculative_algorithm}")
```

### [scheduler/scheduler.py:48-59](python/picosgl/scheduler/scheduler.py#L48-L59)
line 52 `config.num_spec_tokens` → `config.speculative_num_draft_tokens`(Part 2 会整体换成 drafter,届时这行消失)。Part 3 阶段 VerifyManager 仍收 mtp。

### [tests/mtp/test_mtp_e2e.py:74-87](python/picosgl/tests/mtp/test_mtp_e2e.py#L74-L87)
`make_config` kwargs:`enable_mtp=...` → `speculative_algorithm=("MTP" if enable_mtp else None)`,`num_spec_tokens=K` → `speculative_num_draft_tokens=K`。覆盖 eos/abort 两个 import 者。`result["enable_mtp"]` JSON key 保留不动。

## Part 2 — `picosgl/speculator` 模块(核心)

### 2.1 模块布局 + 接口

新包 `python/picosgl/speculator/`:
- **`base.py`**:`DraftState`(ABC)、`Drafter`(ABC)、`DraftTarget`(Protocol)、跨卡传输 helper。
- **`mtp.py`**:`MTPDraftState`、`MTPDrafter`(含批量 draft + 逐字节应急路径)。
- **`dflash.py`**:`DFlashDrafter` 骨架,构造即 `raise NotImplementedError`(`_adjust_config` 已在 config 层拦截,此处纯占位死代码)。
- **`__init__.py`**:`create_speculator` 工厂。

接口(遵循用户 VerifyState 草图;用 `DraftTarget` Protocol 避免 speculator → verify 的循环 import):
```python
# base.py
class DraftState(ABC):
    """多态 drafter 状态。MTP 子类持 carry_positions / carry_hidden / mtp_kv。"""

class DraftTarget(Protocol):
    draft_tokens: list[int]
    draft_probs: torch.Tensor | None
    draft_state: DraftState

class Drafter(ABC):
    num_spec_tokens: int
    vocab_size: int
    device: torch.device   # 可异于主卡(DT 分离);MTP 用主卡零拷贝
    def init_state(self, req, full_hidden, mapping, C) -> DraftState      # prefill 后 seed carry
    def draft(self, reqs, targets: list[DraftTarget]) -> list[int]        # 批量;填 targets[i].draft_*;返回每 req n_drafts
    def update_carry(self, st: DraftState, full_hidden, row_start, C, num_sampled) -> None
```
跨卡 helper(`to_main_device` / `from_main_device`)对 MTP 是恒等;存在是为让未来 DT draft 目标只碰传输边界。`draft` 产出 host `int` 的 draft_tokens(现状如此)、draft_probs 在 `drafter.device` 上。

`MTPDrafter` 构造:`(mtp, sampler, device, token_pool, num_spec_tokens, window_size=128)`。`token_pool` 用于读 carry tokens,`sampler.draft_token`/`_target_dist` 做 draft 侧采样。`window_size` 从 VerifyManager 移入 drafter。

### 2.2 批量 draft 算法(Design B:batched feed + per-req attention)

`MTPDrafter.draft(reqs, targets)`:
- 每 req 前奏(与今天 `_draft` 头一致):`n_drafts_i = min(K, remain-1) if remain>0 else 0`;重置 `draft_tokens=[]`、`draft_probs = zeros(K,vocab) if sampling else None`;`n_drafts_i==0` 跳过。

**Step-0(carry materialization)**:
1. 每 active req:`start_i = 0 if mtp_kv is None else mtp_kv[0].shape[1]`;tail = `carry_positions[start_i:]`(长度 `L_i`:首轮整窗,之后恒 1..K+1);取 `carry_tok_i`、`carry_pos_i`、`carry_hidden_i`。
2. 尾部 padding 到 `L_max`;组 `(bs, L_max)` input_ids/positions + `(bs, L_max, hidden)`(垃圾行值随意,只用 `[0:L_i)` 行)。
3. **批量** feed:`embed → pre_fc_norm_embedding → pre_fc_norm_hidden → cat → fc`(在 `(bs*L_max, hidden)` 上)。
4. **批量** `input_layernorm → q/k/v_proj → q_norm/k_norm → rotary`(positions 摊平;垃圾行算出即弃)。
5. **per-req attention**:对 req i 切 `[i*L_max : i*L_max+L_i)` 的 q/k/v/gate,调 `layer.self_attn._eager_attention(q, k, v, past_k, past_v)`(past 取 `st.mtp_kv` 或 None),写回 `st.mtp_kv = (k, v)`;gate 逐元素乘;注意力输出写回 padding buffer 的 `[0:L_i)` 行。
6. **批量** `o_proj → residual(add) → post_attention_layernorm → mlp → residual → norm → lm_head`(在 `(bs*L_max, hidden)` 上)。
7. 每 req:切 `logits[i*L_max+L_i-1]` → `draft_0 = sampler.draft_token(...)`;sampling 时 `draft_probs[0]`;保留 `mtp_hidden_i` 和 `first_pos_i = carry_positions[-1]`。

**Steps j in 1..K-1**(active = `n_drafts_i > j` 的 req,每 req 恰 1 行):
1. `input_ids=[draft_{j-1}]`、`positions=[first_pos_i+j]`、`hidden=stack(mtp_hidden_i)`。
2. **批量** `embed → pre_fc_norms → fc → input_layernorm → qkv → q/k_norm → rotary`(`(bs_j, hidden)`)。
3. **per-req attention**:`past_kv = cat([st.mtp_kv[0]] + [kv[0] for kv in draft_kv[i]], dim=1)`(v 同理),其中 `draft_kv[i]` 是本 `draft` 调用内的**临时**逐步 KV 列表(等价今天被丢弃的局部 `mtp_kv`);返回的 `(k,v)` 追加进 `draft_kv[i]`。
4. **批量** `o_proj → residual → post_ln → mlp → residual → norm → lm_head`。
5. 每 req:`draft_j = sampler.draft_token(...)`;sampling → `draft_probs[j]`;更新 `mtp_hidden_i`。
6. `draft_kv[i]` 随 `draft` 返回丢弃。**`st.mtp_kv` 只在 step-0 更新,与今天完全一致**。

KV cat 顺序/内容与今天逐 request 的 `forward_with_kv` 累积完全一致(cat 精确、无算术);`init_state`/`update_carry` 从 verify 逐字搬移(`mtp_kv` 前端裁剪按 `reserved_len` 不变)。

### 2.3 字节一致分析与应急

**结构上可证明一致的部分**:所有非 GEMM 算子都是行独立的——RMSNorm(沿最后一维 mean)、silu/sigmoid、rotary(逐行 cos/sin)、**per-req attention 保持今天的精确 shape `(heads, T_i, L_i)`**(Design B 不做 block-diagonal padding,零填充 softmax 的担忧不存在)、per-req softmax。任何算子都没有跨行交互。

**唯一风险**:批量 GEMM(`fc`/qkv/o_proj/mlp/lm_head)的 M 从 `L_i`(或 `1`)变 `ΣL_i`(或 `bs_j`)。bf16 输入 + fp32 累加下,cublas 可能为 M=1 vs M>1 换 kernel/tiling → 逐元素 fp32 累加顺序变 → bf16 低尾位翻 → 近并列处 argmax 变 → draft 变 → verify 窗口内容变 → verify_margins/accept_hist 变。**因此批量 MTP draft 不能由构造保证逐字节一致,只能经验验证。**

**应急(门驱动)**:`MTPDrafter(byte_identity: bool)`。`True` 时 `draft()` 逐 req 循环、调用今天精确 shape(`verify._draft` 的纯搬移,构造上一致);`False` 用 2.2 的批量路径。默认值由 Phase 0 的 `test_mtp_byte_gate.py` 决定:
- 批量路径跑 `--mtp 1`,gate PASS → 默认 `False`(批量常驻),文档记录 GEMM kernel 风险与已观测差异。
- gate FAIL → 默认 `True`(逐 req 回退),记录是哪些 shape 破坏一致;批量路径保留在 flag 后供未来做"逐 GEMM 保形"优化。

### 2.4 VerifyManager 重构([verify.py](python/picosgl/scheduler/verify.py))

- `VerifyState`(line 20-27)→ `draft_tokens`/`draft_probs` 留在 VerifyState(用户草图),`carry_positions`/`carry_hidden`/`mtp_kv` 移入 `MTPDraftState`,加 `draft_state: DraftState` 字段。
- `__init__`(line 30-49)→ `(config, device, cache_manager, table_manager, eos_token_id, drafter)`,去掉 sampler/mtp/num_spec_tokens/window_size;`self.drafter = drafter`、`self.num_spec_tokens = drafter.num_spec_tokens`、`self.vocab_size = drafter.vocab_size`。
- `schedule_next_batch`(line 81-119):收集 scheduable reqs+states,`n_drafts_list = self.drafter.draft(reqs, [st.draft_state for st in states])` 一次批量调用;其余(per-req `scheduable=False`、`device_len = C+n_drafts+1`、token_pool 落位、`batch.draft_tokens/probs` 组装)原样保留。
- `process`(line 121-198):commit/reject 逻辑不变;line 164 `self._update_carry(st, ...)` → `self.drafter.update_carry(st.draft_state, full_hidden, row_start, C, num_sampled)`。
- `on_prefill_done`(line 66-79):`req.complete_n(1)` 保留;内联 carry 构建 → `st = VerifyState(draft_tokens=[], draft_probs=None, draft_state=self.drafter.init_state(req, full_hidden, mapping, C))`。
- `abort_req`/`remove_req`(line 51-64):不动(pop state_table 即可,DraftState 随 GC 释放)。
- **删除** `_draft`(line 222-262)和 `_update_carry`(line 200-220),移入 `MTPDrafter`。

### 2.5 scheduler.py 接线

```python
from picosgl.speculator import create_speculator
...
self.speculator = create_speculator(
    config, self.engine.model, self.engine.sampler, self.device, self.token_pool,
    window_size=128,
)
if self.speculator is not None:
    self.ar_manager = VerifyManager(config, self.device, self.cache_manager,
                                    self.table_manager, self.eos_token_id, self.speculator)
else:
    self.ar_manager = DecodeManager(config, self.device, self.cache_manager,
                                    self.table_manager, self.eos_token_id)
```
`engine.py:131`(`getattr(config, "enable_mtp", False)`)、`engine.py:247` 走 property 不变。

## 文件清单

**新增**:`python/picosgl/speculator/{__init__,base,mtp,dflash}.py`、`tests/mtp/test_mtp_byte_gate.py`、`tests/mtp/test_linear_pool_bytes.py`、`tests/mtp/test_batched_draft.py`。

**修改**:
- `python/picosgl/cache/linear/state_pool.py`(字节函数)
- `python/picosgl/cache/__init__.py`(导出 + 复用 dims)
- `python/picosgl/engine/engine.py`(Part 1 预算;Part 3 字段/断言)
- `python/picosgl/engine/config.py`(Part 3 字段 + property)
- `python/picosgl/server/args.py`(Part 3 CLI)
- `python/picosgl/scheduler/scheduler.py`(Part 3 line 52;Part 2 接线)
- `python/picosgl/scheduler/verify.py`(Part 2 核心重构)
- `tests/mtp/test_mtp_e2e.py`(make_config kwargs)

## 验证

1. **Part 1 单元** `test_linear_pool_bytes.py`:Qwen3.5 ModelConfig 下,`linear_state_pool_size_bytes` 等于真实 `LinearStatePool` 的 `sum(t.numel()*t.element_size())`(depth=1 和 depth=K+1);`_determine_num_pages` 对非 hybrid 配置前后页数一致(linear 字节 == 0)。
2. **批量 vs 逐 req 单元** `test_batched_draft.py`:载 Qwen3.5-0.8B + MTP 头,构造 2-3 个合成 carry 的 Request(变长 `L_i`、含 `mtp_kv=None`),`MTPDrafter.draft` 两模式(`byte_identity=True/False`)跑,断言 draft_tokens/probs/mtp_kv/carry 一致。这是 GEMM kernel 问题最快的反馈环。
3. **硬门**:Phase 0 baseline vs 新 `--mtp 1`,`test_mtp_byte_gate.py` diff。决策 `byte_identity` 默认值(见 2.3)。
4. **非 MTP 回归**:`--mtp 0` 重跑 `test_nonmtp_byte_identity.py`(Part 2 不碰非 MTP 路径;Part 1/3 不得扰动它)+ `--compare` 报告 greedy 一致性。
5. **回归**:`test_mtp_eos.py`、`test_mtp_abort.py`(经更新 make_config)、`test_mtp_rollback.py`、`test_mtp_detokenize.py`。
6. **DT 就绪检查**:`create_speculator` 对 "MTP"/None 返回 drafter、对 "DFLASH" 在 `_adjust_config` 抛 NotImplementedError;跨卡 helper 以恒等执行。
7. **27B 现场**(seetacloud):默认 `max_running_req=256` 应启动期 assert(提示降 max_req)而非 OOM;`--max-running-requests 8` 正常工作。

## 构建顺序

1. **Phase 0**(baseline + gate 脚本)→ 2. **Part 1**(独立;先于 Part 3 避免对着改名后的字段写,且两者都动 engine.py 相邻行)→ 3. **Part 3**(行为保持、可独立验证)→ 4. **Part 2**(依赖 Part 3 的 config 面 + factory 调用点)→ 5. **全量验证 + byte_identity 取舍决策**。

## 不在本期范围

- DFlash 的实际实现(扩散模型推理、独立 draft 模型加载)——只留接口 + stub。
- MTP drafter 跑在另一张卡(权重拷贝 + 跨卡传输)——DT 抽象就绪,具体落地留给未来。
- linear pool 的 TP 分片(每 rank 只持本卡 shard)——当前每 rank 全尺寸,预算也按全尺寸算,一致但浪费,留作后续。
- verify 前向(main model)里线性层 per-token 顺序循环的 chunked 融合——独立优化。
