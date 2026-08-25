# SemanticLogKV 算法规格与实现方案

> 精简版。本文是设计规格；`log_kv_semantic_clusters` 生产路径已按 §5.18 落地在默认关闭
> 的开关后。当前路线跳过 S0.8 CPU reference，直接维护 GPU/torch cache 实现。

## 5. 算法规格与实现方案

### 5.1 参数和硬性互斥

| 参数 | 默认/当前候选 | 含义 |
|---|---|---|
| `log_kv_semantic_clusters` | `false` | 总开关，关闭时必须保持现有行为 |
| `log_kv_cluster_k_max` | 待定 | 簇数上界；32k 先看 `K_max≈64-128`，不要用 binding-only 的 `≈280` 当默认 |
| `B′` / `--b_prime` | Stage 0 默认 `128` | 每簇每层 entry 数；生产前要和 `K_max` 成对定尺 |
| `log_kv_cluster_lambda_rel` | `1.0` 主线，`0.875` 高召回候选 | 新簇阈值系数 |
| `log_kv_seg_eta` | `1.0` | join cost 的时序 tie-break 权重 |
| `log_kv_seg_g0` | `2048` | 时序项饱和尺度 |
| `log_kv_seg_gap_max` | 先看 `inf/8192/4096` | 开新 segment 的间隔阈值 |
| `log_kv_seg_block_level` | `0/1` 优先；生产只允许 `{0,1,2}` | 低层段边界保护深度 |
| `log_kv_seg_forget` | `0.5` | 跨段 centroid 遗忘 |
| `log_kv_lambda` | 现有参数 | mass bias 系数，只控制 `+λ·log(w)` |

语义簇模式必须拒绝：

- `importance_pooling=True`
- `pin_size>0` 或训练期 pin 注入参数非零
- `rope_interleave=True`（当前锚点旋转只覆盖 split-half RoPE）
- `MultiheadLatentAttention`（本方案假设显式 per-KV-group `k/v`）

新增 YAML 字段若需要 CLI 扫参，默认写 `null`。

### 5.2 距离和标定

聚类输入是 **qk-norm 之后、apply_rope 之前**的 key，记作 `k_raw`。距离用平方欧氏：

```text
S(x, c) = ||k_x - μ_c||²
λ_new  = λ_rel · s_h
```

`s_h` 是离线标定常量，按 `(layer, KV group)` 统计整个标定集上的
`E||k-k̄||²`。v1 不做在线估计；在线估计会引入训练/推理轨迹差异和冷启动不稳定。

### 5.3 三路路由

对 flush 出 recent window 的每个 token `x`（绝对位置 `p`）：

```text
D(x,c) = ||k_x - μ_c||² + η · φ(p - p_hi_c)
φ(g)   = g / (g + g0)
c*     = argmin_c D(x,c)

if K_max == 1:
    JOIN 或 NEW_SEGMENT，不触发 novelty
elif ||k_x - μ_c*||² > λ_new:
    NEW_CLUSTER
elif p - p_hi_c* > g_max:
    NEW_SEGMENT
else:
    JOIN
```

`η` 只影响候选排序；新簇判定必须用赢家 `c*` 自己的语义距离，不用全局
`min_c ||k-μ_c||²`。

### 5.4 批量路由

逐 token 串行会把训练拖死。v1 采用三阶段近似：

1. **Phase 1 direct**：冻结当前 centroid 和 `p_hi_c`，矩阵化计算 `D/S`。能直接加入
   既有簇的 token 先批量落地。
2. **Phase 2 orphan**：只对需要开新簇的少数 token 串行处理；`K_max` 满时先 Ward 合并。
3. **Phase 3 metadata**：更新 `current_segment`、`p_hi_c`、`level0_phase` 等持久状态。

S0.8 必须用两份朴素 CPU 参考实现比较：`cache_serial`（严格逐 token）和
`cache_batch`（三阶段近似的朴素版）。它们不是生产向量化实现。

### 5.5 centroid 状态

维护两个计数：

- `n_eff`：浮点，给 centroid 在线均值用；开新 segment 前乘 `γ`。
- `n_total`：整数，真实 token 总数，不含 pad；给 Ward 代价和空间界用。

更新：

```text
开新 segment 前：n_eff <- γ · n_eff
收到成员 k：
  if n_eff == 0: μ <- k
  else:          μ <- (n_eff·μ + k) / (n_eff + 1)
  n_eff   <- n_eff + 1
  n_total <- n_total + 1
```

### 5.6 `K_max`、Ward 和退化边界

默认形式仍是：

```text
K_max_default = max(4, ceil(c · log2(N)))
```

但 `c=1` 只是占位。当前 32k 数据显示：

- binding-only/no-Ward 上界约 `c≈19`，只表示“几乎不触发 Ward”。
- `K_max=64/128` 的 exact-entry 质量已接近 unclipped 平台。
- 下一步先用 `c≈6-8` 附近候选，再用 S0.6 成本裁掉过贵配置。

`K_max` 满时执行 Ward 合并：

```text
cost(a,b) = (n_total_a · n_total_b) / (n_total_a + n_total_b) · ||μ_a - μ_b||²
```

屏蔽对角线和 dead slot；`K_max>=2` 时不存在“找不到可合并对”的 fallback，失败应断言。
`K_max=1` 是单独退化路径：不做 novelty，只在唯一簇上 JOIN/NEW_SEGMENT。

### 5.7 簇内 ladder

每个 cluster 一条 ladder。level 0 存单 token entry（`w=1`），不是现有 LogKV 的
2-token 起步。非顶层超过 `B′` 时合并最老的相邻 entry 并向上一层进位；顶层是饱和
累加器，继续合并最老内容但不丢 entry。

小簇（最终 entry 仍是单真实成员）是 NIAH 机制的承重点：score/value/position 都退回
稠密。

### 5.8 合并算子

```text
w       = w_A + w_B
k_raw   = (w_A·k_A + w_B·k_B) / w
v       = (w_A·v_A + w_B·v_B) / w
p_lo    = min(p_lo_A, p_lo_B)
p_hi    = max(p_hi_A, p_hi_B)
sum_wp  = sum_wp_A + sum_wp_B
Σ/Γ     = 复用现有 Chan-style rank-1 合并，只是统计空间改为 pre-RoPE
```

`compact()` 的“时间序拼接后相邻配对”语义可复用；需要改的是驱动控制流和新增字段。

### 5.9 段边界和 pad

Fenwick compact 没有“跳过这一对”的接口。段边界保护靠 level 0 插入 `w=0` pad，把边界
推到配对边界上。

```text
count = (-level0_phase[cluster]) mod 2^ℓ_block
level = 0
```

规则：

- `PAD_INSERT` 必须写在它服务的 `NEW_SEGMENT` 之前。
- `level0_phase` 是独立持久计数器，统计真实 token+pad 的 level-0 逻辑插入相位。
- pad 的所有张量字段必须显式清零且 finite；不能只设 `w=0`。
- `pad_mask` 只作断言/统计；读出有效性以 `w>0` 和 `slot_valid` 为准。
- `ℓ_block=0` 是关闭保护，`1/2` 是可负担保护，`>=3` 不进生产。

### 5.10 位置模块

`litgpt/log_kv_position.py` 需要提供：

- `merge_anchors(p_lo, p_hi, sum_wp, w)`
- `mid_anchor(sum_wp, w, p_lo, p_hi)`
- `dedup_anchors(lo, mid, hi, w) -> anchors, slot_valid, M`
- `materialize_anchor_keys(k_raw, anchors, cos_cache, sin_cache, rope_n_elem)`
- 同样的 anchor materialization 用于 `sigma_u` / `gamma_a`

`p_mid` 只在读出时算。空 entry 的哨兵锚点不能直接索引 `cos_cache`，必须先 mask 或 clamp。

### 5.11 attention 接口

语义模式下 `get_attention_state()` 返回具名结构 `CacheAttentionState`，不要再靠位置
解包。字段至少包括：

```text
slot_k, slot_v, slot_w,
slot_sigma_u, slot_sigma2, slot_gamma_a, slot_gamma_b, slot_gamma,
slot_valid, M_s
```

`log_kv_slot_attention` 保持现有 `mask`/`causal_tail` 语义，同时新增 pooled 前缀的
`slot_valid` 和 `M_s`：

```text
bias = -log(clamp(M_s, min=1))
if λ != 0:
    bias += λ · log(clamp(w, min=1))
score += bias
score[~slot_valid] = -inf
```

`slot_valid` 只覆盖 pooled 前缀；exact 后缀隐式 `M=1,w=1`。softmax 前必须确认每个
query 至少有一个有效候选，避免整行 `-inf` 产生 NaN。

### 5.12 预算公式

```text
L_alloc = ceil(log2(N / (K_max · B′) + 1)) + δ
```

`δ` 是安全余量。`K_max` 覆盖时必须同步重算 `L_alloc`。生产文档和论文只用最终候选重算
后的预算表；旧 `B′=8` 表不再引用。

### 5.13 buffer 清单

新增持久状态：

```text
centroid, n_eff, n_total, p_hi_c, current_segment,
level0_phase, alive, level_count, pad_mask, s_h
```

训练路径额外分配：

```text
op_log, op_log_len
```

`op_log` 不属于 serving cache；推理路径不分配。

### 5.14 `op_log` 契约

每条 op 是 4 个 int32：

| op | 参数 | 含义 |
|---|---|---|
| `NEW_CLUSTER` | `(slot_idx, 0, token_idx)` | token 建新簇 |
| `NEW_SEGMENT` | `(cluster, new_seg, token_idx)` | 同簇开新段 |
| `JOIN` | `(cluster, segment, token_idx)` | 加入当前段 |
| `WARD_MERGE` | `(keep_slot, free_slot, -1)` | 不消费 token，紧邻服务的 `NEW_CLUSTER` 之前 |
| `PAD_INSERT` | `(cluster, 0, count)` | 不消费 token，紧邻服务的 `NEW_SEGMENT` 之前 |
| `CARRY` | `(cluster, level, resulting_count)` | debug-only，重放不依赖 |

backward 重放只读 `op_log[:op_log_len]`，不重算 centroid、不重算路由。`op_log/op_log_len`
要在每次训练 forward 重新绑定成新张量并存进 `ctx`，不能像普通 buffer 那样原地清零复用。

### 5.15 `litgpt/log_kv_cache.py`

主要改动：

- cache 写入收 `k_raw` 和绝对位置。
- entry buffer 增加锚点、簇元数据和有效位。
- `_binary_carry` 的 host `_counts` 镜像在语义路径删除，改成按 level 静态循环 + 掩码。
- `get_attention_state()` 返回 `CacheAttentionState`。
- `log_kv_slot_attention` 支持 `slot_valid/M_s` 和新的 mass bias。

### 5.16 `litgpt/model.py`

模型侧必须同时持有：

- `k_raw`：qk-norm 后、RoPE 前，写入 semantic cache。
- `k_roped`：RoPE 后，供 in-flight exact attention 和普通 attention 使用。

不要在计算 `k_roped` 前 detach `k_raw`，否则会切断现有梯度链。`k_raw` 进入自定义
Function 后梯度返回 `None`，沿用现有 stop-gradient-through-cache 训练目标。

### 5.17 batching

- 聚类按 `(batch, KV group)` 独立。
- 张量保持矩形：`K_max × L_alloc × B′`，靠 mask 表示空槽。
- 读出时三锚点固定展开是瞬时张量，不是持久 cache。
- ragged/paged/gather 优化不进 v1，等 S0.6 证明值得再做。

### 5.18 实现顺序

0. Stage 0 dump/分析：机制 A 已完成，机制 B 未完成。
1. `log_kv_position.py` + 纯 CPU 单测。
2. `cache_serial` / `cache_batch` 两份朴素 CPU 参考实现，供 S0.8。（当前路线跳过。）
3. `CacheAttentionState` 和 `log_kv_slot_attention` 接口迁移。
4. `K_max=1` 单簇路径。
5. 多簇路由生产向量化。
6. segment padding 和 level 簿记。
7. 训练路径 `op_log` 保存与重放。

### 5.19 易错点

1. 默认关闭必须和现有行为数值等价。
2. `k_raw` 的位置是 qk-norm 后、RoPE 前。
3. `slot_valid` 和 `M_s` 在语义模式下不是可选项。
4. `-log(M)` 不受 `λ` 门控。
5. pad 槽所有字段清零，防止 `0 * NaN`。
6. Ward 只断言合法候选，不写 fallback。
7. `K_max=1` 单独处理。
8. `s_h` 离线标定并写进 eval metadata。
9. `level_w/w` 不能用 fp16/bf16 存大计数。
10. 生产前重算预算，不沿用旧表。
11. 语义路由的批量实现全程不能有 host-device 同步（`.item()`、CUDA 上的
    `nonzero()`/布尔索引）；判定阶段（`_semantic_existing_assignments`）已经是纯张量运算，
    但 `_semantic_join_or_segment`/`_semantic_join`/`_semantic_new_cluster` 这条 Phase 1
    写入路径每次调用仍有多次 `.item()`，比算法复杂度本身更致命。

### 5.20 复用边界

可复用：

- `compact()` 的加权均值核心。
- Chan-style Σ/Γ rank-1 合并数学。
- `log_kv_slot_attention` 的分数/读出骨架。
- GQA 展开、fp32 分数缓冲、`causal_tail`。
- `LogKVStreamTrainingAttention` 的局部重放框架。

必须重写或迁移：

- cache 成员划分和 `_binary_carry` 控制流。
- post-RoPE 存储改为 pre-RoPE 存储 + 锚点物化。
- `get_attention_state()` 所有调用点从位置解包改字段访问。
- 训练重放依据从“重算路由”改为“回放 `op_log`”。

### 5.21 开工前定死

生产开工前必须给出：

1. `lambda_rel`、`K_max:B′`、`g_max`、`ℓ_block` 的第一组候选。
2. 最终候选的持久 entry、读出 slot、训练 `op_log` 三笔预算。
3. `s_h` 标定文件格式和 metadata 字段。
4. S0.8 可接受阈值和失败后的取舍。
5. 是否把 Γ 的 delta-rule 构造纳入 v1；默认不纳入，除非 S0.7 触发。

### 5.22 CPT 训练路径

CPT 不走“dense 训练、semantic LogKV 推理”。训练路径继续基于
`LogKVStreamTrainingAttention`：forward 流式构建 semantic cache，backward 重置 cache
后重放 forward 记录的操作。

必须拆清两层串行：

- chunk/flush batch 之间按 token index 从左到右推进。这和现有 LogKV 一样，不能消掉。
- flush batch 内不能逐 token 全串行。默认实现必须走 §5.4 三阶段：Phase 1 批量处理
  direct token，Phase 3a 立刻批量更新这些 direct token 的 metadata，然后 Phase 2 只
  串行处理 orphan 组，并把 Phase 3b 内联到每个 orphan 主操作之后。

训练和推理共享同一个 `route_and_flush_batch(..., record_op_log: bool)`。本地 op 缓冲
始终构建，用来驱动 Phase 3a/3b 和 ladder 写入；只有 `record_op_log=True` 时，才把本地
缓冲追加进持久 `op_log`。训练入口传 `True`，推理入口传 `False`。

实现顺序约束：

1. flush 调度只由绝对 token index 和 `flush_granularity` 决定，不能依赖
   `train_block`、prefill 分块或 batch 内样本内容。
2. Phase 1 先冻结 `centroid/p_hi_c/current_segment/alive` 快照，矩阵化计算 `D/S`；
   `alive_mask` 排除空簇，`direct_mask` 只允许 `s*[t] <= λ_new` 的 token 落地。
3. Phase 1 的物理 ladder 写入和 Phase 3a metadata 更新必须在 Phase 2 开始前完成；
   Ward 合并必须看到包含本批 direct token 的最新状态。
4. Phase 2 不把 orphan 重新并入既有簇，只在 orphan 集合内部做小 DP-means；`K_max`
   满时先按 `ward_mask` 选择合法合并对，再 `WARD_MERGE`，随后紧邻 `NEW_CLUSTER`。
5. `op_log` 的线性顺序是“Phase 1 direct 组，然后 Phase 2 orphan 组”，不是原始 token
   顺序；每条消费 token 的主操作必须显式带 `token_idx`。
6. 同一逻辑簇内的主操作必须保持真实到达顺序；`PAD_INSERT` 紧邻并先于它服务的
   `NEW_SEGMENT`，`WARD_MERGE` 紧邻并先于它服务的 `NEW_CLUSTER`。
7. attention 读出仍必须使用 `slot_valid/M_s` 和 `causal_tail` mask。mask 是正确性边界，
   不是可选优化。
8. backward 只读 `op_log[:op_log_len]` 重放，不重算 centroid、不重算路由、不依赖
   forward 结束后 cache 里残留的状态。
9. Phase 1 把多个 direct token 写入同一簇 ladder（可能触发多级 carry）时，数值累积和
   结构决策要分开处理：carry 满足结合律，等价于给二进制计数器批量加 K，可以用平衡归并树
   在 O(log K) 深度算完，不需要逐 token 循环（参考 `_flush_pairs` 对按位置路径的批量
   flush 写法）；只有 `NEW_CLUSTER`/`NEW_SEGMENT`/Ward 配对选择这类改变簇集合本身的决策
   必须留在 Phase 2。判定阶段（`_semantic_existing_assignments`）已经这样矩阵化了，Phase 1
   的 ladder 写入这一半还没有。

速度闸门：完整 CPT 前先补一个合成 microbenchmark，固定层数、KV group、batch 和
`flush_granularity`，扫描 orphan 比例与 `K_max` binding 率，分别报告 Phase 1、Phase 2、
Phase 3、`op_log` commit、backward replay 的 wall-clock。若 Phase 2 时间接近随 `T`
线性增长，先修批量路由或缩小候选配置，不进入完整 CPT。当前 `_semantic_route_three_phase`
的判定阶段已矩阵化，但 Phase 1 的 ladder 写入还是逐 token；闸门要在 §5.19 第 11 条（消
同步）和上面第 9 条（批量写入）都做完后跑才有参考价值。
