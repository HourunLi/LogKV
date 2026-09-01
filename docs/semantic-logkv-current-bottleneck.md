# SemanticLogKV 当前最大瓶颈：推理不是训练结束就会自动变快

> 更新于 2026-08-28。本文记录当前聊天里对 SemanticLogKV 慢的判断，重点解释：
> 为什么已经拿到 logKV checkpoint 后，语义簇推理仍然慢；哪些代码路径是真正瓶颈；
> Phase 1 / Phase 2 到底是什么意思；下一步应该先修什么。

## 1. 一句话结论

当前最大问题不是 checkpoint，也不是训练还没有收敛，而是 **SemanticLogKV 的运行时 cache
构建方式太重**：

1. 路由判定有一部分已经矩阵化并行。
2. 但 token 真正写入语义簇、更新簇状态、写 slot、Fenwick ladder 进位合并，仍然是逐
   token Python 循环。
3. 读出 attention state 时又按 `K_max * L_alloc * B_prime` 的矩形布局全量物化，并把每个
   entry 展成最多 3 个 anchor slot，很多无效 slot 也先参与大 matmul，再被 mask。
4. 当前评测配置 `K_max=64, B_prime=128, prefill_block=128, second_order_scale=0.2` 会把
   读出宽度和 fp32 score buffer 放大到比 32K dense 更重的量级。

所以“训练完了，只推理”并不能自动快。每条新 prompt 的语义簇都要在线构建，checkpoint
只保存模型权重，不会预先保存未知输入的语义簇结构。

## 2. 和前面质量问题的关系

此前 LogKV 这条线已经遇到过两个质量问题：

1. 朴素均值池化在长程检索上损失很大，NIAH 这类任务掉得明显。
2. 二阶修正有正收益，但 pin 在当前训练/推理分布下是负收益。

这些是“模型能不能用好压缩 cache”的问题。当前这份文档记录的是另一类更靠底层的问题：

```text
就算质量能接受，当前 semantic cache 的运行时也太慢。
```

二者有关，但不是同一个瓶颈：

- checkpoint / CPT 可以改善模型适应压缩后的表示。
- pin 选择、二阶修正、`s_h` 标定会影响语义簇质量。
- 但在线路由、slot 写入、attention state 物化、manual attention matmul 的成本，必须靠
  runtime 实现和配置定尺解决。

换句话说，质量问题靠训练和机制设计；速度问题靠实现路径和预算控制。当前最大痛点是后者。

## 3. 当前慢在哪些路径

### 3.1 推理 prefill 也走 logKV streaming cache

推理时，`litgpt/model.py` 里 `input_pos is not None` 且 cache 是 `LogStructuredKVCache`
时，会走：

```text
Attention.forward
  -> _log_kv_training_forward(... reset_cache=False, defer_last_single=True)
```

相关位置：

- `litgpt/model.py:929-936`：推理使用同一个 streaming chunker 做 prefill 和 decode。
- `litgpt/model.py:1372-1465`：prefill 按 `log_kv_prefill_block` 分块，每块先读 cache
  state 做 attention，再 `cache.add_recent(...)` 写入 cache。

这意味着长 prompt 的 prefill 不是一次 dense FlashAttention 完事，而是每层、每个 prefill
block 都要反复：

```text
get_attention_state
append current exact block
log_kv_slot_attention
add_recent
flush/route/write semantic cache
```

### 3.2 训练和推理共享语义 cache 写入逻辑

训练低显存路径在 `LogKVStreamTrainingAttention` 里：

- `litgpt/log_kv_cache.py:3268-3308`：forward no_grad 流式构建 cache。
- `litgpt/log_kv_cache.py:3270-3272`：`K_max > 1` 的语义模式会记录 `op_log`。
- `litgpt/log_kv_cache.py:3372-3379`：backward reset cache 后按 `op_log` 回放。

推理路径在 `model.py:_log_kv_training_forward` 里调用同一套 `cache.add_recent`。

因此只要打开 `log_kv_semantic_clusters=true`，训练和推理都会碰到同一条在线建簇写入路径。
区别是：训练为了 backward 确定性还要记录/回放 `op_log`；推理不需要 backward，但仍要在线
route 和写 cache。

## 4. 写入路径实际发生了什么

主路径是：

```text
add_recent
  -> _semantic_add_recent
  -> _semantic_flush_tokens
  -> route_and_flush_batch
  -> _semantic_route_three_phase
```

相关位置：

- `litgpt/log_kv_cache.py:2113-2155`：`_semantic_add_recent` 把新 token 放进 recent buffer。
- `litgpt/log_kv_cache.py:2068-2112`：`_semantic_flush_tokens` 从 recent window 头部 flush。
- `litgpt/log_kv_cache.py:1511-1587`：`route_and_flush_batch` 进入语义路由。
- `litgpt/log_kv_cache.py:1431-1509`：`_semantic_route_three_phase`。

flush 触发规则不是“每 `recent_size` 个 token 固定路由一次”，而是：

```text
recent_count + n > recent_size 时产生 overflow
flush_len = ceil(overflow / semantic_flush_granularity) * semantic_flush_granularity
```

当前 eval 配置里：

```yaml
log_kv_recent_size: 1024
log_kv_semantic_flush_granularity: 1024
log_kv_prefill_block: 128
```

所以看起来像是 recent window 满后，每多来约 1024 个 token，就 flush/route 一批 1024 个旧
token。训练配置里如果 `recent_size=4096, flush_granularity=1024, train_block=1024`，则前
4 个 block 只是填 recent，之后基本每个 1024-token block 都会 flush 1024 个 token。

## 5. `_semantic_existing_assignments` 是训练专用吗

不是。

`_semantic_existing_assignments` 是训练和推理共享的路由判定函数，位置在
`litgpt/log_kv_cache.py:1343-1370`。它做的是：

```text
对一个 flush batch 的所有 token：
  计算 token 到每个 live cluster centroid 的语义距离
  加 temporal gap cost
  如果有 capacity penalty 也加进去
  对 cluster 维度 argmin 得到 winner
  判断 winner 是否足够近，得到 direct=True/False
```

这部分是 GPU tensor 并行的，形状大致是：

```text
[batch, kv_group, flush_tokens, K_max]
```

但它当前有两个问题：

1. 它用 `diff = k_raw.unsqueeze(3) - centroid.unsqueeze(2)` 显式构造
   `[B, G, T_flush, K, D]` 的 fp32 临时张量。`K=64, T_flush=1024, G=8, D=128` 时，单层这个
   `diff` 约 67M 个 fp32 元素，约 256 MiB，还没算 `square/sum/cost` 的中间结果。
2. 判定结束后马上 `winner.detach().cpu().tolist()` 和 `direct.detach().cpu().tolist()`，
   把结果拉回 CPU，后续进入 Python 循环。

所以它只是“判定阶段并行”，不是“整个语义簇更新并行”。

## 6. 现在到底哪些是并行的

| 环节 | 当前是否并行 | 说明 |
|---|---|---|
| `_semantic_existing_assignments` 路由判定 | 是 | GPU tensor 计算整批 token 到 live clusters 的距离和 winner/direct。 |
| Phase 1 direct token 筛选 | 半并行 | winner/direct 来自并行结果，但随后转 CPU list。 |
| Phase 1 direct 写入已有簇 | 否 | Python 逐 token 调 `_semantic_join_or_segment`，每个 token 单独写 ladder。 |
| Phase 2 orphan 处理 | 基本否 | 新簇、加入本批新簇、Ward merge 都有顺序依赖。 |
| Phase 3 metadata 更新 | 否 | 当前内联在每个 token 的 join/new_cluster/new_segment 里。 |
| `_semantic_attention_state` 物化 | GPU tensor 操作 | 算子本身并行，但矩形全量物化，算得太多。 |
| `log_kv_slot_attention` | GPU matmul | matmul 并行，但 slot 数过大，且不是 FlashAttention。 |

最短判断：

```text
路由判定并行；
写入语义簇不并行；
读出 attention 并行但过量。
```

这就是“看起来有 batch routing，但实际推理还是慢”的根本矛盾。

## 7. Phase 1 / Phase 2 是什么意思

当前函数名叫 `_semantic_route_three_phase`，但实现里最清楚的边界是：

```text
判定阶段：_semantic_existing_assignments
Phase 1：direct token 加入已有簇
Phase 2：orphan token 新建簇 / 加入本批新簇 / Ward merge 后腾簇
Phase 3：metadata 更新，设计文档里单独列出，但当前实现大多内联在 join/new_cluster 里
```

### 7.1 判定阶段：并行决定 winner/direct

这一步冻结当前的 `centroid/p_hi_c/alive/n_total` 等状态，对整批 flush token 算：

```text
winner = 当前最合适的已有簇
direct = 是否可以直接加入这个已有簇
```

它不真正修改 cache。

### 7.2 Phase 1：direct token

Phase 1 处理 `direct=True` 的 token。

含义：这个 token 离某个已有语义簇足够近，所以直接加入已有簇。

当前代码在 `litgpt/log_kv_cache.py:1456-1474`：

```text
for b:
  for g:
    idx = 当前 batch/group 里所有 direct token
    for i in idx:
      _semantic_join_or_segment(...)
```

所以 Phase 1 的语义是“direct 加旧簇”，但当前实现方式仍然是逐 token Python 循环。

每个 token 会继续调用：

```text
_semantic_join_or_segment
  -> _semantic_join / _semantic_new_segment
  -> _semantic_append_token_entry
  -> _semantic_append_entry
  -> _semantic_write_slot / _semantic_clear_slot
```

这就是现在最不理想的地方：Phase 1 理论上最适合批量化，因为这些 token 的目标簇已经由
冻结快照判定好了；但目前没有批量写入 ladder。

### 7.3 Phase 2：orphan token

Phase 2 处理 `direct=False` 的 token。

含义：这个 token 离已有簇都不够近，属于 novelty/orphan。它会按顺序尝试：

1. 如果本 flush batch 前面已经创建了 orphan cluster，先看能不能加入这些新簇。
2. 如果不能加入，就找 free cluster 开新簇。
3. 如果没有 free cluster，就先做 Ward merge，把两个已有簇合并，腾出一个 cluster，再开
   新簇。

当前代码在 `litgpt/log_kv_cache.py:1476-1509`。

Phase 2 比 Phase 1 更难并行，因为第 `i` 个 orphan 的动作会改变第 `i+1` 个 orphan 看到的
簇集合。例如前一个 orphan 新建了簇，后一个 orphan 可能就应该加入它；或者 Ward merge 后
簇编号和 centroid 都变了。

所以 Phase 2 有一部分串行是合理的。真正不合理的是：Phase 1 这种 direct 写入也还在逐
token 串行。

### 7.4 Phase 3：metadata

算法规格里 Phase 3 是更新：

```text
current_segment
p_hi_c
level0_phase
n_eff / n_total
centroid
```

当前实现没有一个独立的 Phase 3 函数，而是在这些函数里内联更新：

- `_semantic_new_cluster`
- `_semantic_join`
- `_semantic_new_segment`
- `_semantic_update_member_metadata`
- `_set_semantic_*`

所以从代码执行看，Phase 3 不是一个单独的批处理阶段，而是夹在 Phase 1/Phase 2 的每个
token 操作里。这也加重了 Python 控制流和小 CUDA op 的开销。

## 8. K=1 为什么也不代表快

`K_max == 1` 时，`_semantic_route_three_phase` 直接走特殊路径：

```text
_semantic_route_k1_batch
```

位置在 `litgpt/log_kv_cache.py:1440-1442` 和 `litgpt/log_kv_cache.py:1410-1429`。

这条路径没有必要算 token 到多个簇的 winner，因为只有一个簇。但它仍然是：

```text
for batch
  for group
    for token
      new_cluster 或 join_or_segment
```

所以 K=1 省掉了路由距离矩阵，但没有省掉逐 token 写 slot、ladder carry、metadata 更新。
它可能比 K64 少一部分判定开销，但不是一个真正高吞吐的实现。

## 9. 每个 token 写入语义簇为什么慢

单个 token 加入簇后，会进入 `_semantic_append_entry`，位置在
`litgpt/log_kv_cache.py:997-1018`。

这个函数维护每个 cluster 自己的 Fenwick-style ladder：

```text
从 level 0 开始：
  如果当前 level 还有空位：
    写入 slot，结束
  如果当前 level 满了：
    取最老两个 entry
    compact/merge 成一个更宽 entry
    shift 剩余 slots
    把 merged entry carry 到下一层
```

它内部会频繁调用：

- `_semantic_write_slot`：`litgpt/log_kv_cache.py:890-913`
- `_semantic_clear_slot`：`litgpt/log_kv_cache.py:874-888`
- `_semantic_slot_entry`：`litgpt/log_kv_cache.py:915-930`
- `_semantic_merge_entries`：`litgpt/log_kv_cache.py:932-974`
- `_semantic_shift_after_oldest_pair_merge`：`litgpt/log_kv_cache.py:976-995`

问题不只是算法有循环，而是这些循环里有很多很小的 GPU tensor copy/zero，以及 `.item()`、
`.tolist()` 这类 host-device 同步。GPU 不擅长被 Python 一下一下喂很小的操作；长 prompt
下每层都这么做，墙钟时间会非常差。

## 10. 读出路径也很重

即使写入修好，当前读出也会很贵。

`_semantic_attention_state` 在 `litgpt/log_kv_cache.py:2386-2486`。它会把语义 cache 的矩形
buffer 展平成 attention slots：

```text
entry 数 = K_max * L_alloc * B_prime
每个 entry 最多展开成 p_lo / mid / p_hi 三个 anchor
slot 数约 = 3 * K_max * L_alloc * B_prime + recent_count
```

关键问题：

1. 存储是矩形的：`[batch, group, K_max, L_alloc, B_prime, dim]`。
2. 即使很多 cluster/level/slot 目前无效，也会先被 `flip/reshape/materialize_anchor_keys`
   物化出来。
3. 无效 slot 是在 `log_kv_slot_attention` 里 matmul 之后才 `masked_fill` 掉。
4. fixed-3 anchor 会把每个 entry 最多放大成 3 个 virtual slots。
5. 开启二阶修正时，`sigma_u/gamma_a/gamma_b/gamma` 也要一起物化和参与额外 matmul。

当前 attention 不是 FlashAttention，而是 `log_kv_slot_attention` 里的手写 attention：

- `litgpt/log_kv_cache.py:2962-2964`：GQA 分支计算 fp32 scores。
- `litgpt/log_kv_cache.py:2976-2982`：二阶 score 修正额外一次 `q @ sigma_u`。
- `litgpt/log_kv_cache.py:3012-3020`：value 读出后，二阶 readout 修正还要 `q @ gamma_a`
  和 `gamma_weight @ gamma_b`。
- `litgpt/log_kv_cache.py:2990-2994`：slot_valid mask 发生在 scores 已经算完之后。

这解释了为什么语义模式可能比 dense 还慢：它节省的不是当前实现的实际 matmul 宽度。

## 11. 当前 K64/B128 配置把读出成本放大了

当前 `exp/qwen1.7b-32k/eval.yaml`：

```yaml
log_kv_B: 128
log_kv_recent_size: 1024
log_kv_second_order_scale: 0.2
log_kv_prefill_block: 128
log_kv_pin_size: 0
log_kv_semantic_clusters: true
log_kv_cluster_k_max: 64
log_kv_semantic_s_h_path: null
log_kv_semantic_flush_granularity: 1024
```

对 32K 序列，语义模式的 `L_alloc` 近似：

```text
L_alloc = ceil(log2(max_seq_length / (K_max * B_prime) + 1)) + 2
```

几组预算直觉：

| 配置 | L_alloc | raw entries `K*L*B` | fixed-3 anchors | 加 recent 后读出 slot |
|---|---:|---:|---:|---:|
| K=64, B=128 | 5 | 40,960 | 122,880 | 约 123,904 |
| K=16, B=128 | 7 | 14,336 | 43,008 | 约 44,032 |
| K=8, B=128 | 8 | 8,192 | 24,576 | 约 25,600 |
| K=1, B=512 | 9 | 4,608 | 13,824 | 约 17,920 |

上表中 K64/K16/K8 按 `recent_size=1024` 估算；K1/B512 对应训练侧常见的
`recent_size=4096`，所以加 recent 后约 17,920。

32K dense attention 的 key 长度是 32,768。也就是说：

- K64/B128 的 fixed-3 读出宽度约是 dense 的 3.75 倍。
- K16/B128 约是 dense 的 1.31 倍。
- K8/B128 才低于 dense。

以 Qwen3-1.7B 的常见形状估算：

```text
n_layer = 28
n_head = 16
n_query_groups = 8
head_dim = 128
prefill_block = 128
K64/B128 slot 数约 122,880
```

GQA score buffer 形状近似：

```text
[batch=1, kv_group=8, repeat_factor=2, Tq=128, S=122880]
```

这是约 251M 个 fp32 元素，单层单 prefill block 接近 1 GiB 的 score buffer。二阶打开后还会
增加额外 matmul 和中间张量。长 prompt 下每层、每个 block 重复这个过程，自然慢。

## 12. `s_h_path: null` 不是主瓶颈，但会让路由质量不稳

`log_kv_semantic_s_h_path: null` 时，`LogStructuredKVCache.__init__` 里会使用默认
`s_h = 1.0`，位置在 `litgpt/log_kv_cache.py:617-630`。

`s_h` 控制新簇阈值：

```text
direct = semantic_distance <= cluster_lambda_rel * s_h
```

没有离线标定时，阈值可能过紧或过松：

- 过紧：orphan 太多，Phase 2 新簇/Ward 变多，速度和质量都可能差。
- 过松：不同语义混在一起，value smear 加重。

但即使 `s_h` 标定好了，当前写入串行和读出过量物化的问题仍然存在。`s_h` 是质量和 orphan
比例问题，不是全部墙钟问题的根治。

## 13. 还有一个独立的推理浪费：prefill logits

`litgpt/generate/base.py:84` 当前：

```python
logits = model(x, input_pos, input_pos_maxp1=input_pos_maxp1)
```

但 `sample()` 只使用：

```python
logits = logits[0, -1]
```

也就是 generate prefill 时可能为整个 prompt 的每个位置都计算 LM head logits，最后只取最后
一个位置。32K prompt、词表约 152K、bf16 logits 会接近 9.3 GiB 的输出量。

这不是 semantic route 的问题，但会让 `generate_until` 的长上下文推理更慢、更吃显存。最小
修复方向是让 `next_token()` 在 prefill 时只要求最后一个位置的 logits，例如调用模型时传
`lm_head_start=x.size(1)-1`。这个修复和语义簇无关，但收益直接。

## 14. 为什么 `decode_ms_per_token` 可能看起来更坏

`eval.py` 的 generate_until 计时如果把整条请求的 `prefill + decode` 总时间除以生成 token
数，那么长 prompt、短答案任务会把巨大的 prefill 成本摊到很少的 generated tokens 上。

NIAH/LongBench 这类任务常常是长输入、短输出，所以指标名如果叫 `decode_ms_per_token`，
实际可能并不是纯 decode 单 token 时间，而是：

```text
(prompt prefill + online cache build + decode) / generated_tokens
```

这不代表慢是假的，而是说明要单独拆分计时，否则会误判“decode 慢”还是“prefill 建簇慢”。

## 15. 当前瓶颈排序

按当前判断，瓶颈优先级如下。

### P0：读出宽度过大

K64/B128/fixed-3 在 32K 下让 slot 数超过 dense 很多，而且不是 FlashAttention。这个配置不
能拿来证明“语义簇推理应该快但实现小慢”，它本身就没有速度优势。

### P0：Phase 1 写入仍逐 token

`_semantic_existing_assignments` 并行后，`winner/direct` 被拉回 CPU，然后 Phase 1 direct
token 逐个 `_semantic_join_or_segment`。这是最明显的实现瓶颈，也是设计文档里已经记录的
问题。

### P1：无效 slot 先 matmul 后 mask

矩形布局导致大量空 slot 被物化；`slot_valid` 在 attention 分数算完后才 mask。真实有效
entry 少的时候，这会浪费大量算力和显存。

### P1：二阶修正成本高

`second_order_scale=0.2` 会启用 rank-1 统计构建和 attention 修正。它可能改善质量，但当前
速度排查应先做 `second_order_scale=0.0` 的速度 A/B，分清一阶语义簇和二阶语义簇各自的
成本。

### P1：路由判定的 `diff` 临时张量过大

`_semantic_existing_assignments` 可以改成：

```text
||x-c||^2 = ||x||^2 + ||c||^2 - 2 x c^T
```

用 batched matmul 直接得到 `[B, G, T_flush, K]` 距离，避免构造
`[B, G, T_flush, K, D]` 的巨大 fp32 diff。

### P2：prefill 全位置 LM head logits

这是独立浪费，修起来最小，应该尽快修。但它不是 semantic cache 自身的核心问题。

## 16. 最小修复顺序

不要一上来重写整个 semantic router。按最小闭环做：

1. **先换速度评测配置**：不要用 K64/B128/prefill128 做速度主结论。先跑
   `K=8, B=128, prefill_block=1024` 或 `K=16, B=64/128, prefill_block=1024`。
2. **先关二阶做速度 A/B**：`log_kv_second_order_scale=0.0`，确认一阶语义簇本身的成本。
3. **修 generate prefill logits**：`generate/base.py:next_token` 只算最后位置 logits。
4. **加分段计时**：至少拆出 `get_attention_state`、`log_kv_slot_attention`、
   `_semantic_existing_assignments`、Phase 1、Phase 2、LM head。
5. **改 `_semantic_existing_assignments` 距离公式**：避免巨大 `diff` 临时张量。
6. **做 valid-slot gather/packed readout**：只物化有效 entry/anchor，不让空 slot 参与
   matmul。
7. **批量化 Phase 1 direct 写入**：按 `(batch, group, cluster)` 分组，把 direct tokens
   批量归并进每个 cluster 的 ladder。Phase 2 仍允许保留小规模串行。

Ponytail 判断：第 7 步才是大工程；前 1-5 步成本小，足够先验证“慢主要在哪里”。

## 17. 不建议现在优先做的事

### 不要先把 Phase 2 完全并行化

Phase 2 的 orphan 决策天然带状态依赖。它应该被优化和计时，但不是第一刀。

### 不要只盯 checkpoint

checkpoint 可以让模型适应某种 cache 分布，但不能减少每条 prompt 在线构建语义簇的算力。
速度问题必须改 runtime 路径。

### 不要继续用 K64/B128 作为速度默认

这个配置的读出 slot 数已经超过 dense 很多。它可以作为质量探索配置，但不适合当速度基线。

### 不要把 `flush_granularity` 当纯性能旋钮随便改

flush batch 大小会改变三阶段近似边界，也可能影响路由质量。可以扫参，但需要同时看质量、
orphan 比例、Phase 1/2 时间，而不是只看速度。

## 18. 需要补的最小诊断数据

下一轮最好先补这些数据，避免继续凭体感判断：

1. 每层每 request 的 `get_attention_state` 时间。
2. 每层每 request 的 `log_kv_slot_attention` 时间。
3. 每次 flush 的 `_semantic_existing_assignments` 时间、显存峰值。
4. Phase 1 direct token 数和耗时。
5. Phase 2 orphan token 数、Ward merge 次数和耗时。
6. 有效 slot 数 vs 矩形物化 slot 数。
7. `second_order_scale=0.0` 和 `0.2` 的速度差。
8. `K=8/16/64`、`B=64/128`、`prefill_block=128/1024` 的速度和质量对照。

这些指标一旦出来，基本就能定量回答：

```text
到底是写入慢，读出慢，二阶慢，还是 LM head/prefill 计时口径造成的慢
```

## 19. 当前最重要的判断

SemanticLogKV 现在遇到的最大问题可以定义为：

```text
设计目标是用语义簇减少长程 KV 读写成本；
但当前实现里，写入端仍有大量逐 token Python cache mutation，
读出端又用 K_max * L_alloc * B_prime 的矩形 fixed-3 展开，
导致 runtime 成本没有随“语义压缩”真正下降，甚至在 K64/B128 下超过 dense。
```

所以后续主线不应该再问“为什么 checkpoint 推理还是慢”，而应该问：

```text
这条在线语义 cache 的运行时实现，什么时候才真的比 dense/vanilla LogKV 少算？
```

答案目前是：先降 K 和读出 slot，补计时；再修 prefill logits、路由距离临时张量、
valid-slot gather；最后才投入 Phase 1 批量 ladder 写入。
