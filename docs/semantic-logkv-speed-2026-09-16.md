# SemanticLogKV 加速（2026-09-16）

## 上一轮改动的复核结论：没有 bug

把 `HEAD` 版 `litgpt/log_kv_cache.py` exec 成独立模块与工作区版逐项对照（plain /
second-order / segment+pad / hard-cap+capacity-β / K=1 / chunk-tree，forward + host 和
device 两条 replay 路径）：21 个 `level_*` / `centroid` / `n_eff` / `n_total` / `alive`
等 buffer、`get_attention_state(with_stats=False/True)` 的全部字段、op_log 全部一致。
唯一差异是 `ward_cost` 的 ≤2.3e-5 浮点漂移（增量 Lance-Williams 更新 vs 重建），不进入
任何决策。去 flip 的映射 `physical_idx = entry_idx + (L_alloc - 1 - 2·ell)·B'` 也单独
验过，正确。

慢的原因不在那一轮改动里，是三个结构性问题。

## 本轮改了什么

| # | 问题 | 改法 |
|---|---|---|
| 1 | ladder append 按 `(b, g, cluster, level)` 逐 lane 走 Python，13 个 buffer 各自索引；op 数 ∝ `batch·groups·K_max·L·13` | `_semantic_append_entries_batched`：把 `level_*` 看成扁平行空间（`row = ((b·G+g)·K+c)·L·B + ell·B + slot`），每层对所有 lane 一次 gather、一次 `compact`、一次 scatter。索引算术全在 host 的 `_semantic_counts` 上做，不产生同步 |
| 2 | K 个簇槽用满后**每个 orphan 触发一次全量 Ward merge**（搬两簇整条 ladder + 一次 `.cpu()` 同步 + 全部重新 append），O(cache size)/orphan | `_semantic_route_orphans_fast`：每个 `(b, g)` 最多开 `n_free` 个新簇（farthest-point 选种，全程在 device 上、只回传一次），其余 orphan 并入最近 centroid。稳态零 Ward merge |
| 3 | 逐 lane 的 device 标量写入（`level_count` / `n_total` / `p_hi_c` / `current_segment` / `level0_phase` / `alive`）本身就是几百个 op | `_semantic_deferred_scalars()`：路由期间 host mirror 为准，退出时每个脏 buffer 一次 H2D `copy_` |
| 4 | replay 每个 `(b, g)` 串行走 cursor，每条 run 一次单 lane append | 两趟：先在 host 把每条 lane 的 op 行解析成工作项，再按「簇不重复」打包成轮次批量提交。ordering 约束是按簇而非按 lane 的，所以 phase 1 的 K 条 per-cluster run 合并成一次 append |

`centroid` / `n_eff` 也改成一次 `index_add_` 段求和加一次 scatter；forward 与 replay 共用
同一个 `_semantic_apply_join_plan`，只在「计划怎么来的」上不同。

旧路由保留在 `log_kv_semantic_legacy_route`（默认 false），可以做 A/B。

## 实测（CPU，torch 2.7，单线程；b=1, g=8, D=128, 32k 配置，每次 1024-token flush）

生产形状 K=16 / B=512 / second_order=on：

| | 改前（HEAD） | 改后 | 倍数 |
|---|---:|---:|---:|
| forward | 121.6 ms / 24,908 op | 28.9 ms / **1,309 op** | 19× op |
| replay | 3,997.6 ms / 1,220,610 op | 15.8 ms / **107 op** | **11,407× op** |

orphan 敏感性（K=8 / B=64 / 一阶）：

| orphan 比例 | legacy op / ms | fast op / ms |
|---|---:|---:|
| 0% | 643 / 47 ms | 643 / 47 ms |
| 5% | **8,872,435 / 37,357 ms** | 1,547 / 51 ms |
| 100% | **9,050,186 / 37,810 ms** | 1,524 / 51 ms |

op 数现在与 lane 数无关：groups 8 → 16，op 数完全不变（643 → 643）。这条由
`TestSemanticRoutingCost::test_flush_op_count_does_not_scale_with_lanes` 守住。

按每 micro-batch 2,604 次 `route_and_flush_batch`（28 层 × [fwd + AC 重算 + bwd replay]
× 31 chunk）折算：约 11.0 亿 → 约 236 万次 aten 调用；关掉 Block 级 activation
checkpoint 后约 123 万。

`unused/semantic_logkv_microbench.py` 在目标配置（K=8, B=64, T=1024, 8192 ops）下：
route 41.8 / 51.4 / 92.0 ms，replay 39.7 / 42.4 / 87.1 ms（orphan 0 / 0.1 / 1）。

## 回归与修正（目标 GPU 上 10 min/step -> 30 min/step）

第一版批量化在目标机上把 step 从 10 分钟拖到 30 分钟。原因是**优化指标选错了**：
我用 CPU 上的 aten op 数当目标，那个指标看不见 GPU 上最贵的两件事。

**1. 主机侧索引构建（主因）。** 批量 ladder 把每层的 gather/scatter 索引用 Python
`list.extend(range(...))` 攒出来，再 `torch.tensor(list, device=cuda)`。后者在 CUDA 上
是 pageable 内存的阻塞式 H2D 拷贝，而列表构建本身在 GPU 上一点也不会变快：

| | 每次 route 的 list->tensor | 搬运的 int 数 |
|---|---:|---:|
| 改之前 705eba90 | 128 次 | 8,192 |
| 第一版批量化 | 32 次 | **94,469** |
| 现在（numpy 修复后） | **5 次** | **5** |

9.4 万个 int 走 `torch.tensor(list)` 是每次 route 约 4.3 ms 的纯主机开销，
折算每 optimizer step 约 **180 秒**。改成 numpy 建（索引全是连续 range 的拼接，
`np.concatenate([np.arange(...)])` 快约 35 倍）并且每层只做一次传输后，这项归零。
本机 CPU 上同配置 route 也从 46.7 ms 降到 15.8 ms。

由 `TestSemanticRouterHostCost` 守住：一次 flush 经 `torch.tensor(list)` 的元素数
必须远小于 token 数。

**2. `activation_checkpointing: false` 是个地雷，已从 yaml 撤掉。** 按 Qwen3-1.7B
实测口径（n_layer=28, n_embd=2048, intermediate=6144, bf16），32k 下关掉 Block 级
checkpoint 需要保留的激活是 micro_batch_size=1 约 **49 GB**、=4 约 **196 GB**，
H200 单卡才 141 GB。当初写这条时doc 里标了"必须在目标 GPU 上先看显存"，
但不该直接写进入口 yaml。

`arc_semantic_fast.yaml` 现在退回成**纯代码路径 A/B**：尺寸、二阶、AC 全部继承
threephase 基线，只有路由实现不同，和已知的 10 min/step 基线只差一个变量。
`log_kv_B: 64` / `log_kv_second_order_scale: 0.0` 作为可选档留在注释里，逐个单独试。

## 配置：K=16/B=512 在 32k 下是负压缩

| K | B | L | live entries | 最坏 anchor slots |
|---|---|---|---|---|
| 16 | 512 | 5 | 18,432 | 38,912 |
| 16 | 64 | 8 | 5,152 | 13,408 |
| **8** | **64** | **9** | **3,080** | **8,216** |
| 8 | 32 | 10 | 1,784 | 4,840 |

dense 是 32,768。K=16/B=512 存的 entry 展开后比 dense 还大，attention 不可能比 dense
快，`second_order_scale=0.2` 还要再叠 5 份 payload gather 和 QΣ / QΓ 两组矩阵乘。
`exp/qwen1.7b-32k/arc_semantic_fast.yaml` 因此用 K=8 / B=64 / `second_order_scale=0`，
并加了 `activation_checkpointing: false`。

```bash
bash majob.sh exp/qwen1.7b-32k/arc_semantic_fast.yaml
```

## 验证边界

- `tests/test_log_kv_cache.py` 等 6 个缓存相关文件 350 项通过（`test_utils.py` 的 10 项
  失败是缺 `jsonargparse` / `bitsandbytes`，与本改动无关）。
- 新增 `TestSemanticFastRouteEquivalence`（无 orphan 时 fast 与 legacy 逐位相等；有
  orphan 时 fast 的 forward 与 replay 逐位相等，覆盖 plain / segment+pad / hard-cap /
  K=1 / chunk-tree × 二阶开关 × host/device op-log）和 `TestSemanticRoutingCost`
  （op 数上限、op 数不随 lane 数增长、饱和簇不做 per-orphan Ward merge）。
- 另用 `HEAD` 版模块逐项对照：legacy 模式在 11 个配置下与改前逐位一致（state、
  attention state、op_log 全等）。
- **未验证**：GPU kernel launch 实际耗时、bf16 数值、FSDP 显存（关掉 Block 级
  activation checkpoint 后是否 OOM）、NIAH / LongBench 精度。以上全部是 CPU 合成 cache
  的 op 数和 wall time，不能当成目标机 tokens/s 结论。`activation_checkpointing: false`
  必须在目标 GPU 上先看显存再固化。

## 仍然值得做的

1. **attention 仍是显式 score/softmax 路径**，且 `get_attention_state` 每个 chunk 重建
   一次（每层每遍 32 次）。一阶下可以考虑融合，但要保住 `log(w)`、anchor 去重和
   causal/valid mask。
2. **LM head 与交叉熵**：`demo.py` 先生成完整 logits 再分块算 CE，`entropy_chunk_size=128`
   带来很多小调用。
3. **chunk-tree 路径没有批量化**，仍走 `_semantic_ward_merge` 和逐 node 提交；当前入口
   用三阶段路由（`semantic_cluster_chunk_size: 0`），没动它。
