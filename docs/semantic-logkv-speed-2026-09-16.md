# SemanticLogKV 加速（2026-09-16）

## Flash SDPA 后续优化（最新实现）

本节覆盖下方历史配置描述。`arc_semantic_fast.yaml` 显式采用 K=8、B=64、
`second_order_scale=0`、`activation_checkpointing=true`、`entropy_chunk_size=1024`。

- 永久一阶训练和推理不分配五份 Sigma/Gamma buffer，路由、合并、replay 也不搬运它们。
  训练入口按最终 scale 决定分配；零到非零 warmup 保留统计量存储。
- `demo.py` 的 FSDP checkpoint 只包 MLP，Block 的 FSDP 分片边界不变。
  attention 仍执行自身的 backward replay，但不再因整个 Block checkpoint 多跑一次 routing。
- Flash 输入先 padding，再在新 tensor 上原地清除无效位置，避免 K/V 各多一份中间副本。
- semantic forward 保存轻量 anchor 索引供 backward 使用，省去 replay 的排序和计数同步。
  只存索引、anchor 位置、multiplicity、有效性，不存各 chunk 的 K/V；通过 autograd 的
  saved tensors 管理生命周期。代价是额外的 `O(chunks × pooled_slots)` 整数元数据。
- `GPT.forward(targets=..., loss_chunk_size=...)` 将 LM-head 与 CE 一起分块 checkpoint，
  返回标量 loss，`demo.py` 已接入。不生成完整序列 logits 或 logits 列表；CE 用 fp32 累加，
  支持 softcap、bias 和 -100 标签。1024 是展平后的 token 数，0 表示不分块。

验证：新增输出/梯度、bf16、一阶无统计存储、单 ladder / K=8 / legacy / chunk-tree replay、
MLP 重算次数及二阶 warmup 检查。缓存、模型接线、新增优化及 Flash 三个测试文件共
240 项通过，4 项真实 CUDA 检查因本机无 CUDA 跳过。与本轮修改前 HEAD 对照，
scale=0 和 0.2 的合成 semantic stream 输出、梯度和 attention state 逐位一致。
本机 Python 3.9 / torch 2.6，模型检查用源码加载并延迟类型注解，绕开不可用的
Lightning 导入；未运行完整依赖环境或多卡 FSDP。

固定 batch=1、groups=8、dim=128、32K、K=8、B=64、recent=2048、bf16，按已分配
buffer 字节数计算，单层缓存（不含共享 RoPE）由 70.65 MiB 降到 43.51 MiB。
CPU 单线程、缓存已有 8192 token 时，状态构建中位数 60.12 ms，复用 plan 后 56.78 ms；
该位置一份 plan 为 857.4 KiB。以上不是 GPU tokens/s 或整步峰值显存结果，
真实 Flash/FSDP 速度、显存和 B=64 的下游精度仍需在目标机测量。

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

## 每 step 的 host 时间分项（新增）

两轮盲猜之后加的：`LOGKV_HOST_STATS` 累计 `route_and_flush_batch` 和
`get_attention_state` 的**主机**墙钟时间，demo.py 每个 step 打印并写进 tensorboard：

```
... | logKV_host: route 12.3s/868 + attn_state 45.6s/896 = 31% | Time: 187.4s
```

刻意用 CPU 时间而不是 CUDA event：要抓的失效模式就是"主机喂不饱 GPU"
（索引构建、同步），CUDA 计时器反而看不见。每次调用两个 `perf_counter()`，
相对毫秒级的被测区间可忽略。

本机 CPU、塞进 8k token 后的稳态，每次调用：

| 配置 | route | attn_state | 合计 | entries |
|---|---:|---:|---:|---:|
| K=16 B=512 二阶0.2 | 167.6 ms | 342.1 ms | 509.7 ms | 77,824 |
| K=16 B=512 二阶0 | 103.2 ms | 128.2 ms | 231.3 ms | 77,824 |
| K=16 B=64 二阶0 | 70.3 ms | 79.9 ms | 150.2 ms | 28,416 |
| K=8 B=64 二阶0 | 69.3 ms | 67.1 ms | 136.4 ms | 17,984 |
| K=8 B=32 二阶0 | 60.4 ms | 35.5 ms | 95.9 ms | 10,816 |

两点修正之前文档里的说法：

- **`get_attention_state` 比路由更贵**（当前基线下 342 vs 168 ms）。之前整轮优化都
  只盯着 route，方向就偏了。
- **B=512 -> 64 在稳态下让 route 变快而不是变慢**（103 -> 70 ms）。早先"B=64 慢 2 倍"
  的结论是在低占用率下测的：稳态时 B=512 每次 append 要搬 512 行 x 13 个字段，
  B=64 只搬 64 行。

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

1. **二阶 attention 仍是显式 score/softmax 路径**；一阶现已接入下述 Flash SDPA。
   `get_attention_state` 仍会在每个 chunk 重建，slot gather/anchor materialization
   的开销不由 Flash 解决。
2. **LM head 与交叉熵**：`demo.py` 先生成完整 logits 再分块算 CE，`entropy_chunk_size=128`
   带来很多小调用。
3. **chunk-tree 路径没有批量化**，仍走 `_semantic_ward_merge` 和逐 node 提交；当前入口
   用三阶段路由（`semantic_cluster_chunk_size: 0`），没动它。

## 一阶 Flash SDPA 接入

`log_kv_second_order_scale: 0.0` 时，共享的 `log_kv_slot_attention` 自动尝试 CUDA
bf16/fp16 Flash SDPA，训练前向、backward replay 和推理均可使用，无需安装 flash-attn。
当前 fast YAML 已显式设置 `log_kv_second_order_scale: 0.0`；其他 YAML 若仍为 0.2，
会继续走二阶手写 attention 路径。

用附加维度编码 `lambda*log(w)-log(M)`，Q/K/V 补齐到相同的 8 倍数维度；128 维输入
变为 136 维。非方形 `causal_lower_right(Tq, S)` 保持历史 prefix 全可见、当前 tail
因果可见。无效槽先将 K/V 清零，再用有限偏置 -10000 屏蔽，以避免扩维 backward 的
inf*0；这对正常模型 logits 下溢为零，不承诺任意极端 logits 下等同于 -inf。
偏置和附加 query 常量会量化到激活 dtype，因此不再逐位等于原来的 fp32 bias 路径。

快路径先用 `can_use_flash_attention` 检查，再限定 `SDPBackend.FLASH_ATTENTION`，
不会静默转到 SDPA math。CPU、fp32、二阶非零、自定义通用 mask、扩维后超过 192 维、
scale 不在 [0.001, 1] 或当前 GPU 不支持时回退原实现。运行时 OOM 等错误不会被吞掉。
cache 状态 replay 仍然保留；减少的是每块 attention 的完整 score/probability 张量。

本机验证：11 项新测试通过（fp32/bf16/fp16 的输出、Q/K/V 梯度、因果隔离、屏蔽槽零
梯度、K=8 streaming replay 与旧路径对照），171 项隔离缓存测试通过。
本机无 CUDA，4 项真实 Flash 前后向测试跳过，尚无 GPU 吞吐或峰值显存结论。
目标 GPU 上运行：

```bash
python -m pytest tests/test_log_kv_flash.py -v -rs
```

其中 CUDA 用例会检查 profiler 确实出现 Flash forward/backward 算子；若显示 skipped，
需查看跳过原因，不能将其当作 Flash 验证通过。
