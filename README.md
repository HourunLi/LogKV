# SemanticLogKV（attach + Alpha + Beta + Gamma）

基于 LitGPT 的流式 KV 压缩：flush 时把新 token 用 attach 路由分给冻结中心的旧簇，
远离所有旧簇的 orphan 开新簇；每个簇维护有界层级缓存。Alpha 在同一缓存预算内保留可替换的
短片段精确槽，尝试减少孤立事实被均值压缩的损失；Beta 在 Alpha 上增加旧簇新颖度评分，
以及簇内局部四进二的自适应切分，不增加持久 KV 预算。Gamma 修正 Alpha 延迟归档对旧内容的
重复压缩（只重排第 0 层），并让饱和顶层合并最轻的相邻对。目标是固定簇数与窗口大小下
O(log N) 的推理 KV 存储。

当前默认：attach 路由、K=12、recent/flush=2048、精确池 256 token、片段上限 64 token、
Beta 两项开启、Gamma 第 0 层重排与最轻对顶层合并开启、mid anchor、一阶 attention。精确池从原
B=128 的预算中扣除，实际 B 由代码计算。训练/评测入口对语义簇默认开启 Alpha/Beta/Gamma、关闭
unified；YAML 写 `log_kv_alpha_exact_tokens: 0` 关闭 Alpha 与 Beta，`log_kv_beta_*: false` 只关
Beta，`log_kv_gamma_level0_reinsert: false` 与 `log_kv_gamma_top_merge: fold` 恢复旧梯子，
`log_kv_semantic_unified_route: true` 切回 unified 对照。Gamma 之前的实验 YAML 显式写回旧梯子。
attach + Alpha + Beta + Gamma 本身尚无评测结果；精确槽选择是启发式，收益需实测确认。

## 文档

| 文档 | 内容 |
|---|---|
| [共享算法规格](docs/algorithm-spec.md) | 缓存结构、attach / unified 路由、层级压缩、训推边界 |
| [Alpha 精确片段算法](docs/alpha-logkv.md) | 分段、打分、替换、预算，以及延迟归档 |
| [Beta 算法与运行入口](docs/beta-logkv.md) | 新颖度评分、低失真切分、回放约束、训练与评测 |
| [Gamma 梯子修正](docs/gamma-logkv.md) | 延迟归档只重排第 0 层、顶层最轻对合并、同字节余量层、针的去向诊断 |
| [统一路由实现](docs/semantic-unified-routing.md) | 可选 unified 路由的 Ward/半径公式、增量更新和数值约束 |
| [位置与注意力](docs/position.md) | pre-RoPE 内容、mid/multi 锚点、质量偏置 |
| [全程 SWA 对照](docs/swa-niah-comparison.md) | 同权重、同持久缓存字节预算的比较口径 |

## 运行入口

默认训练配置是 `exp/qwen1.7b-32k/arc_gamma_k12_b128_2k.yaml`（同目录有 B=256、精确池 1024 与
`level_slack: 0` 的变体），算法开关集中在其父配置 `arc_semantic_fast.yaml`。只训练、不触发默认的全套后续评测：

```bash
BENCHMARKS=none NIAH_BENCHMARKS=none \
  bash majob.sh exp/qwen1.7b-32k/arc_gamma_k12_b128_2k.yaml
```

训练后仅评测 32K single2/3（`eval.sh` 从 YAML 的 `save_path` 加载产物，不转发 YAML metadata）：

```bash
DIAG_ARGS='--metadata {"pretrained":"/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/","max_seq_lengths":[32768]}' \
  bash eval.sh exp/qwen1.7b-32k/arc_gamma_k12_b128_2k.yaml niah_single_2,niah_single_3 none
```

多卡评测各 rank 汇合时每 `LOGKV_EVAL_SYNC_HEARTBEAT_S`（默认 300s）打印仍未到达的 rank，
超过 `LOGKV_EVAL_SYNC_TIMEOUT_S`（默认 7200s）报错；长上下文生成负载差距更大时调高后者。

### KV 槽位压缩率

`eval.sh` 默认同时报告有效 KV 槽位压缩，无需重新训练或增加开关。先用 4 条 32K 样本检查输出：

```bash
DIAG_ARGS='--limit 4 --metadata {"pretrained":"/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/","max_seq_lengths":[32768]}' \
  bash eval.sh exp/qwen1.7b-32k/arc_attach_alpha_beta_k12_b128_2k.yaml niah_single_2 none
```

每次模型调用结束、重置缓存之前统计。令 `S` 为全部层、batch、KV groups 的有效存储条目总数，
包含层级缓存、recent（含注意力层暂存、待提交的尾 token）和 Alpha 精确池（未闭合片段已在池内），排除无效 padding。
`D` 是相同层数、batch、KV groups 下，已进入模型的真实 token 数对应的 dense 槽位总数：

| 输出字段 | 定义 | 含义 |
|---|---|---|
| `kv_retention_ratio` | `S / D` | 保留比例，越低越省 |
| `kv_saving_ratio` | `1 - S / D` | 节省比例，越高越省 |
| `kv_compression_factor` | `D / S` | 压缩倍数，例如保留 25% 即 4 倍 |

控制台打印全部 rank 的汇总；结果 JSON 的 `kv_compression` 保存汇总、按调用类型分组及逐次明细，
同目录的 `*.kv_compression.csv` 保存明细。汇总使用 `ΣS / ΣD`，不直接平均各样本比例；选择题每个
选项、rolling PPL 每个窗口各算一次模型调用。生成任务的分母为截断后的 prompt 加已回填的生成
token 数，**不包含最后一个尚未写入 KV 的输出 token**。这是请求结束时的快照，不是 prefill
结束值或生成期间的平均值；短输入可能尚未发生压缩，需结合明细中的 `processed_tokens` 解读。

此处统计存储条目，`multi` 的虚拟 attention 锚点不重复计数。槽位比例不是显存比例：预分配空位、
元数据、额外 K 副本、打包工作区及临时峰值不在该指标内；原有 `cache_budget` 仍仅报告持久缓冲预算。
Gamma 之前开启 Alpha 时，延迟归档反复合并旧条目，存活条目偏少、报告的压缩倍数偏高；Gamma 把条目数
恢复到设计值，压缩倍数随之下降而分配不变（见 [Gamma](docs/gamma-logkv.md)）。

更早实验的 YAML 显式写出训练时的开关，复评不受新默认值影响：纯 attach / unified 基线为
`arc_semantic_attach_route_*`、`arc_semantic_unified_route_*`；unified + Alpha 为 `arc_alpha_*`；
unified + Alpha + Beta 为 `arc_beta_cpt100.yaml`；attach + Alpha + Beta（Gamma 之前）为
`arc_attach_alpha_beta_*`。在训练环境、仓库根目录运行，并核对配置中的
模型和输出路径。CPT 从 Base 初始化；若实验输出目录已有 checkpoint，`auto_resume` 会恢复它。

## 性能开关与无 GPU 检查

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `LOGKV_ROUTE_OVERLAP` | `0` | attach 路由决策在 CUDA 侧流上与注意力重叠；开启前先在目标 GPU 运行 `python unused/check_route_overlap.py`，须逐位一致 |
| `LOGKV_ROUTE_CUDA_GRAPH` | `1` | unified 路由从第 2 轮起整轮 CUDA graph 重放；`0` 逐核启动 |
| `LOGKV_ROUTE_TILE_MB` | `1024` | unified 路由每 tile 距离矩阵目标大小；显存紧张设 `256` |

无 GPU 时可用 Triton 解释器在 CPU 上运行融合路径（路由归约、融合距离、attach 核、Alpha
分区等），与 Torch 参考逐位比较：

```bash
TRITON_INTERPRET=1 LOGKV_TRITON_CPU=1 python -m pytest -q tests/test_alpha_log_kv.py
python -m pytest -q tests/test_log_kv_triton_interpret.py
```

解释器把 fp32→bf16 转换截断而非就近舍入（CUDA 与 Torch 为就近舍入），因此 bf16 的 op-log 回放
用例在解释器下标为预期失败。解释器不覆盖 CUDA graph、流与性能；GPU 上用 `unused/benchmark_log_kv_unified.py --route attach`
测量路由耗时。

LitGPT 通用用法见 [tutorials](tutorials/)。运行产生的 profile 和评测文件是实验产物，
其时间、形状和代码版本须一起解读；它们不定义当前算法。
