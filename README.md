# SemanticLogKV（attach + Alpha + Beta + Gamma）

基于 LitGPT 的流式 KV 压缩：flush 时把新 token 用 attach 路由分给冻结中心的旧簇，
远离所有旧簇的 orphan 开新簇；每个簇维护有界层级缓存。Alpha 在同一缓存预算内保留可替换的
短片段精确槽，尝试减少孤立事实被均值压缩的损失；Beta 在 Alpha 上增加旧簇新颖度评分，
以及簇内局部四进二的自适应切分，不增加持久 KV 预算。Gamma 修正 Alpha 延迟归档对旧内容的
重复压缩（只重排第 0 层），让饱和顶层合并最轻的相邻对，并在精确池选片段时给重复开头指令里
稀有 token 的片段加分（cue）。目标是固定簇数与窗口大小下 O(log N) 的推理 KV 存储。

当前默认：attach 路由、K=12、recent/flush=2048、精确池 256 token、片段上限 64 token、
Beta 两项开启、Gamma 第 0 层重排、最轻对顶层合并与 cue（开头 64 个 token）开启、mid anchor、
一阶 attention。精确池从原 B=128 的预算中扣除，实际 B 由代码计算。训练/评测入口对语义簇默认开启
Alpha/Beta/Gamma、关闭 unified；YAML 写 `log_kv_alpha_exact_tokens: 0` 关闭 Alpha、Beta 与 cue，
`log_kv_beta_*: false` 只关 Beta，`log_kv_gamma_level0_reinsert: false` 与
`log_kv_gamma_top_merge: fold` 恢复旧梯子，`log_kv_gamma_cue_tokens: 0` 关闭 cue，
`log_kv_semantic_unified_route: true` 切回 unified 对照。更早的实验 YAML 显式写回各自训练时的开关。
attach + Alpha + Beta + Gamma 本身尚无评测结果；精确槽选择是启发式（cue 只在 RULER 代理上验证过），
收益需实测确认。

## 文档

| 文档 | 内容 |
|---|---|
| [共享算法规格](docs/algorithm-spec.md) | 缓存结构、attach / unified 路由、层级压缩、训推边界 |
| [Alpha 精确片段算法](docs/alpha-logkv.md) | 分段、打分、替换、预算，以及延迟归档 |
| [Beta 算法与运行入口](docs/beta-logkv.md) | 新颖度评分、低失真切分、回放约束、训练与评测 |
| [Gamma 梯子修正与 cue](docs/gamma-logkv.md) | 延迟归档只重排第 0 层、顶层最轻对合并、同字节余量层、精确池的开头提示词与字面量选择、针的去向诊断 |
| [niahLogKV 学习保留打分](docs/niah-logkv.md) | 不看开头与问题的精确池打分头、合成检索数据、`train_retention.py`、惊异度准入与诊断、评测与 CPT 配置 |
| [统一路由实现](docs/semantic-unified-routing.md) | 可选 unified 路由的 Ward/半径公式、增量更新和数值约束 |
| [位置与注意力](docs/position.md) | pre-RoPE 内容、mid/multi 锚点、质量偏置 |
| [全程 SWA 对照](docs/swa-niah-comparison.md) | 同权重、同持久缓存字节预算的比较口径 |
| [评测指标与性能口径](docs/evaluation-metrics.md) | 槽位、字节、峰值、训练吞吐与 prefill/decode 时延 |
| [Baseline 选型记录](docs/baselines.md) | 论文目标、候选分类、公开实现、优先级与公平比较 |

## 运行入口

默认训练配置是 `exp/qwen1.7b-32k/arc_gamma_cue_k12_b128_2k.yaml`，算法开关集中在其父配置
`arc_semantic_fast.yaml`。同目录不带 cue 的 `arc_gamma_k12_*` 是梯子修正的对照（B=256、精确池 1024、
`level_slack: 0` 等）；`arc_gamma_cue_k12_b128_p1024*_2k` 在同字节下把 cue 与 P=1024（及 `level_slack: 0`）组合；
`eval_gamma_cue.yaml` 只在评测时给已训练的 checkpoint 打开 cue。精确池的字面量选择（随机数字串、编码的
首次出现）默认关闭：`eval_gamma_literal*.yaml` 只在评测时打开，`arc_gamma_cue_literal_k12_b128_2k.yaml` 在 cue 之上训练。
cue 保护（开头须以换行结束、出现超过 16 次的 cue 词不计）默认关闭，`eval_gamma_cue_guard*.yaml` 只在评测时打开。
查询感知 cue（生成请求 prompt 末尾的问题也作为 cue，字面量降到 cue 之下一档）默认关闭，`eval_gamma_query*.yaml` 只在评测时打开。
学习到的保留打分头（niahLogKV：每层一个小打分头按片段 K/V 估计不可压缩度，不看开头与问题）默认关闭，
`bash retention.sh <eval_niah_*.yaml>` 训练（同 `majob.sh` 的训练环境），`eval_niah_*.yaml` 只评测、`arc_niah_*.yaml` 训练，见 [niahLogKV](docs/niah-logkv.md)。cue 在多针与改写提示下的适用边界评测见
[Gamma 文档](docs/gamma-logkv.md)（RULER 多针任务与 `eval_tasks/niah_cue`；两个启动脚本的 NIAH 阶段默认跑全部
20 个捞针任务，含 niahLogKV 的 3 个 words 针）。只训练、
不触发默认的全套后续评测：

```bash
BENCHMARKS=none NIAH_BENCHMARKS=none \
  bash majob.sh exp/qwen1.7b-32k/arc_gamma_cue_k12_b128_2k.yaml
```

全程 dense CPT 对照使用 `exp/qwen1.7b-32k/cpt_dense.yaml`：从 Qwen3-1.7B-Base
初始化，沿用 Gamma cue 的 32K 数据与训练超参，独立保存为 `qwen1.7b-32k-cpt-dense`。
`log_kv_dense_mode: true` 在训练中使用完整因果 attention，后续评测使用完整 KV；
保留 Block activation checkpoint 与分块 LM-head loss，不启用压缩缓存或路由重放。
该 dense 对照的推理 KV 为 O(N)，不属于 O(log N) 压缩路线。

```bash
BENCHMARKS=none NIAH_BENCHMARKS=none \
  bash majob.sh exp/qwen1.7b-32k/cpt_dense.yaml
```

`majob.sh` 与 `eval.sh` 自动检测当前进程可见的 CUDA GPU 数量，支持 1、2、4、8 卡，
遵守 `CUDA_VISIBLE_DEVICES`；ModelArts 的 `MA_NUM_GPUS` 作为分配上限。
可用 `GPUS_PER_NODE=2` 限制进程数，指定数量不得超过可见卡数或分配上限。
启动时打印检测结果，YAML 中的 `num_devices: 8` 不会覆盖 torchrun 的实际卡数。
1.7B 配置每卡 micro batch 为 1、全局 batch 为 32，单机 1/2/4/8 卡分别累积 32/16/8/4 次；
多机按全部进程数计算。无法整除时在加载模型前报错，不静默改变全局 batch。

8B 训练配置 `exp/qwen8b-32k/arc_gamma_k12_b128_p1024_2k.yaml` 完整声明训练与 LogKV 参数，
8B 的训练与评测配置只在 `exp/qwen8b-32k/` 内继承。全局 batch 为 64、每卡 micro batch 为 1，
64 个训练进程时累积 1 次，单机 1/2/4/8 卡分别累积 64/32/16/8 次。
8B CPT 的训练目标为 2000 个优化器步（`max_steps: 2000`）；中断后从本实验 checkpoint 自动续训。

8B 启用 `fsdp_low_memory: true`：使用 `FULL_SHARD`，前向后重新分片参数，并在每个 micro-batch
反向后同步和分片梯度，避免梯度累积期间常驻完整参数与梯度。同时关闭 AdamW 的 `foreach`
批量更新，避免 `optimizer.step()` 为整组参数的二阶矩同时分配开平方结果；恢复旧 checkpoint
后也强制应用此设置。代价是更多通信与优化器 kernel 启动，浮点运算顺序也会变化。
仅多卡 FSDP 能获得分片收益；该开关不减少缓存与回放记录。
未设置时保留原 `SHARD_GRAD_OP` 与延迟梯度同步行为。

`NIAH_ONLY=1` 只跳过主评测，仍会跑捞针；在 `majob.sh` 中也不会跳过尚未完成的训练。
只训练需像上面一样将两个 benchmark 环境变量都设为 `none`。之后单独评测捞针可用
`NIAH_ONLY=1 bash eval.sh exp/qwen1.7b-32k/cpt_dense.yaml`，默认全部 20 个任务。

全量评测（32K 主线，再 32K / 64K / 128K / 256K 逐档的长文本评测，主线测过的不重测；任务清单与数据见
[NIAH 文档](docs/niah-logkv.md)的“全量评测”）：

```bash
bash eval.sh exp/qwen1.7b-32k/eval_full_word_first.yaml
```

8B CPT 使用五个独立的全量评测 YAML，运行方式与 1.7B 相同：`bash eval.sh <yaml>`。
五个入口均在 `exp/qwen8b-32k/` 内继承配置，并读取同一份训练产物：
`/home/ma-user/work/bucket-pangu-green/lihourun/ckpt-logKV/qwen8b-32k-gamma_k12_b128_p1024_2k/lit_model.pth`。
`eval.sh` 从训练配置的 `save_path` 取得该目录并传入 `eval.py`；加载器优先读取目录自身的 `lit_model.pth`。
仅当目录自身没有权重时，才选择其中最大的 `step_*` 保存点。启动日志会打印实际使用的权重路径。
若挂载把 `lit_model.pth` 的软链接变成保存目标路径的小文本文件，加载器会只读解析其中的绝对或相对路径
（包括末尾 NUL），沿引用链读取真实权重，不创建软链接、不移动文件。`Checkpoint weights:` 是选择的逻辑路径，
`Checkpoint reference:` 记录引用跳转，`Checkpoint binary:` 打印实际二进制文件及大小。
目标缺失、引用循环、空文件或 Git LFS 占位会明确报错；需恢复同一 checkpoint 的真实文件，不自动改用旧 step。

| YAML 文件 | 评测方式 | 结果子目录 |
|---|---|---|
| `eval_full_cpt_dense.yaml` | CPT 权重，完整 KV 对照 | `evaluate/full_cpt_dense/` |
| `eval_full_baseline.yaml` | CPT 权重，K/V 分数准入 | `evaluate/full_baseline/` |
| `eval_full_word_first.yaml` | CPT 权重，惊异度准入 | `evaluate/full_word_first/` |
| `eval_full_query.yaml` | CPT 权重，问题 cue 与 cue 保护 | `evaluate/full_query/` |
| `eval_full_word_first_query.yaml` | CPT 权重，惊异度准入加问题 cue | `evaluate/full_word_first_query/` |

在仓库根目录分别执行：

```bash
bash eval.sh exp/qwen8b-32k/eval_full_cpt_dense.yaml
bash eval.sh exp/qwen8b-32k/eval_full_baseline.yaml
bash eval.sh exp/qwen8b-32k/eval_full_word_first.yaml
bash eval.sh exp/qwen8b-32k/eval_full_query.yaml
bash eval.sh exp/qwen8b-32k/eval_full_word_first_query.yaml
```

每份 YAML 先跑 32K 主线的常识推理、综合、LongBench v1 / LongBench-E、逐轮问与困惑度曲线，以及 1K–32K 捞针；
再逐档跑 32K / 64K / 128K / 256K 的 RULER、BABILong、LongBench v2、逐轮问、困惑度与 InfiniteBench。
主线不限条数，逐档默认每个任务 100 条；可用 `LIMIT=2` 先试跑。
`eval_full_dense.yaml` 是原始 Qwen3-8B-Base 的独立对照，训练后权重的完整 KV 评测使用上表的 `eval_full_cpt_dense.yaml`。

1.7B 与 8B 的全量评测均必跑 32K / 64K / 128K / 256K 的 LongBench v2；缓存缺失、缺子集或文件损坏会明确报错。
先用相同 `HF_HOME` 准备数据，再补测已有配置的 LongBench v2（不重跑其它指标）。下载脚本还会把 20 个子集、
503 道题复制为 `<HF_HOME>/longbench2` 下的普通 parquet 文件，逐文件校验 SHA256 和行数；已有数据完整时只需复制。
1.7B 和 8B 的评测只读取这份副本，目录缺失或为空也不回退到 hub 快照。三个启动脚本统一导出并打印
`LONGBENCH2_DIR`（默认 `<HF_HOME>/longbench2`），也可用它指定另一份共享副本目录，
下载和评测时须一致。多节点作业须确保每个节点都能读到该目录；失败信息会标明节点、文件大小、链接状态和文件首尾字节。
训练文件系统不支持软链接，缓存修复也直接复制完整 blob，不尝试创建软链接；下载源里末尾带 NUL 的链接文本会先去掉终止符。
LongBench v2 通过 PyArrow 只读加载 parquet，并在内存中构造 Dataset；不生成共享目录中的 Arrow builder 缓存，
避免 `datasets` 将 `.incomplete` 目录重命名时触发 `Operation not permitted`。已有完整副本无需重新下载。

```bash
bash diag.sh unused/fetch_long_eval_data.py --longbench2 --no-babilong --no-ruler-qa
EVAL_MAIN=0 EVAL_SUITES=longbench2 bash eval.sh exp/qwen1.7b-32k/eval_full_word_first.yaml
EVAL_MAIN=0 EVAL_SUITES=longbench2 bash eval.sh exp/qwen8b-32k/eval_full_word_first.yaml
```

训练完成后只跑捞针评测（默认全部 20 个 NIAH 任务；`majob.sh` 遇到已完成的 checkpoint 会跳过训练）：

```bash
NIAH_ONLY=1 bash majob.sh exp/qwen1.7b-32k/arc_gamma_cue_k12_b128_2k.yaml
NIAH_ONLY=1 bash eval.sh exp/qwen1.7b-32k/arc_gamma_cue_k12_b128_2k.yaml
```

训练后仅评测 32K single2/3（`eval.sh` 从 YAML 的 `save_path` 加载产物，不转发 YAML metadata）：

```bash
DIAG_ARGS='--metadata {"pretrained":"/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/","max_seq_lengths":[32768]}' \
  bash eval.sh exp/qwen1.7b-32k/arc_gamma_cue_k12_b128_2k.yaml niah_single_2,niah_single_3 none
```

评测默认只由全局 rank 0 输出阶段信息与低频进度（每 30s，另含起止），进度标明是本 rank 的分片；
全部 rank 汇总完成后再报告总条数，最终指标和结果文件不变。关闭 lm-eval 内部进度条及 INFO 日志，
截断警告保留每 rank 首次详情和本轮次数汇总，其它异常诊断保留。排查多卡问题时恢复详细输出：

```bash
LOGKV_EVAL_VERBOSE=1 bash eval.sh exp/qwen1.7b-32k/arc_gamma_cue_k12_b128_2k.yaml
```

该环境变量也适用于 `majob.sh` 的评测阶段和直接运行 `eval.py`。
多卡评测各 rank 汇合时仍每 `LOGKV_EVAL_SYNC_HEARTBEAT_S`（默认 300s）打印缺席 rank，
保留每个等待进程的低频心跳，避免 ModelArts 将等待误判为卡死。
超过 `LOGKV_EVAL_SYNC_TIMEOUT_S`（默认 7200s）报错；长上下文生成负载差距更大时调高后者。

### 多针 NIAH（RULER）

lm-eval 自带的 RULER 多针任务可直接作为任务名传入：`niah_multikey_1/2/3`（多个 key 只问其一，
2/3 的草堆全是干扰针）、`niah_multivalue`（同一 key 4 个值）、`niah_multiquery`（4 个 key 全问）。
`eval_tasks/niah_multi/` 补充针数可变的版本，`eval.py` 自动注册：

| 任务 | 针 | 生成预留 `max_gen_toks`（N=2/8/16/32） |
|---|---|---|
| `niah_multivalue_n{2,8,16,32}` | 同一 key 的 N 个值 | 128 / 128 / 256 / 512 |
| `niah_multiquery_n{2,8,16,32}` | N 个 key 全部提问 | 128 / 192 / 384 / 768 |

N=1、N=4 即 `niah_single_2` 与内置 `niah_multivalue`/`niah_multiquery`（同一生成器与 essay 草堆），
分数口径见 [NIAH 任务分数](docs/evaluation-metrics.md#niah-任务分数)。样本按生成预留缩短 prompt，避免 eval.py 为留出
生成空间而左截断；放不下全部针、问题和预留的长度会跳过并告警。metadata 的 `niah_num_samples`
（默认 500）设每个长度的样本数；`--limit` 只取最前面的样本，多长度时只会落在第一个长度。
RULER 以 500 词为步长添加草堆：针数较多时 1K–2K 的草堆很少甚至为空，多针结果宜从 4K 起看。

```bash
DIAG_ARGS='--metadata {"pretrained":"/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/","max_seq_lengths":[32768],"niah_num_samples":100}' \
  bash eval.sh exp/qwen1.7b-32k/arc_attach_alpha_beta_k12_b128_2k.yaml \
  niah_multivalue_n2,niah_multivalue,niah_multivalue_n8,niah_multivalue_n16,niah_multivalue_n32 none
```

长度超过训练 `context_length` 时，还要在 `DIAG_ARGS` 加 `--config_overrides {"block_size":131072}`
（按最大长度取值）：eval.py 的长度上限取 checkpoint 记录的 `block_size`，超出部分会被左截断。
essay 草堆总量有限，1M 时可能不足，此时会告警，该长度的分数对应更短的上下文。

## 评测输出与性能口径

`eval.sh` 自动报告 KV 有效槽位压缩率和推理时延；训练自动写入
`save_path/training_metrics.jsonl`。输出文件、计算公式、计时边界以及新旧指标差异统一见
[评测指标与性能口径](docs/evaluation-metrics.md)。

有效槽位比例、持久缓存分配字节和 CUDA 峰值是三种不同指标。当前
`decode_ms_per_token` 只计首 token 之后的 decode；跨 rank 推理汇总是单设备有效吞吐，
不能直接当作集群吞吐。指标采集不自动执行 Full KV 对照，也不代表已证明性能收益。

Gamma 之前开启 Alpha 时，延迟归档反复合并旧条目，存活条目偏少、报告的压缩倍数偏高；Gamma 把条目数
恢复到设计值，压缩倍数随之下降而分配不变（见 [Gamma](docs/gamma-logkv.md)）。

更早实验的 YAML 显式写出训练时的开关，复评不受新默认值影响：纯 attach / unified 基线为
`arc_semantic_attach_route_*`、`arc_semantic_unified_route_*`；unified + Alpha 为 `arc_alpha_*`；
unified + Alpha + Beta 为 `arc_beta_cpt100.yaml`；attach + Alpha + Beta（Gamma 之前）为
`arc_attach_alpha_beta_*`；只带 Gamma 梯子修正、不带 cue 的为 `arc_gamma_k12_*`。在训练环境、仓库根目录运行，并核对配置中的
模型和输出路径。CPT 从 Base 初始化；若实验输出目录已有 checkpoint，`auto_resume` 会恢复它。

## 性能开关与无 GPU 检查

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `LOGKV_ROUTE_OVERLAP` | `0` | attach 路由决策在 CUDA 侧流上与注意力重叠；开启前先在目标 GPU 运行 `python unused/check_route_overlap.py`，须逐位一致 |
| `LOGKV_ROUTE_CUDA_GRAPH` | `1` | unified 路由从第 2 轮起整轮 CUDA graph 重放；`0` 逐核启动 |
| `LOGKV_ROUTE_TILE_MB` | `1024` | unified 路由每 tile 距离矩阵目标大小；显存紧张设 `256` |

无 GPU 时可用 Triton 解释器在 CPU 上运行融合路径（路由归约、融合距离、attach 核、Alpha
分区与片段打分、join 暂存与清空等），与 Torch 参考比较（片段打分按舍入容差，其余逐位）：

```bash
TRITON_INTERPRET=1 LOGKV_TRITON_CPU=1 python -m pytest -q tests/test_alpha_log_kv.py
python -m pytest -q tests/test_log_kv_triton_interpret.py
```

解释器把 fp32→bf16 转换截断而非就近舍入（CUDA 与 Torch 为就近舍入），因此 bf16 的 op-log 回放
用例在解释器下标为预期失败。解释器不覆盖 CUDA graph、流与性能；GPU 上用 `unused/benchmark_log_kv_unified.py --route attach`
测量路由耗时。

LitGPT 通用用法见 [教程索引](tutorials/README.md)；通用 CLI 不自动启用本项目压缩路线。运行产生的 profile 和评测文件是实验产物，
其时间、形状和代码版本须一起解读；它们不定义当前算法。
