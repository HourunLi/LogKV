# GammaLogKV：梯子修正与精确池的开头提示词（cue）选择

Gamma 建立在 attach + Alpha + Beta 之上（见 [Alpha](alpha-logkv.md)、[Beta](beta-logkv.md)），
不改变路由。前三个开关只改变条目在层级（梯子）里怎样合并；第四个开关在精确池选择片段时，
给重复了流开头区域里稀有 token 的片段加分（第 4 节）。四个开关：

| 参数（省略 `log_kv_` 前缀） | Cache API 默认 | 入口 / `arc_semantic_fast.yaml` | 作用 |
|---|---|---|---|
| `gamma_level0_reinsert` | `false` | `true` | Alpha 延迟归档只重排第 0 层 |
| `gamma_top_merge` | `fold` | `lightest` | 顶层饱和时合并质量和最小的相邻对 |
| `gamma_level_slack` | `2` | `2` | 余量层数；小于 2 时用同样字节换更宽的 B′ |
| `gamma_cue_tokens` | `0` | `64` | 开头区域长度；精确池按 cue 证据给片段加分，`0` 关闭 |

Gamma 之前的实验 YAML 显式写回旧行为（`false` / `fold` / `2` / `0`）；早于 cue 的四个 Gamma
梯子实验（`arc_gamma_k12_*`）写 `gamma_cue_tokens: 0`。复评都与训练时一致。

## 1. 延迟归档只重排第 0 层

Alpha 每次 flush 都会从精确池淘汰旧片段；这些 token 的位置早于目标簇的 `p_hi`（迟到 token）。
旧实现对收到迟到 token 的簇取出**所有层**的条目，与本次 token 一起按位置排序，清空整个簇后从
第 0 层重新灌入。已经合并过的条目在每一层又被两两合并；而几乎每次 flush 都有淘汰 token 散入
大多数簇，所以旧内容在每次 flush 都被再压一遍。attach 与 unified 都经过这条路径
（`_alpha_commit_joins`）。

Gamma 只取第 0 层的条目与本次 token 一起按 `p_lo` 稳定排序，清空第 0 层后正常追加：溢出时照常
把最老的成对条目进位，上层条目不会被重新合并。早于第 0 层全部条目的迟到 token 会在溢出时与第 0
层最老的条目合并后进位，这类条目的位置跨度会变宽，但只涉及迟到 token。重新追加的行接续该簇
第 0 层的顺序号；训练回放记录新动作 `clear_level0_batch`，不重新路由或选择。

真实代码在合成键上跑 32K（K=12、B=128 预算、P=256、片段 8–60 token、Beta 关）。机制与键的
分布无关，幅度取决于每次 flush 淘汰多少片段。"跨度"是条目覆盖的位置范围 `p_hi−p_lo+1` 的
质量加权均值；w 是按条目中点划分的八段上下文里的质量加权条目质量（左边最老）：

| 1 个 KV group、均衡簇 | 存活条目 | 跨度 | 质量在 >4K 跨度条目中 | w（最老 → 最新） |
|---|---|---|---|---|
| 关 Alpha | 6607 | 89 | 0% | 16 15 8 8 7.5 4 2.8 1.3 |
| Alpha，旧行为 | 3153 | 11057 | 64% | — 1314 1492 213 269 51 4.7 1.2 |
| Alpha + Gamma | 6440 | 199 | 1.1% | 16 16 8.5 7.9 7.8 4 2.8 1.2 |
| Alpha + Gamma + `level_slack: 0` | 9351 | 115 | 0.6% | 8 8 7.8 4.2 4 3.1 1.8 1.0 |

2 个 KV group、偏斜簇时：旧行为 6749 条目、跨度 6069、46% 质量在宽条目中；Gamma 12410、212、
0.7%；关 Alpha 12725、83。真实代码与一个只记质量和位置的梯子模拟器逐项一致，修复与模拟器的
预测也逐项一致。这些数字说明梯子的形状，不说明检索准确率；后者需要真实评测。

## 2. 顶层合并最轻的相邻对

整条梯子写满后，旧实现每来一个条目就把 slot 0 和 slot 1 合并（左折叠），最老的内容堆进一个
越来越重的条目（测试中占顶层质量的 72–88%，带 `log w` 的注意力偏置）。Gamma 合并质量和最小的
相邻对（并列取最老的），再左移后续条目、把新条目放在最后，顶层各条目质量保持相近（同一测试中
最重条目只占 16–18%）。32K、两层余量时顶层通常不饱和，这一项主要影响更长序列和更少余量层。

一次 flush 让 m 个条目进入某个饱和顶层时，要按顺序做 m 步，每步依赖上一步的结果。实现里槽位
只记行号：合并结果写回这一对中较老的那一行，后续槽位左移的是行号，不搬整行 K/V。CUDA 一阶路径
用一次 Triton 启动处理所有饱和顶层，每个顶层由一个程序跑完全部 m 步；其他路径按批量 PyTorch
逐步执行同样的行池。两条路径都没有主机同步，结果与逐条参考路径逐位一致。

## 3. 余量层数（可选）

```text
L = max(2, ceil(log2(N / (K·B′) + 1)) + slack)
```

`log_kv_B` 仍表示经典梯子（两层余量）的字节；开启 Alpha 或 `slack≠2` 时，在这份字节内取
最大的 B′（精确池同样从中扣除）。32K、K=12、B=128、P=256：`slack 2` 为 B′=125、7 层；
`slack 0` 为 B′=218、4 层。上表中老内容的 w 约减半，代价是存活条目约多 45%：每步注意力的条目
更多，`kv_compression_factor` 下降，持久分配字节不变。偏斜簇会更早填满顶层，因此必须与
`gamma_top_merge: lightest` 一起使用；默认不启用。

`slack 0` 时，偏斜簇的顶层在 32K 以内就会写满。合成偏斜簇（8 个 KV group，簇权重
`exp(−3c/K)`）从第 7 次 flush（约 14K token）起每次都有顶层合并，到 32K 时 96 个顶层中有 31 个
写满，每次 flush 最长的顶层要连续做约 65 步。这一步融合前每步约 230 个 PyTorch 算子，每层每次
flush 约 1.5 万个，是同一次 flush 其余路由与写入的 7 倍以上；训练时前向、checkpoint 重算和
反向回放各执行一遍。现在 CUDA 上是一次启动；PyTorch 路径每步约 95 个算子（下面的基准用
`--device cpu` 时，每次 flush 平均 341 → 273 ms）。在目标 GPU 上对比：

```bash
python unused/benchmark_log_kv_unified.py --device cuda --groups 8 --B 128 --flushes 15 --iters 3 --route attach \
  --pattern clustered --cluster-skew 3 --alpha-exact-tokens 256 --alpha-span-max-tokens 64 \
  --beta-novelty --beta-adaptive-merge --gamma-level0-reinsert --gamma-top-merge lightest --gamma-level-slack 0
```

输出里的 `saturated_top_levels` 是结束时写满的顶层数，`levels` 是层数。

## 4. 精确池选择：开头提示词（cue）

**问题。** 32K、P=256 时，每次 flush 要从约 80 个候选片段里只留约 8 个，旧片段只有 1.1 倍加成。
分数不随时间变化时，精确池会收敛到全文得分最高的约 1% 片段：针要在约 1100 个片段中排进前
8 左右。片段内方差比 R 与 Beta 新颖度都不指向针；在下文的代理评测中，用随机数代替这类分数时，
针在末尾仍被精确保留的样本只有 3/40，文档内 token 稀有度单独使用也只有 3–5/40。

**规则。**

- 开头区域：每条序列的前 M=`gamma_cue_tokens` 个 token，截到其中最后一个换行结束处；没有换行
  则截到最后一个片段结束处；都没有则取全部 M 个。区域里出现的不同 token id 称为 cue。RULER 的
  指令行（`A special magic number is hidden within the following text. … afterwards.\n`）正是
  这个区域：lm-eval 把文章里的空白压成单个空格，正文没有换行。
- 证据：设本次归档块的末尾位置为 n，某个 cue 在前 n 个位置出现 c 次；recent 窗口里更晚的
  token 不计入，因此决策与输入怎样分块无关。区域外的候选片段对它包含的每个**不同** cue 累加
  `max(0, ln((n+1+2048)/(c+1)) − ln 100)`：频率按多出 2048 个不含该 cue 的 token 估计，流开头的
  几块不只凭几百个 token 判断；频率高于约 1% 的 cue 不提供证据。n 还没有覆盖开头区域时价值为 0。
- 价值：`max(exp(证据) − exp(4), 0) / 片段长度`（证据上限 30），加到 K/V 分数（R 或 Beta 分数，
  约在 0 到 2 之间）上，再进入原有的整段贪心（含旧片段 1.1 倍）。指数相当于把证据当作对数几率；
  证据不足 4 nats 的片段保持原分数。针同时重复指令里两个稀有词（"special magic"），32K 时证据约
  6–7 nats，价值是 K/V 分数的几倍到几十倍，进入并留在精确池。

**性质。**

- 只依赖 token id，所有层得到相同的 cue 价值，同一个针在各层同时保留；各层的 K/V 分数仍各自不同。
- 证据只依赖归档块的末尾位置及其之前的 token：逐层整段 prefill、不同 `prefill_block`、逐 token
  流式与训练分块做出相同的选择（缓存级与模型级测试）。
- 主机端计算：模型每次前向把 token id、边界类型和起始位置一次拷回主机，交给全模型共享的
  `CueStream`（`litgpt/alpha_log_kv.py`）。不增加 KV，推理 KV 仍为 `O(log N)`；主机状态是开头区域
  和 cue 的出现记录，与 cue 在流中的出现次数成正比。
- 选择只在前向发生；反向与 checkpoint 重算回放记录的动作，不重新选择；开关写入回放签名。
  分数预取的仍是 K/V 分数，cue 价值在决策时加上，预取与现算的选择一致（有测试）。
- 换行表来自分词器（`alpha_line_end_ids`），入口在开启 cue 时把它与边界表一起装入模型；
  不提供换行表时，区域截到最后一个片段结束处。

**代理评测（不是模型准确率）。** lm-eval 的 RULER 模板与针格式、Paul Graham 文章作干草堆、Qwen
预分词正则作 token 代理（不是 BPE），仓库里的 `cut_spans`、`choose_spans` 与 `add_cue_values`
（证据按归档块末尾计数），flush 与 recent 各 2048、片段上限 64；K/V 分数用每个片段固定的随机数
U(0.3, 1.0) 代替。统计针所在片段在末尾是否仍完整地留在精确池或 recent 中，每组 40 个样本、深度
均匀，P=256（括号内为 P=128，相同时省略）：

| 干草堆 / 前导（约 28K 代理 token） | 针 | 无 cue | M=48 | M=64 | M=128 |
|---|---|---|---|---|---|
| 随机起点 | number / uuid | 3 / 4 (3) | 40 / 40 | 40 / 40 | 40 / 40 |
| 从第一篇文章开头（同 lm-eval） | number / uuid | 3 / 3 | 40 / 40 | 40 / 40 | 40 / 40 |
| 加 38 个代理 token 的聊天模板式前导 | number / uuid | 3 / 3 | 3 / 3 | 40 (39) / 40 (39) | 40 (39) / 40 (39) |

随机起点、M=64 的更短上下文：约 14K token 时无 cue 5 / 6、有 cue 40 / 40；约 7K token 时无 cue
12 / 12 (11)、有 cue 40 / 40。不加 2048 的频率先验时，流开头几块里的针证据不足：约 28K 时深度
≤6%、约 7K 时深度 ≤29% 的针会丢。聊天前导一组的指令在第 64 个代理 token 结束，说明区域必须覆盖
指令末尾的换行：M=48 时区域停在前导里，cue 不含 "special"、"magic"。

**对普通文本的影响。** 开头没有指令时，开头区域就是正文的第一段，后文重复其中稀有词的片段同样
会加分。代理中（16 段约 30K token 的文章拼接，保留换行，M=64、P=256）约 7% 的候选片段价值提高
超过 0.1，但精确池只保留约 1% 的片段，最终的池子与不开 cue 时几乎完全不同（Jaccard 约 0.02）：
池里是与开头话题相关的片段，不再是 K/V 分数最高的片段。这对困惑度和其他任务的影响没有测过；
默认训练配置开启 cue，模型在这种池子上训练。需要对照时写 `log_kv_gamma_cue_tokens: 0`。

**限制。**

- 词面启发式：依赖开头的指令点名要记住的内容。问题只放在末尾的任务（如 RULER QA）不受益；
  词形变化（number / numbers）不算命中。
- 代理里没有模型：K/V 分数是随机数，token 是预分词而非 Qwen BPE。准确率要在 GPU 上评测。
- M 必须覆盖到指令末尾的换行。默认 64 覆盖 base 模型 RULER 的指令（约 28 个 token），以及
  不超过约 35 个 token 的前导；带系统提示的聊天模板等更长前导需调大 M。

**配置。** 默认训练配置 [`arc_gamma_cue_k12_b128_2k.yaml`](../exp/qwen1.7b-32k/arc_gamma_cue_k12_b128_2k.yaml)
与 `arc_gamma_k12_b128_2k` 只差 cue。成本最低的第一个实验是只评测、不重训：
[`eval_gamma_cue.yaml`](../exp/qwen1.7b-32k/eval_gamma_cue.yaml) 用已训练的 attach + Alpha + Beta
checkpoint（B=128，旧梯子）只打开 cue；换 B=256 的 checkpoint 改其 `config:` 一行。

```bash
DIAG_ARGS='--metadata {"pretrained":"/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/","max_seq_lengths":[32768]}' \
  bash eval.sh exp/qwen1.7b-32k/eval_gamma_cue.yaml niah_single_2,niah_single_3 none

BENCHMARKS=none NIAH_BENCHMARKS=none bash majob.sh exp/qwen1.7b-32k/arc_gamma_cue_k12_b128_2k.yaml
```

## 与槽位压缩率统计的关系

`eval.sh` 报告的 `kv_compression` 统计**存活**条目。旧行为因重复合并而条目偏少，报告的压缩倍数
偏高；Gamma 把条目数恢复到设计值（接近关 Alpha 时），压缩倍数随之下降，持久分配不变。
比较不同版本时，应同时看分配字节（`cache_budget`）和存活条目，不能只比压缩倍数。

## 预算、训练与评测

- 不增加持久张量，推理 KV 仍为 `O(log N)`；Gamma 重排的数据只有第 0 层，比旧行为（整簇）少。
- 训练要求与 Alpha 相同（attach 或 unified、mid、一阶、`semantic_replay_updates=true`）。
- Gamma 之前的 checkpoint 是在旧梯子上训练的。可以在评测 YAML 里覆盖梯子三项开关来复评，但
  训练与评测不一致；结论以 Gamma 重训为准。cue 只改变精确池选哪些片段，评测时单独打开的
  偏差小得多（`eval_gamma_cue.yaml`）。

默认训练配置是 [`arc_gamma_cue_k12_b128_2k.yaml`](../exp/qwen1.7b-32k/arc_gamma_cue_k12_b128_2k.yaml)。
不带 cue 的梯子实验有 B=128（`arc_gamma_k12_b128_2k`）、B=256 对照（`_b256_2k`）、同字节精确池
P=1024（`_b128_p1024_2k`，B′ 125→116）和 `level_slack: 0`（`_b128_slack0_2k`）：

```bash
BENCHMARKS=none NIAH_BENCHMARKS=none bash majob.sh exp/qwen1.7b-32k/arc_gamma_k12_b128_2k.yaml

DIAG_ARGS='--metadata {"pretrained":"/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/","max_seq_lengths":[32768]}' \
  bash eval.sh exp/qwen1.7b-32k/arc_gamma_k12_b128_2k.yaml niah_single_2,niah_single_3 none
```

在目标 GPU 上对比路由与写入耗时（Gamma 只重排第 0 层，搬运的数据比旧行为少，实际收益以此为准；
`--gamma-cue-tokens 64` 用合成 Zipf token id 计入 cue 的主机开销）：

```bash
python unused/benchmark_log_kv_unified.py --device cuda --batch 4 --groups 8 --B 128 --iters 3 --route attach \
  --alpha-exact-tokens 256 --alpha-span-max-tokens 64 --beta-novelty --beta-adaptive-merge \
  --gamma-level0-reinsert --gamma-top-merge lightest --gamma-cue-tokens 64 > route_gamma.jsonl
```

## 每次 flush 的主机开销

训练按层、按 2048 token 的块推进，每块一次 flush。这一步 GPU 计算量小，耗时主要在主机：
PyTorch 算子与 Triton 启动、NumPy 规划，以及两次回读（片段分数、attach 决策；训练里分数在上一次
flush 后已预取）。当前默认 Gamma 路线（attach、Beta、Gamma 梯子与 cue）的主机路径：

- 候选片段由整数组切出；一个 Triton 核给全部片段打分（各 KV group 的方差比与 Beta 新颖度）。
- 选择结果直接带出归档行的位置；attach 的分组、空闲簇枚举与 run 元数据都是整数组运算。
- 普通 join 与延迟行共用一个暂存块，由同一个核分两次写入（延迟行在清空第 0 层之前）；清空所有
  条目字段一次启动。
- 层级追加先在主机规划全部层，所有层的索引一次上传；CUDA 上只上传区间（每次 flush 约一千个，
  而不是约十万个下标），在设备上按主机已知的大小展开，不回读。
- cue 的逐 cue 权重按归档块终点计算一次，供所有层复用。

在 CPU 上用 Triton 解释器跑融合路径、扣除解释器执行核的时间，4 核机器，不含 profiler：

| 每次 flush 的输入 | 主机时间 |
|---|---|
| 512 token，8 个 KV group，B=128，K=12，P=64，约 12 token 的自然片段，片段分数与位置无关（多数 flush 有延迟 join），cue 64 | 10.8 ms |
| 256 token，8 个 KV group，B=128，K=12，P=64，每 32 token 一个边界的合成块状键（保留的总是块尾片段，几乎没有延迟 join），cue 64 | 5.8 ms |

这里小张量算子在 CPU 上很便宜，GPU 上每个 PyTorch 算子和每次 Triton 启动的主机开销更大。按次数计
（512 token、2 个 KV group、dim 32、B=32、P=64、bf16），每次 flush 约 32 个 PyTorch 算子、17 次
Triton 启动、约 13 次主机到设备上传、2 次回读。

GPU 上的实际收益以训练日志为准：`logKV_host` 的 `route`、`replay` 分阶段计时，或在 YAML 里设
`log_kv_profile_steps: [3, 4, 5]` 记录 CUDA 计时。训练形状下（一次 flush 归档整个窗口块）可以让
路由决策与注意力重叠（`LOGKV_ROUTE_OVERLAP=1`），开启前先在目标 GPU 上运行
`python unused/check_route_overlap.py`，须逐位一致。

## 针的去向诊断

[`unused/niah_needle_trace.py`](../unused/niah_needle_trace.py) 用 lm-eval 自己的 single_2/3
样本，按实验 YAML 构建同一模型与缓存，逐层报告针在精确池、recent 还是梯子条目（及其 w），
针所在片段在现有评分（开启 cue 时含 cue 价值）与几种诊断评分下的排名、需要多大的精确池才能
留住，以及 `--oracle all` 强制保留针时的准确率上限。诊断评分 `cue` 是单独的 cue 价值（开头 64 个
token），配置未开启 cue 时也会报告。追踪期间关闭分数预取，选择规则不变（`--oracle` 除外）。

```bash
python unused/niah_needle_trace.py --config exp/qwen1.7b-32k/eval_gamma_cue.yaml \
  --hf-tokenizer <Qwen3-1.7B-Base 目录> --task niah_single_2 --samples 16
```

测试：`tests/test_gamma_log_kv.py`（修复前后的梯子形状、上层条目不被改写、同字节余量层），
`tests/test_gamma_cues.py`（区域截断、去重与阈值、因果与分块不变、针留在精确池、模型接线与回放签名），
`tests/test_alpha_log_kv.py`（守恒、延迟归档、反向回放与预取，覆盖 Gamma 与 cue 开关），
`tests/test_log_kv_updates.py`（顶层两种合并的批量与逐条路径一致），
`tests/interpret_update_check.py`（经 `tests/test_log_kv_triton_interpret.py` 在解释器上比较片段打分、
暂存与清空核、饱和顶层的最轻对合并核，以及整段 Gamma 轨迹）。
