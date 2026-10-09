# GammaLogKV：修正延迟归档，并把梯子容量留给分辨率

Gamma 建立在 attach + Alpha + Beta 之上（见 [Alpha](alpha-logkv.md)、[Beta](beta-logkv.md)），
不改变路由、精确片段选择或 Beta 评分，只改变条目在层级（梯子）里怎样合并。三个开关：

| 参数（省略 `log_kv_` 前缀） | Cache API 默认 | 入口 / `arc_semantic_fast.yaml` | 作用 |
|---|---|---|---|
| `gamma_level0_reinsert` | `false` | `true` | Alpha 延迟归档只重排第 0 层 |
| `gamma_top_merge` | `fold` | `lightest` | 顶层饱和时合并质量和最小的相邻对 |
| `gamma_level_slack` | `2` | `2` | 余量层数；小于 2 时用同样字节换更宽的 B′ |

Gamma 之前的实验 YAML 显式写回旧行为（`false` / `fold` / `2`），复评与训练时一致。

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
批量实现对所有仍在溢出的簇每步做一次 argmin、合并和移位，没有主机同步，与逐条参考路径逐位一致。

## 3. 余量层数（可选）

```text
L = max(2, ceil(log2(N / (K·B′) + 1)) + slack)
```

`log_kv_B` 仍表示经典梯子（两层余量）的字节；开启 Alpha 或 `slack≠2` 时，在这份字节内取
最大的 B′（精确池同样从中扣除）。32K、K=12、B=128、P=256：`slack 2` 为 B′=125、7 层；
`slack 0` 为 B′=218、4 层。上表中老内容的 w 约减半，代价是存活条目约多 45%：每步注意力的条目
更多，`kv_compression_factor` 下降，持久分配字节不变。偏斜簇会更早填满顶层，因此必须与
`gamma_top_merge: lightest` 一起使用；默认不启用。

## 与槽位压缩率统计的关系

`eval.sh` 报告的 `kv_compression` 统计**存活**条目。旧行为因重复合并而条目偏少，报告的压缩倍数
偏高；Gamma 把条目数恢复到设计值（接近关 Alpha 时），压缩倍数随之下降，持久分配不变。
比较不同版本时，应同时看分配字节（`cache_budget`）和存活条目，不能只比压缩倍数。

## 预算、训练与评测

- 不增加持久张量，推理 KV 仍为 `O(log N)`；Gamma 重排的数据只有第 0 层，比旧行为（整簇）少。
- 训练要求与 Alpha 相同（attach 或 unified、mid、一阶、`semantic_replay_updates=true`）。
- Gamma 之前的 checkpoint 是在旧梯子上训练的。可以在评测 YAML 里覆盖三项开关来复评，但
  训练与评测不一致；结论以 Gamma 重训为准。

默认训练配置是 [`arc_gamma_k12_b128_2k.yaml`](../exp/qwen1.7b-32k/arc_gamma_k12_b128_2k.yaml)。
同目录还有 B=256 对照（`_b256_2k`）、同字节精确池 P=1024（`_b128_p1024_2k`，B′ 125→116）
和 `level_slack: 0`（`_b128_slack0_2k`）：

```bash
BENCHMARKS=none NIAH_BENCHMARKS=none bash majob.sh exp/qwen1.7b-32k/arc_gamma_k12_b128_2k.yaml

DIAG_ARGS='--metadata {"pretrained":"/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/","max_seq_lengths":[32768]}' \
  bash eval.sh exp/qwen1.7b-32k/arc_gamma_k12_b128_2k.yaml niah_single_2,niah_single_3 none
```

## 针的去向诊断

[`unused/niah_needle_trace.py`](../unused/niah_needle_trace.py) 用 lm-eval 自己的 single_2/3
样本，按实验 YAML 构建同一模型与缓存，逐层报告针在精确池、recent 还是梯子条目（及其 w），
针所在片段在现有评分与几种诊断评分下的排名、需要多大的精确池才能留住，以及 `--oracle all`
强制保留针时的准确率上限。追踪期间关闭分数预取，选择规则不变（`--oracle` 除外）。

```bash
python unused/niah_needle_trace.py --config exp/qwen1.7b-32k/arc_gamma_k12_b128_2k.yaml \
  --hf-tokenizer <Qwen3-1.7B-Base 目录> --task niah_single_2 --samples 16
```

测试：`tests/test_gamma_log_kv.py`（修复前后的梯子形状、上层条目不被改写、同字节余量层），
`tests/test_alpha_log_kv.py`（守恒、延迟归档与反向回放，覆盖 Gamma 开关），
`tests/test_log_kv_updates.py`（顶层两种合并的批量与逐条路径一致）。
