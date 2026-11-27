# GammaLogKV：梯子修正与精确池的 cue、字面量选择

Gamma 建立在 attach + Alpha + Beta 之上（见 [Alpha](alpha-logkv.md)、[Beta](beta-logkv.md)），
不改变路由。前三个开关只改变条目在层级（梯子）里怎样合并；其余开关在精确池选择片段时加分：
重复了流开头区域里稀有 token 的片段（cue，第 4 节，可加保护；评测时可把 prompt 末尾的问题也作为 cue），
以及含随机字面量首次出现的片段（第 5 节）。七个开关：

| 参数（省略 `log_kv_` 前缀） | Cache API 默认 | 入口 / `arc_semantic_fast.yaml` | 作用 |
|---|---|---|---|
| `gamma_level0_reinsert` | `false` | `true` | Alpha 延迟归档只重排第 0 层 |
| `gamma_top_merge` | `fold` | `lightest` | 顶层饱和时合并质量和最小的相邻对 |
| `gamma_level_slack` | `2` | `2` | 余量层数；小于 2 时用同样字节换更宽的 B′ |
| `gamma_cue_tokens` | `0` | `64` | 开头区域长度；精确池按 cue 证据给片段加分，`0` 关闭 |
| `gamma_cue_guard` | `false` | `false` | cue 保护：开头区域须以换行结束，出现超过 16 次的 cue 不计证据 |
| `gamma_query_cue` | `false` | `false` | 查询感知 cue：生成请求 prompt 末尾的问题也作为 cue；字面量价值降到 cue 之下一档 |
| `gamma_literals` | `false` | `false` | 精确池给含随机字面量（数字串、编码）首次出现的片段加分 |

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
- 开头是正文时 cue 会占满精确池。32K 实测（cue + P=1024 的 checkpoint）：`niah_single_3_qend` 为 0.17，
  而开头指令不含针词的 `_paraphrase` 为 0.49；`_preamble` 居中（0.35）。代理中 P=1024 时 cue 价值高于 2
  （K/V 分数的上限）的片段占末尾精确池的比例：qend 99%、preamble 65%、paraphrase 2%：qend 的开头区域
  是第一篇文章的开头，片段重复其中的话题词就得到远超 K/V 分数的价值，K/V 分数本可留住的 uuid 进不了池。
  数字针（single_2 各变体约 0.2）不受 P 影响，K/V 分数本来就不偏向它们。下面的保护针对这一条。

**保护（`gamma_cue_guard`，可选）。** 两条规则，只作用于 cue：

- 开头区域必须截在换行处：前 M 个 token 里没有换行时，这一行没有 cue（指令自成一行；lm-eval 把正文
  空白压成单个空格，正文开头没有换行）。
- 在归档块末尾之前出现超过 16 次（`CUE_MAX_COUNT`）的 cue 不计证据：指向针的词重复次数约等于针的根数，
  普通开头的话题词与常用词重复几十次。qend 的例子里，第一篇文章开头的 less、hard、things、more 等词各出现
  12–80 次，每个只有 1–2.5 nats，但几个凑在一个片段里就超过 4 nats 的下限。

代理（同上，开头为第一篇文章）中，P=1024 时 cue 价值高于 2 的片段占末尾精确池的比例：

| | 原模板 | qend | preamble | paraphrase |
|---|---|---|---|---|
| 不加保护 | 0.07 | 0.99 | 0.65 | 0.02 |
| 加保护 | 0.04 | 0 | 0 | 0 |

原模板下 1 个 number、1 个 uuid（各 40）、4 个与 8 个 number（各 20）的针在 P=256 时仍全部留住。上限取 8 时
8 针只留住 3/20；只限制 cue 占精确池的比例（如 1/4）在 P=256 时装不下 4 根针。preamble 的指令在第 64 个
token 之后，加保护后这一变体不用 cue，表现回到不开 cue 的水平。保护默认关闭；只改变精确池选择，可以只评测：
[`eval_gamma_cue_guard.yaml`](../exp/qwen1.7b-32k/eval_gamma_cue_guard.yaml)（cue checkpoint）、
[`eval_gamma_cue_guard_p1024_slack0.yaml`](../exp/qwen1.7b-32k/eval_gamma_cue_guard_p1024_slack0.yaml)
（cue + P=1024 + `level_slack: 0`）与
[`eval_gamma_cue_guard_literal_p1024_slack0.yaml`](../exp/qwen1.7b-32k/eval_gamma_cue_guard_literal_p1024_slack0.yaml)
（同一 checkpoint 再开字面量）。现有 cue checkpoint 训练时 cue 在普通训练文本上照常触发，结论以重训为准。

```bash
DIAG_ARGS='--limit 100 --metadata {"pretrained":"/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/","max_seq_lengths":[32768]}' \
  NIAH_ONLY=1 bash eval.sh exp/qwen1.7b-32k/eval_gamma_cue_guard_p1024_slack0.yaml
```

**查询感知 cue（`gamma_query_cue`，可选，只用于评测）。** 问题在末尾（`_qend`）或开头指令不含针词
（`_paraphrase`）时，开头区域给不出指向针的 cue。生成请求的整段 prompt 在 prefill 之前已经到手，
问题就在它的末尾，开启后问题里的词也作为 cue：

- 问题区域：prompt 最后 `CUE_QUERY_TOKENS = 64` 个 token，从其中第一个换行之后开始（RULER 模板的
  问题前有换行）；没有换行、或只有最后一个 token 是换行时取整段 64 个。`eval.py` 在每个生成请求的
  prefill 之前只把这 64 个 id 拷到主机，用 `query_region` 截取，经 `model.set_gamma_queries` 交给
  CueStream。这一行下一次从位置 0 开始时取用一次，不会带到后面的请求。
- 问题里的 token 与开头区域的 cue 取并集（每个 cue 只计一次），证据、权重、下限、上限与保护规则不变；
  `gamma_cue_tokens: 0` 时只用问题，从位置 0 起都计证据。问题本身在 recent 里，归档时计数只来自正文：
  针格式里反复出现的 special、magic、number 超过 16 次后不计（应与保护一起开），区分针的是 key 里的词。
- 字面量降一档：开启后字面量价值为 `max(exp(min(E − 8, 6)) − exp(4), 0) / 片段长度`。下限仍是
  E > 12 nats；上限等于 cue 证据 6 nats 的价值，所以 cue 证据超过 6 nats 的片段排在任何字面量片段之前，
  而随机 7 位数（已到上限）在最长 64 个 token 的片段里仍有约 5.4，高于 K/V 分数（不超过 2）。
  不降档时，同格式针组成的干草堆里每根针都是新字面量、价值都到上限，会挤掉问题指向的那一根。
- 只在主机上运行：每个请求一次 64 个 id 的拷贝与一次并集；权重按归档块末尾缓存、各层共用；不增加 KV，
  推理 KV 仍为 `O(log N)`。训练与 loglikelihood 请求没有问题，cue 与不开时相同（字面量的档位除外）。

**代理评测（不是模型准确率）。** 与第 4、5 节相同的代理（P=256，约 28K 代理 token；问题前有换行，
与 RULER 模板一致），用仓库里的 `CueStream`、`query_region` 与 `LiteralStream`。"问题"一列是 64 个 token
的开头 cue 加保护再加问题。数字为针在末尾精确保留的样本数（共 40）：

| 干草堆 / 模板 / 针 | 无 | cue+保护 | 字面量 | cue+保护+字面量 | 问题 | 问题+字面量 |
|---|---|---|---|---|---|---|
| 文章 / 原模板 / number | 3 | 40 | 40 | 40 | 40 | 40 |
| 文章 / 原模板 / uuid | 4 | 40 | 40 | 40 | 40 | 40 |
| 文章 / qend / number | 4 | 4 | 40 | 40 | 40 | 40 |
| 文章 / qend / uuid | 4 | 4 | 40 | 40 | 40 | 40 |
| 文章 / paraphrase / number | 3 | 3 | 40 | 40 | 40 | 40 |
| 文章 / paraphrase / uuid | 3 | 3 | 40 | 40 | 40 | 40 |
| 文章 / preamble / number | 4 | 4 | 40 | 40 | 40 | 40 |
| 文章 / preamble / uuid | 4 | 4 | 40 | 40 | 40 | 40 |
| 同格式针 / 原模板 / number | 2 | 2 | 3 | 3 | 40 | 40 |
| 同格式针 / 原模板 / uuid | 1 | 1 | 1 | 1 | 40 | 40 |

"同格式针"是 `niah_multikey_2/3` 式的干草堆：全文都是同格式的针，只有 key 不同，只有问题能指出是哪一根。
不含针的文章上，末尾精确池与只用 K/V 分数时的 Jaccard（20 个样本）：P=256 时 cue+保护 0.57、加问题 0.47、
问题+字面量 0.29（只开字面量 0.66）；P=1024 时为 0.60 / 0.50 / 0.41（0.75）。问题里较少见的词
（如 mentioned、provided）在正文中出现不超过 16 次时也会给片段加分。代理里没有模型，K/V 分数是随机数；
准确率要在 GPU 上评测。

**32K 实测（每个任务 500 条）。** cue + P=1024 + `level_slack: 0` 的 checkpoint 只加保护与问题：
`AVG_NIAH_cue` 由 0.527 升到 0.9996（9 个提示扰动任务都不低于 0.998），`niah_multikey_2` 由 0.138 升到
0.94，`AVG_NIAH_multi` 由 0.584 升到 0.751。同一 checkpoint 再开字面量，提示扰动任务全部为 1，但
`niah_multikey_2` 降到 0.878，`niah_multivalue` 由 0.9225 降到 0.9125，其余差异不超过 0.008。P=256 的
cue checkpoint 开保护、问题与字面量：`AVG_NIAH_cue` 0.9993，`niah_multikey_1` / `_2` 为 0.918 / 0.912。
`niah_multikey_3` 在三种设置下都约 0.015。

**限制。**

- 问题按单个 token 匹配。Qwen 把 uuid 切成单个十六进制字符与 `-`，在全是 uuid 的干草堆
  （`niah_multikey_3`）里每个都出现上千次，权重为 0，uuid 形式的 key 给不出证据。
- 精确池里已有片段的 1.1 倍保留系数乘在整个分数上，含 cue 与字面量价值。同格式针的干草堆里每根针的
  字面量都到上限，这 10% 成为很大的绝对差，问题证据弱的目标针可能换不进去；开问题时不建议同时开字面量。

这是查询感知的压缩：缓存在 prefill 之前看到了问题（与用 prompt 末尾观察窗口选择 KV 的方法同类），
结果要与不看问题的设置分开报告。问题在缓存建好之后才出现时（多轮对话、同一文档多次提问）不起作用。
默认关闭；只改变精确池选择，可以只评测：
[`eval_gamma_query_p1024_slack0.yaml`](../exp/qwen1.7b-32k/eval_gamma_query_p1024_slack0.yaml)
（cue + P=1024 + `level_slack: 0` 的 checkpoint，加保护与问题）、
[`eval_gamma_query_literal_p1024_slack0.yaml`](../exp/qwen1.7b-32k/eval_gamma_query_literal_p1024_slack0.yaml)
（同一 checkpoint 再开字面量）、
[`eval_gamma_query.yaml`](../exp/qwen1.7b-32k/eval_gamma_query.yaml)（P=256 的 cue checkpoint，保护与问题）与
[`eval_gamma_query_literal.yaml`](../exp/qwen1.7b-32k/eval_gamma_query_literal.yaml)（同一 checkpoint 再开字面量）。

```bash
DIAG_ARGS='--limit 100 --metadata {"pretrained":"/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/","max_seq_lengths":[32768]}' \
  NIAH_ONLY=1 bash eval.sh exp/qwen1.7b-32k/eval_gamma_query_p1024_slack0.yaml
```

**配置。** 默认训练配置 [`arc_gamma_cue_k12_b128_2k.yaml`](../exp/qwen1.7b-32k/arc_gamma_cue_k12_b128_2k.yaml)
与 `arc_gamma_k12_b128_2k` 只差 cue。成本最低的第一个实验是只评测、不重训：
[`eval_gamma_cue.yaml`](../exp/qwen1.7b-32k/eval_gamma_cue.yaml) 用已训练的 attach + Alpha + Beta
checkpoint（B=128，旧梯子）只打开 cue；换 B=256 的 checkpoint 改其 `config:` 一行。

```bash
DIAG_ARGS='--metadata {"pretrained":"/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/","max_seq_lengths":[32768]}' \
  bash eval.sh exp/qwen1.7b-32k/eval_gamma_cue.yaml niah_single_2,niah_single_3 none

BENCHMARKS=none NIAH_BENCHMARKS=none bash majob.sh exp/qwen1.7b-32k/arc_gamma_cue_k12_b128_2k.yaml
```

**组合实验。** 同字节（B=128）下把 cue 与更大的精确池、更少的余量层组合：
[`arc_gamma_cue_k12_b128_p1024_2k.yaml`](../exp/qwen1.7b-32k/arc_gamma_cue_k12_b128_p1024_2k.yaml)
（P=1024，B′=116、7 层；与 cue 默认配置只差 P，与 `arc_gamma_k12_b128_p1024_2k` 只差 cue）和
[`arc_gamma_cue_k12_b128_p1024_slack0_2k.yaml`](../exp/qwen1.7b-32k/arc_gamma_cue_k12_b128_p1024_slack0_2k.yaml)
（再加 `level_slack: 0`，B′=203、4 层）。训练命令同上，换 YAML 即可。

**适用边界评测。** 两组 32K 任务检查 cue 在 RULER 原模板之外的表现，`unused/parse_lmeval_table.py`
汇总为 `AVG_NIAH_multi` 与 `AVG_NIAH_cue`（不计入 `AVG_all`）：

- RULER 多针（lm-eval 自带）：`niah_multikey_1`（文章干草堆里 4 根同格式的针，只问其中一根）、
  `niah_multikey_2` / `niah_multikey_3`（干草堆全由同格式的针组成，开头指令里的词在全文都不稀有）、
  `niah_multivalue`（同一个 key 的 4 个值）、`niah_multiquery`（一次问 4 个 key）。
- 提示扰动（[`eval_tasks/niah_cue`](../eval_tasks/niah_cue/niah_cue_utils.py)，`eval.py` 自动加载，
  tag `niah_cue`）：与 single_1 / single_2 / single_3 相同的干草堆、针、深度与评分，只改提示。`_qend`
  去掉开头指令，只在末尾提问；`_preamble` 在 RULER 指令前加约 90 个 token 的前导，超过 64 个 token 的
  cue 区域；`_paraphrase` 的开头指令不含针里的任何稀有词。

`majob.sh` 训练后的 NIAH 阶段与 `eval.sh` 的第三个参数默认跑全部 20 个捞针任务（single 3 个、多针 5 个、
提示扰动 9 个，以及 niahLogKV 的 3 个 words 针，见 [niahLogKV](niah-logkv.md)；`DEFAULT_NIAH_BENCHMARKS`），
每个任务在 1K–32K 六个长度上各 500 条，约为原先只跑 single_1/2/3 的 6.7 倍；`NIAH_BENCHMARKS=<逗号分隔的任务>`（`eval.sh` 为第三个参数）只跑一部分。
只跑 NIAH、跳过主评测：两个脚本都用 `NIAH_ONLY=1`（`majob.sh` 遇到已完成的 checkpoint 会跳过训练）；
`BENCHMARKS=none` 或 `eval.sh <yaml> none` 等价。任务列表里的空白会被去掉，空列表按跳过处理。
同一组任务分别评测 cue checkpoint 与不带 cue 的对照（如 `arc_gamma_k12_b128_2k.yaml`）；只看 32K、
每个任务先看前 100 条（去掉 `--limit` 即为全部 500 条）：

```bash
DIAG_ARGS='--limit 100 --metadata {"pretrained":"/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/","max_seq_lengths":[32768]}' \
  bash eval.sh exp/qwen1.7b-32k/arc_gamma_cue_k12_b128_2k.yaml none
```

## 5. 精确池选择：字面量（不可压缩度）

**问题。** cue 依赖开头的指令点名要记住的内容：问题只放在末尾（`_qend`）、指令被改写
（`_paraphrase`）或被前导挤出开头区域（`_preamble`）时，它不提供证据。字面量不看查询，看的是
片段进入梯子后损失有多大：梯子把同簇 token 求均值，随机数字串和编码合并后无法还原，
文中也没有别处重复它们。

**规则。**

- 字面量：由 ASCII 字母、数字和 `-_.,:/+#` 组成、中间没有空白的一串 token；遇到其他 token
  结束，遇到以空白开头的 token 重新开始，所以中文或正文里的数字各自成段。只保留含数字的字面量。
- 符号：字面量里的数字，以及夹在两个数字之间、不超过 3 个字母的字母段（十六进制与编码片段）；
  其他字母属于普通单词，不计。
- 编码长度：符号序列的自适应零阶编码长度 `E = Σ_i −ln((n_i(c_i)+0.5)/(i+0.5A))`，`n_i(c)` 为
  前 i 个符号中 c 出现的次数，A=10（只有数字）或 36（含字母）。随机 7 位数约 18 nats，年份与
  `1,000,000` 约 9–10 nats，uuid 超过 100 nats；日期（`2024-10-19` 约 19）、IP、电话号码等长数字串
  也高于下限。
- 首次出现：以符号串为键（`4,829,173` 与 `4829173` 相同），只有它在流中第一次出现的那段计编码
  长度，重复出现计 0。后文再引用它时，第一次出现的片段不降分。
- 价值：`max(exp(min(E, 30)) − exp(12), 0) / 片段长度`，E 取片段触及的首次出现字面量中最大的
  编码长度。与 cue 价值同形；两者都开时相加，再加到 K/V 分数上进入原有整段贪心（含旧片段 1.1 倍）。
  开查询感知 cue 时改为降一档的形式（第 4 节）。

**性质。**

- 只依赖 token id 与分词器逐 token 解码的文本（`Tokenizer.gamma_literal_texts`，启动时解码一次；
  SentencePiece 式词首片段单独解码会丢掉空格，这里补回），所有层得到相同的价值。
- 价值按归档块的末尾位置读取：跨过该位置的字面量只计它之前的 token。逐层整段 prefill、
  不同 `prefill_block`、逐 token 流式做出相同的选择（流与逐 token 参考实现逐位比较，并有模型级测试）。
- 主机端计算：模型每次前向把 token id 交给全模型共享的 `LiteralStream`（`litgpt/alpha_log_kv.py`），
  只保存含数字的字面量及其 token id，状态与字面量的出现次数成正比；不增加 KV，推理 KV 仍为
  `O(log N)`。下面的代理中，约 28K token 的一行接收约 7 ms，每次选择约 0.1 ms。
- 选择只在前向发生；反向与 checkpoint 重算回放记录的动作；开关写入回放签名。

**代理评测（不是模型准确率）。** 与第 4 节相同的代理（RULER 模板与针格式、Paul Graham 文章、
Qwen 预分词正则作 token 代理，K/V 分数用每个片段固定的随机数代替），约 28K 代理 token，用仓库里的
`cut_spans`、`choose_spans`、`add_cue_values` 与 `LiteralStream`。"原模板"即 RULER 模板，"qend"
去掉开头指令、只在末尾提问。单针 40 个样本、深度均匀；多针 20 个样本、深度随机；数字为全部针都在
末尾精确保留（精确池或 recent）的样本数，P=256：

| 针 / 模板 | 无 | cue | 字面量 | cue + 字面量 |
|---|---|---|---|---|
| 1 个 number / 原模板 | 3 | 40 | 40 | 40 |
| 1 个 number / qend | 3 | 3 | 40 | 40 |
| 1 个 uuid / 原模板 | 3 | 40 | 40 | 40 |
| 1 个 uuid / qend | 3 | 5 | 40 | 40 |
| 4 个 number / qend | 0 | 0 | 20 | 19 |
| 4 个 uuid / qend | 0 | 1 | 20 | 20 |
| 8 个 number / qend | 0 | 0 | 20 | 20 |
| 8 个 uuid / 原模板或 qend | 0 | 0 | 0（每样本留住 67% 的针） | 0（67%） |

8 个 uuid 针约 350 个代理 token，超过 P=256；P=1024 时为 20/20。只用文档内 token 统计的版本
（稀有度、只出现一次的 token、超额自信息之和）对数字针无效（qend、P=256 时 3/40）：随机数字每个
token 约 ln10 nats，与正文 token 相当，起作用的是字面量自身的随机性。

**对普通文本的影响。** 同一代理中不含针的文章拼接，末尾精确池与只用 K/V 分数时的 Jaccard 为
0.73（P=256）/ 0.78（P=1024）；开 cue 时为 0.05 / 0.10。字面量只替换含随机数字串的少数片段，
文章里的脚注编号、数字与单词粘连（如 `2008Technology`）也会被选中。对困惑度与其他任务的影响没有测过。

**限制。**

- 词面启发式，只覆盖含数字的字面量；纯字母的人名、地名等事实不在内。
- 文中数字很多时（表格、日志）字面量相互竞争，价值又远高于 cue 价值，会挤掉 cue 选中的不含数字的片段；
  由同格式针组成的干草堆（`niah_multikey_2/3`）里每根针都是新字面量，精确池装不下，帮不到被问到的那一根。
  开查询感知 cue 时字面量降到 cue 之下一档（第 4 节）。
- 少数随机数字串编码长度接近下限（如 `1810111` 约 12.3 nats）。
- 代理里没有模型；准确率要在 GPU 上评测。

**配置。** 默认关闭（Cache API 与入口均为 `false`，`arc_semantic_fast.yaml` 显式写 `false`），
已有实验不受影响。只评测、不重训（只改变精确池选择，与 cue 的 `eval_gamma_cue.yaml` 同理）：
[`eval_gamma_literal.yaml`](../exp/qwen1.7b-32k/eval_gamma_literal.yaml) 在 cue checkpoint
（`arc_gamma_cue_k12_b128_2k`）上再开字面量；
[`eval_gamma_literal_nocue.yaml`](../exp/qwen1.7b-32k/eval_gamma_literal_nocue.yaml) 在不带 cue 的
梯子 checkpoint（`arc_gamma_k12_b128_2k`）上只开字面量；
[`eval_gamma_literal_p1024_slack0.yaml`](../exp/qwen1.7b-32k/eval_gamma_literal_p1024_slack0.yaml) 在
cue + P=1024 + `level_slack: 0` 的 checkpoint 上再开字面量。训练配置
[`arc_gamma_cue_literal_k12_b128_2k.yaml`](../exp/qwen1.7b-32k/arc_gamma_cue_literal_k12_b128_2k.yaml)
与 cue 默认配置只差字面量。

```bash
DIAG_ARGS='--limit 100 --metadata {"pretrained":"/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/","max_seq_lengths":[32768]}' \
  NIAH_ONLY=1 bash eval.sh exp/qwen1.7b-32k/eval_gamma_literal.yaml

BENCHMARKS=none NIAH_BENCHMARKS=none bash majob.sh exp/qwen1.7b-32k/arc_gamma_cue_literal_k12_b128_2k.yaml
```

## 与槽位压缩率统计的关系

`eval.sh` 报告的 `kv_compression` 统计**存活**条目。旧行为因重复合并而条目偏少，报告的压缩倍数
偏高；Gamma 把条目数恢复到设计值（接近关 Alpha 时），压缩倍数随之下降，持久分配不变。
比较不同版本时，应同时看分配字节（`cache_budget`）和存活条目，不能只比压缩倍数。

## 预算、训练与评测

- 不增加持久张量，推理 KV 仍为 `O(log N)`；Gamma 重排的数据只有第 0 层，比旧行为（整簇）少。
- 训练要求与 Alpha 相同（attach 或 unified、mid、一阶、`semantic_replay_updates=true`）。
- Gamma 之前的 checkpoint 是在旧梯子上训练的。可以在评测 YAML 里覆盖梯子三项开关来复评，但
  训练与评测不一致；结论以 Gamma 重训为准。cue 与字面量只改变精确池选哪些片段，评测时单独打开的
  偏差小得多（`eval_gamma_cue.yaml`、`eval_gamma_literal*.yaml`、`eval_gamma_cue_guard*.yaml`、
  `eval_gamma_query*.yaml`）。

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
token），配置未开启 cue 时也会报告；`cue_guard` 是加保护后的 cue 价值，`query_cue` 是加保护并把
样本末尾的问题也作为 cue 的价值，`literal` 是单独的字面量价值。追踪期间关闭分数预取，
选择规则不变（`--oracle` 除外）。

```bash
python unused/niah_needle_trace.py --config exp/qwen1.7b-32k/eval_gamma_cue.yaml \
  --hf-tokenizer <Qwen3-1.7B-Base 目录> --task niah_single_2 --samples 16
```

测试：`tests/test_gamma_log_kv.py`（修复前后的梯子形状、上层条目不被改写、同字节余量层），
`tests/test_gamma_cues.py`（区域截断、去重与阈值、因果与分块不变、针留在精确池、保护的两条规则、
问题区域截取、问题与开头取并集且只用于下一个请求、问题指向的片段留在精确池、评测把问题交给模型、
模型接线与回放签名），
`tests/test_gamma_literals.py`（编码长度、首次出现、与参考实现逐位一致且不依赖分块、随机数针留在精确池、
cue 与字面量相加、降档公式、按归档块末尾缓存、模型接线与回放签名），
`tests/test_alpha_log_kv.py`（守恒、延迟归档、反向回放与预取，覆盖 Gamma 与 cue 开关），
`tests/test_log_kv_updates.py`（顶层两种合并的批量与逐条路径一致），
`tests/test_niah_cue_tasks.py`（提示扰动任务只改提示、经 `eval.py` 的任务索引加载并收到 metadata），
`tests/test_eval_cache_diag.py`（两个启动脚本默认跑同一份全部捞针任务），
`tests/interpret_update_check.py`（经 `tests/test_log_kv_triton_interpret.py` 在解释器上比较片段打分、
暂存与清空核、饱和顶层的最轻对合并核，以及整段 Gamma 轨迹）。
