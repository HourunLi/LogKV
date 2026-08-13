# LogKV 项目工作记录（存档，更新于 2026-08-13）

> 本文件是给下次接续工作时用的存档，记录 LogKV（Fenwick-tree / O(log N) 显存 KV cache
> 压缩，rank-1 Σ_s/Γ_s 二阶修正）这条线目前做了什么、改了什么、卡在哪、下一步该干嘛。
> 代码层面的细节（怎么加插件、怎么跑服务）见仓库根目录上一级的 `~/CLAUDE.md`；这份
> 只讲 LogKV 这一个专题。

## 0. 现状速览（2026-08-13 更新，只想快速接续就读这节，细节看后面对应章节）

**结论先说（2026-08-13）：pin 这条线已终止**。阶段 4（训推一致，训练时注入 pin
分布）跑出真实结果——训练后 ≈ 不训练，niah/LongBench/LongBench_e 三项都没有追上
vanilla（无 pin），详见本节末尾和 6.13。四个独立方向（选择质量、剂量、分数尺度
机制、训推一致）依次验证均为负，不再往 pin 上投入。下一步的高价值方向是压缩本身
（阶段 3 揭示的 85+ 个百分点缺口，比 pin 影响大一个数量级）。

**2026-08-13 同日新增：压缩本身这条线的第一个方向——重要性加权池化——已实现
完毕（代码，非训练结果），见 6.14。核心思路：把 `_flush_pairs`/`compact()` 里
"槽的 pooling 权重"和"log(w) mass bias 用的 token 计数"解耦成两个独立量，
pooling 权重按启发式重要性（post-RoPE key L2 范数）加权而不是均匀 1/n，mass
bias 继续完全不变地用计数 `w`。是 k 的纯函数，无新增可学参数，训练/推理路径自动
一致（这正是 pin 系列失败的根因之一，这次设计上从一开始就规避掉）。默认关闭时
（`importance_pooling=False`）跟改动前逐字节相同，124/124 单测通过（113 条既有 +
11 条新增）。****还没有在真实 checkpoint 上跑过 eval****——本次会话只有本地 CPU
环境，没有 GPU，下一步是在现有 warmup CPT checkpoint 上跑一次纯 eval-time 决定性
实验（不需要重新训练，因为这是确定性启发式），命令见 8.5。**

**进展**：warmup CPT（`second_order_scale` 目标 0.2，warmup 100 步）已经完整训练
1500 步，并跑出了 dense baseline / LogKV vanilla / +importance pin / +2nd order+pins /
+2nd order no-pins 五组下游指标（ACC、LongBench、LongBench_e、niah@32768），四组
压缩变体用的是同一个 checkpoint + 同一套评测命令模板，可比性已确认（详见 6.1）。

**评测结果**（niah 是 niah_single_1/2/3 在 32768 长度上的均值，其余同理，完整细分表
见 6.1）：

| 配置 | ACC | LongBench | LongBench_e | niah |
|---|---|---|---|---|
| dense baseline（CPT 后）| 0.6158 | 0.2463 | 0.2715 | 1.0 |
| LogKV vanilla | 0.6146 | 0.1626 | 0.1803 | 0.032 |
| + importance pin | 0.6146 | 0.1363 | 0.1441 | 0.0313 |
| + 2nd order + pins | 0.6113 | 0.1214 | 0.1331 | 0.0467 |
| + 2nd order, no pins | 0.6113 | 0.1716 | 0.1918 | 0.0827 |

**怀疑的问题（按当前置信度从高到低，2026-08-10 已用真实 pin 诊断数据 + 随机基线更新，
见 6.4）**：
1. **`_log_kv_select_pins` 的选择质量本身就有独立、可定位的问题**——用生产配置
   `log_kv_pin_size=256` 跑通 pin-vs-needle 诊断并对比随机基线后（6.4 节，120 个
   niah 样本）：`exact_hit_rate`（17.2%）稳定超过随机基线（16.0%），但
   `near_hit_rate`（17.4%）只追平、甚至**普遍不如**随机基线（35.0%）——包括表现
   最好的 Layer 25（near 只打平，没有反超）。怀疑根因是 `topk` 选择缺乏空间分散性
   （`model.py:868-928`），预算可能抱团扎堆在少数"泛泛显著"的位置，纯随机采样
   反而因为撒得更开而更容易蒙对 needle 附近。**训练/推理分布不一致**
   （`_log_kv_select_pins` 是 `@torch.no_grad()`、训练路径完全不涉及 pin）仍然是
   一个独立成立的额外因素（即使某次真选中了 needle，模型也未必学会怎么用），但
   现在不再是唯一/首要的解释——选择质量本身就是一个可以先动手修的独立问题。
2. 二阶修正本身是正贡献（niah 0.032→0.0827），但**大部分下游损失在纯均值池化阶段就
   已经发生**（vanilla niah 就已经比 dense 掉了 97%），所以第 2 节"rank-1 在宽槽下
   失真"未必是当前最大的病灶——32768 长度下槽宽可能远超 width=8，rank-1 表达能力
   本身就不够，这正是 pin 该顶上的场景。
3. ACC（常识推理）四组几乎无差异，问题集中在长程检索类任务，短程信息保留良好。

**下一步方向（2026-08-12 更新——阶段 3 已出结果、阶段 5 新增，训练期 pin 注入正在跑，
见 6.10/6.11/6.12）**：

*阶段 1 —— 修选择算法，纯推理期改动，不碰训练（已完成）：*
(a)~(d) 同前，见 6.4/6.6/6.7/6.8——`log_kv_pin_min_distance` 的 NMS 约束把
near-hit 从"不如随机"修到 1.9× 随机，选择质量本身的问题已解决。

*阶段 2 —— 确认选择修好后是否真的解决下游问题（已完成，结论：没有，见 6.9）：*
(e)【已完成，见 6.9】NMS（min_distance=16/64）+ 256 pin 跑真实 LongBench/niah，
两档都明显不如不加 pin（niah 0.02/0.0113 vs vanilla 0.0827），也不如未加 NMS 的
原始 pin 版本；(e2)【已完成，见 6.9】追加 pin_size=4 的极小预算对照组，结果依然
不如 vanilla，但明显优于 pin_size=256 各变体——确认是**剂量效应**（插入的精确槽
越多伤害越大），不是选择精度问题。**阶段 1 的选择质量修复是真实有效的，但选择质量
从来不是下游变差的主因**——即使选得很准，"往压缩层级里混入精确槽"这件事本身对
当前（没在训练里见过这种输入分布的）模型就是净负贡献。

*阶段 3【已完成，见 6.10】—— 稠密检索能力是否被侵蚀：*
(f)(g) `log_kv_dense_mode` 开关真跑了 base vs warmup CPT 的稠密 NIAH 对比：捞针从
0（base）跳到 0.9353（CPT），LongBench/LongBench_e 也大幅提升，common sense 基本
持平（微降 0.3pp，续训常见副作用）。**证实 CPT 训练本身确实学会了长上下文检索
能力**；同时也证实了**压缩本身（不涉及任何 pin）就已经吃掉了 85 个百分点以上**
（0.9353 稠密 → 0.0827 压缩无 pin），这比 pin 相关的任何一组实验的影响都大一个
数量级——压缩损失，不是选点精度，才是最大的单一瓶颈。**但要注意 6.10 的一个重要
限定**：CPT 训练全程走的是 LogKV 分块压缩前向，从没真正用稠密注意力训练过，所以
0.9353 这个数字不是"干净的能力上限"，而是"压缩训练出来的权重泛化到陌生的稠密
输入分布"的结果——如果混淆因素有方向性，大概率是让这个数字偏保守（真实上限可能
更高），不影响"压缩代价远大于 pin 代价"这个结论。

*阶段 5【已完成，见 6.11/6.12】—— pin 分数/mass 尺度失配是否成立、是否可归因到
选点精度：*
新增了独立的 `log_kv_pin_score_diag` 旁路诊断（不复用要求 pin_size=0 的
`log_kv_diag.py`），直接在生产 `log_kv_slot_attention` 里挂钩记录 pin 槽 vs
pooled 槽的原始点积、加完 mass bias 后的最终分数、以及各自吃到的 softmax mass。
**核心结论（1500 个真实 niah 样本，见 6.12）：pin 槽的"塌缩"程度（某个 query 上
pin 吃掉远超比例的 mass）跟这个 pin 是否真的命中 needle，相关系数只有 ±0.09~0.17，
基本不相关**——推翻了"塌缩=模型精准命中后合理地全力押注"这个乐观猜测，支持
6.9 结尾提出的"分数尺度失配"机制假设：塌缩更像是某些位置的原始点积天生偏高（跟
语义相关性无关），不是选点选得准不准的问题。这进一步确认阶段 4（训推一致）是
唯一还没验证过的、有希望的方向。

*阶段 4【已完成，结论为负，见 6.13】—— 训推一致：让模型在训练里见过"精确槽混入
池化层级"这件事：*
(h) 已实现（`model.py`/`log_kv_cache.py`/`demo.py`/`base.yaml`，见 6.13，代码审查
无 bug）：训练时随机挑 `_log_kv_train_lowmem_forward` 已经流过的历史位置，复制
一份精确副本混进当前 chunk 的槽序列，配合 `pin_train_prob`/`pin_train_warmup_steps`
线性爬坡，模拟"精确槽 + 池化槽共存"的输入分布。短续训（`exp/qwen1.7b-32k/
pin_train_shortft.yaml`，从 step 1400 续跑到 1700，`pin_train_max=256`/
`pin_train_prob=0.5`）跑完并评测：**common sense 0.6126、LongBench 0.1328、
LongBench_e 0.1484、niah 0.0213**——common sense 跟 warmup CPT 基线（0.6123）
基本持平，说明短续训没有破坏原有能力，但 LongBench/LongBench_e/niah 三项都
**几乎等于 6.9 里未训练的 NMS-64 版本**（0.1313/0.1429/0.0200），仍然明显不如
vanilla 无 pin（0.1716/0.1918/0.0827）。**结论：训推一致假设也不成立**——即使
模型在训练里真的见过"精确槽+池化槽混合"这个分布，下游表现完全没有改善。选择
质量、剂量、分数尺度机制、训推一致，四个独立方向依次验证均为负，**pin 这条线
到此终止，不再投入**（一个诚实的保留意见：这次续训剂量偏轻——满强度注入只有
150 步、50% 触发概率、LR 5e-6——理论上不能 100% 排除"没训够"，但考虑到四个方向
一致指向同一结论、且已投入的验证成本，不建议为排除这个可能性再单独起一轮更长
训练）。

*下一步的高价值方向 —— 压缩本身，不是 pin：*
阶段 3（6.10）已经证实压缩本身（不涉及任何 pin）在 niah 上吃掉 85+ 个百分点
（0.9353 稠密 → 0.0827 压缩无 pin），比 pin 全系列实验的影响（0.011~0.047 之间
摆动）大一个数量级。往后如果继续做 LogKV，应该把精力放在压缩机制本身——槽宽、
二阶修正、pooling 方式——而不是继续在 pin 上调参。

*压缩本身·方向 1【已实现代码，未跑评测，见 6.14】—— 重要性加权池化：*
一阶均值 key 把 needle 稀释成 1/width，二阶修正（rank-1、只能加不能减）救不回一个
一阶就被冲淡到没法参与 softmax 竞争的槽——这是比继续加高二阶 rank 更根本的杠杆点。
`log_kv_importance_pooling` 开关（默认关闭，向后兼容）已实现并通过单测，下一步是
在 warmup CPT checkpoint 上跑一次纯 eval-time 决定性实验（命令见 8.5），不需要
重新训练。

*继续排在后面、暂不动的：*
(i) 6.3 提到的"按 layer/width 差异化 second_order_scale"接口改动。
(ii) 压缩本身·方向 2（稠密→压缩自蒸馏，让压缩前向对齐同序列稠密前向）：天花板更
高但需要训练时多跑一遍稠密 teacher 前向，成本更大，排在方向 1 出结果之后再评估。
(iii) 压缩本身·方向 3（rank-2/rank-r 槽统计）：跟方向 1 是替代关系（加权池化让
均值 key 已经带上 needle 内容后，需要的残差 rank 天然更低），排在方向 1 之后。

## 1. 这是什么项目

`litgpt/litgpt/log_kv_cache.py` 实现了一个分层（Fenwick-tree 风格，二进制进位合并）的
KV cache 压缩机制：新 token 先落在 width=1 的槽，随着序列增长，相邻两个槽以 2:1 均值
池化的方式合并成更宽的槽（width 2, 4, 8, 16...），整体显存占用是 O(log N) 而不是 O(N)。

朴素均值池化会丢失槽内 token 的方差信息，所以每个槽除了 `k`/`v`/`w`（合并权重），还
额外维护一份 **rank-1 二阶统计量**：`slot_sigma_u` / `slot_sigma2`（对 K 的协方差做
rank-1 近似）和 `slot_gamma_a` / `slot_gamma_b` / `slot_gamma`（对 K-V 互协方差做 rank-1
近似）。Attention 时用这些统计量对分数和读出值做二阶修正：

```
score_s = scale·(q·k_s) + 0.5·scale²·second_order_scale·sigma2_s·(q·sigma_u_s)² + λ·log(w_s)
read_s  = v_s + second_order_scale·scale·gamma_s·(q·gamma_a_s)·gamma_b_s
```

`second_order_scale` 是一个运行时标量，同时缩放分数修正和读出修正，训练和推理路径共用
（`litgpt/litgpt/log_kv_cache.py` 里的 `log_kv_slot_attention`）。目前**全局唯一一个
scale，没有按 layer / 按 width 区分**（这是本次会话里反复确认过的设计现状，不是遗漏——
下面第 5 节会讲为什么，以及诊断工具怎么在不改生产代码的前提下模拟"按 width 门控"）。

模型侧：`litgpt/litgpt/model.py` 的 `set_log_kv_second_order_scale()` 把这个标量设到
所有 transformer block 上（每层统一）；训练时 `_log_kv_train_lowmem_forward` 调用
`LogKVStreamTrainingAttention.apply(q, k, v, cache, scale, train_block,
self.log_kv_second_order_scale)`，梯度经过这个被 scale 加权的修正项回传（cache 状态
本身是 `.detach()` 的 / no_grad，但用它算出来的 attention 输出是可微的）。

## 2. 核心问题：rank-1 近似在宽槽下会失真

`compact()` 合并两个 B-slot 的 block 时，语义是"把 `[k1;k2]` 按时间顺序拼起来，然后
在拼接后的序列里配对相邻槽 (2i, 2i+1) → i"，**不是**按下标跨 block 一一配对。这个语义
本身是对的（通过追踪 `_binary_carry` 的真实行为验证过：某一层自己的 B 个条目和刚进位
上来的 B 个条目，正是按这种"拼接再相邻配对"的方式合并，能正确且独立地把各自内部的
pair 减半）。

合并用 `_rank1_psd_from_factors`（对小 Gram 矩阵做 eigh）算 Σ，`_rank1_cross_from_factors`
（QR+SVD）算 Γ，实现的是 Chan 并行协方差公式 + 强制 rank-1 截断。**这个强制截断就是
误差的来源**：width=2 的槽（正好两个 token）的协方差天然就是 rank-1，此时统计量是精确
的；但从 width=4 开始，两个已经是"近似 rank-1"的子槽再合并，截断误差会累积——这是
D1 自检（`sigma_q_rel` / `gamma_q_rel` / `score2_rel` / `value2_rel`，在 `by_level`
里）测出来的核心结论，而且这个结论是**纯算法性质**：只取决于原始 K/V 和 width，跟
模型权重、checkpoint、`second_order_scale`、`log_kv_B` 都无关（B 只影响某个 width 的
槽**什么时候**首次出现，不影响它出现时统计量的精度）。

**当前假设（部分验证，部分待验证）**：二阶修正在 width ≤ 4 时基本可靠，width ≥ 8 时
rank-1 近似开始明显失真，但这个失真是否在**实际下游指标**（NIAH 检索准确率、LongBench
分数）上造成有意义的伤害——尤其是在 `second_order_scale` 与训练时匹配的前提下——
**还没有一个干净的数据点能回答**（见第 6 节"未解决问题"）。

## 3. 这次会话做的代码改动（都已完成并验证）

### 3.1 诊断工具：width × layer 联合门控（`litgpt/litgpt/log_kv_diag.py`）

纯诊断用途，**从未接入生产路径**（`log_kv_slot_attention` 完全不知道这个东西存在）。
目的是回答"如果二阶修正只在 width ≤ N 且 layer ≤ M 时生效，指标会怎样变化"，不用改
生产代码、不用重新训练就能扫描这个门控阈值对诊断误差的影响。

- `DiagState` 新增 `second_order_max_width` / `second_order_max_layer`（`None` = 不限制）。
- `_diag_slot_core` 里新增一条并行计算路径：`layer_ok`（每次调用算一次）和 `width_ok`
  （每次 run 算一次）都满足时才用真实 `second_order_scale`，否则该项 `eff_scale = 0`；
  用这个 `eff_scale` 重新算一遍 `out_width_gated_c`，误差记到新字段 `err_width_gated`
  （复用命名，向前兼容旧的 `err_baseline` 语义）。
- `diag_mode(...)` context manager 加了对应的两个参数透传。
- `summary()` 的 `by_layer_output` 新增 `err_width_gated` 输出。

### 3.2 修了两个过时的单测（`litgpt/tests/test_log_kv_cache.py` 的 `TestCompact`）

`test_merge_weighted_average` / `test_merge_unequal_weights` 之前的预期值是按"跨 block
按下标一一配对"算的，跟 3.1 节确认的真实语义（拼接后配对相邻槽）不符。已按正确语义
改写期望值和注释。改完后 `TestCompact` 3/3、全文件 73/73 通过。

### 3.3 修了三处同一个模式的 YAML 覆盖 CLI 的 bug

`eval.py` 的 `main()` 里有个 `_o()` 辅助函数，规则是"YAML 里非 null 的值会覆盖 CLI
传的值"——这个规则是为了让 `benchmark`/`metadata`/`log_kv_diag_mode` 这类"故意留空
等 CLI 传"的字段生效。但如果 YAML 里某个字段写死了非 null 默认值，用
`--config <yaml>` 直接跑（而不是走 `eval.sh` 的展平方式）时，CLI 传的同名参数会被
**静默丢弃**，不会报错，很容易跑出一批"以为改了参数、实际上跑的还是旧值"的脏数据。

本次会话发现并修了三处：
- `exp/qwen1.7b-32k/diag.yaml`：`log_kv_second_order_scale: 1.0` → `null`
- `exp/qwen1.7b-32k/diag.yaml`：`log_kv_B: 512` → `null`
- `exp/qwen1.7b-32k/diag_warmup.yaml`：`log_kv_B: 512` → `null`

（`main()` 自己的 Python 函数默认值分别是 `1.0` / `512`，所以不传 CLI 时行为不变，
向后兼容。）**这个 bug 类型值得记住**：以后往任何诊断/评测 YAML 里加新字段、又想让
它支持 CLI 扫参时，默认必须写 `null`，不能写具体数值。

### 3.4 `eval.py` 诊断文件名加了区分度

加了 `_sos{scale}` 标签（scale≠1.0 时）和 checkpoint 名前缀（`diag_{ckpt_name}_{tag}_...`），
避免不同 checkpoint / 不同 scale 跑出来的诊断 JSON 文件名混淆、互相覆盖或分不清谁是谁
（历史上出过一次"新旧数据混淆"的问题，这是针对性修复）。

### 3.5 新建了 warmup CPT 专用配置

`exp/qwen1.7b-32k/arc_warmup.yaml`（训练）+ `exp/qwen1.7b-32k/diag_warmup.yaml`（诊断
评测），是 `arc.yaml` / `diag.yaml` 的平行版本，`expid`/`save_path`/`checkpoint_dir`
等全部换成独立路径（`-warmup` 后缀），跟原来那个不带 warmup 的 naive checkpoint 完全
隔离，互不覆盖。当前内容（已验证，见第 4 节）：`ckpt_dir` 指向 base Qwen3-1.7B-Base，
`max_steps=1500`，`log_kv_second_order_scale=0.2`，`log_kv_second_order_warmup_steps=100`。

### 3.6 本次接续（2026-08-07 续）：pin-vs-needle 只读诊断

先前排查做了三件事：
1. 帮用户核对了 warmup CPT（scale=0.2）五组下游指标数据，确认口径（ACC 笔误、niah
   取 32768 档均值的约定）——结果见 6.1 表格。
2. 读代码定位了"pin 拖累指标"的大概率根因：`model.py:724-725`
   （`_log_kv_train_lowmem_forward`，训练路径）与 `model.py:865/1013-1020`
   （`_log_kv_select_pins`，推理路径 `@torch.no_grad()`）之间完全没有交集——训练从不
   构建 `LogStructuredKVCache`，`log_kv_pin_size` 这个训练配置字段是死参数。
3. 读了 `majob.sh:377,396-408` 和 `unused/parse_lmeval_table.py:338`，确认 6.1 的
   niah 数据没有踩中 6.1 原先担心的"`limit` 截断到浅层 width"的坑（`majob.sh` 调
   `eval.py` 时不传 `--config`/`--limit`，全量跑；niah 数字按约定取的是 32768 档）。

随后实现了只读诊断：
- `litgpt/litgpt/log_kv_pin_diag.py`：定位 NIAH prompt 里的 needle sentence token span，
  汇总每层/每个 KV group 的 `_log_kv_pin_indices` 是否 exact/near hit。
- `eval.py`：新增默认关闭的 `--log_kv_pin_diag_output` 等参数，在真实 `generate_until`
  路径 prefill 后记录 pin indices；多卡时 gather 各 rank 的诊断样本。
- `model.py`：reset/init cache 时清掉旧 `_log_kv_pin_indices`，避免诊断读到上一条样本的
  残留调试状态。
- `unused/diagnose_log_kv_pins_jsonl.py`：离线入口，给已经 dump 好的 prompt JSON/JSONL
  也能复用同一套 LogKV eval 路径跑 pin-vs-needle 诊断。

诊断不改变模型输出、不参与训练，仍需在目标评测环境上跑真实 checkpoint 得出结论。

## 4. 一次真实的训练崩溃排查（已定位根因）

用户报告：第一版 warmup CPT（`second_order_scale` 目标 1.0，`warmup_steps=10`）训练
loss 从原本 2 左右直接飙到 7，LongBench 和 NIAH 几乎崩坏。用户给的分 scale 观察数据：

| scale | loss |
|---|---|
| 0.1 | 0.5896 |
| 0.2 | 2.70 |
| 0.3 | 4.1719 |
| 0.4 | 6.656 |
| 1.0 | 11-12（训练若干轮后降到 7，但整体效果极差）|

排查结论：**不是"二阶修正本身太强/有害"，而是目标 scale（1.0）远超过一个明确存在的
"甜点区"（eval-time-only 的 scale 扫描显示大约在 0.1–0.25 之间），叠加 warmup 只有 10
步（几乎没有缓冲）**，两者叠加导致训练早期梯度爆炸式崩溃。不是根本机制问题。

修复：`arc_warmup.yaml` 改成 `log_kv_second_order_scale: 0.2`、
`log_kv_second_order_warmup_steps: 100`，`ckpt_dir` 确认（用户确认过）是从干净的
Qwen3-1.7B-Base 开始训练、不是从崩溃的 checkpoint 续训。已重新用 `majob.sh` 启动。

**训练/评测双端一致性已核实**（这是本次会话最后确认的问题）：
- 训练（`demo.py --config arc_warmup.yaml`）：前 100 步 `current_second_order_scale`
  从 0 线性爬升到 0.2（`get_log_kv_second_order_scale` 算，`set_log_kv_second_order_scale`
  每步设一次），之后固定 0.2，二阶修正项参与反向传播。
  同一份 yaml 里 `log_kv_pin_size` 没设（继承 `arc.yaml`? 需要下次确认，pin 只影响推理
  期缓存配置、不影响训练权重——`_log_kv_select_pins` 是 `@torch.no_grad()` 的纯推理期
  行为，所以训练用非 0 pin、推理改成 0 pin 是安全的，两者可以自由不一致）。
- 评测（`majob.sh` 的 `LOG_KV_ARGS`，第 242 行拼出 `--log_kv_second_order_scale
  ${LOG_KV_SECOND_ORDER_SCALE}`，值从同一份 yaml 里 bash 提取）：固定用 0.2，跟训练
  目标值一致，没有爬坡（评测不需要）。
- `majob.sh` 正常调用**不会**激活任何 `diag_mode`（`log_kv_diag_mode` 从不传，
  `diag_active=False`，走 `contextlib.nullcontext()`，诊断代码路径完全绕过），但数值
  上等价于 `diag_mode=baseline` 测出来的东西（`baseline` 就是直接委托给生产路径
  `_production_block` → `log_kv_slot_attention`）。第 3.1 节的 width/layer 门控对
  `majob.sh` 的真实产出**没有任何影响**——它只在显式用 `diag_mode` 跑诊断脚本时才会
  生效。

## 5. 为什么现在没有按 layer / 按 width 差异化 scale

这是设计讨论，不是代码限制——`second_order_scale` 目前是运行时传入所有层的单一标量，
`model.py` 遍历所有 transformer block 时统一设置。要做到"每层甚至每个 width 桶用不同
scale"，需要改的是接口形状（比如传一个按 layer 索引的 list/dict），而不是算法本身；
本次会话没有实现这个（用户明确要求先做诊断验证，别急着改生产接口），而是用 3.1 节的
诊断门控在不改生产代码的前提下模拟"只在 width ≤ N 时生效"的效果，用来判断值不值得
真的去做这个接口改动。

## 6. 未解决问题 / 下一步

### 6.1【已解决，2026-08-07】"width≥8 + scale 匹配训练值(0.2) + 生产 B=512"的干净数据

warmup CPT（scale=0.2, warmup=100 步，**1500 步训练已完整跑完**，见 6.2）产出了完整
的下游指标对比，走的是 `majob.sh`（不是本节原先设想的 `diag_warmup.yaml` 单条
torchrun 命令），四组配置用的是**同一个 checkpoint + 同一套评测命令模板**，只切换
`log_kv_pin_size` / `log_kv_second_order_scale`，可比性已确认：

| 配置 | ACC | LongBench | LongBench_e | niah（均值）| niah_single_1 | niah_single_2 | niah_single_3 |
|---|---|---|---|---|---|---|---|
| 完整 transformer（dense baseline，CPT 后）| 0.6158 | 0.2463 | 0.2715 | 1.0 | – | – | – |
| LogKV vanilla（无二阶修正、无 pin）| 0.6146 | 0.1626 | 0.1803 | 0.032 | 0.024 | 0.046 | 0.026 |
| LogKV + importance pin | 0.6146 | 0.1363 | 0.1441 | 0.0313 | 0.044 | 0.03 | 0.02 |
| LogKV + 2nd order + pins | 0.6113 | 0.1214 | 0.1331 | 0.0467 | 0.068 | 0.032 | 0.04 |
| LogKV + 2nd order, no pins | 0.6113 | 0.1716 | 0.1918 | 0.0827 | 0.07 | 0.114 | 0.064 |

（用户报的原始数字里 "LogKV vanilla" 的 ACC 一开始写的是 `10.6146`，按其余四组 ACC
都在 0.61–0.62 区间、且跟下一行 "importance pin" 的 0.6146 完全一致，已确认按笔误
处理，记录为 `0.6146`。）

**为什么现在认为这组数据是干净的（关闭了本节原先反复卡住的顾虑）**：
- `majob.sh:377` 的 `META` 确实是 `max_seq_lengths: [1024, 2048, 4096, 8192, 16384,
  32768]` 多值列表，但 `majob.sh:396-408` 调 `eval.py` 时**没有传 `--config`，也没有
  传 `--limit`**——`eval.py` 里 `limit` 的函数默认值是 `None`（`eval.py:824`），即
  **不截断，6 个长度各 500 条样本全跑**。本节原先担心的"`limit=40` 吃掉浅层长度"这个
  坑，只存在于 `--config diag*.yaml` 那条路径，`majob.sh` 完全没碰到。
- `unused/parse_lmeval_table.py:338` 里 `TARGET_METRIC_MAP` 显式把 `niah_single_1/2/3`
  映射到 `"32768"` 这一档——lm-eval-harness 对 niah 任务是按每个长度单独出一行分数
  （不是跨长度平均），用户确认过表里的 "niah" 数字就是按"每个 niah_single_i 只取
  32768 档、三个取平均"这个约定算的（见上表 niah_single_1/2/3 细分列）。所以这组
  niah 数字**是真实的 32768 长度、深 width 区间数据**，不存在浅层污染。

**解读（现在可以当作阶段性结论，不只是观察）**：
- 二阶修正（scale=0.2，与训练目标匹配）在**不加 pin** 时是四个压缩变体里下游指标
  最好的：LongBench 0.1716 > vanilla 0.1626，niah 0.0827 > vanilla 0.032（约 2.6×，
  三个子任务全面提升：0.07/0.114/0.064 vs 0.024/0.046/0.026）。说明 warmup CPT 训练
  出来的二阶修正在下游任务上确实有正向作用，第 4 节的训练崩溃问题修复后没有留下
  副作用，值得继续往这个方向投入。
- **加 pin 是负向的**，且和是否叠加二阶修正无关：单独 importance pin（LongBench
  0.1363、niah 0.0313）比 vanilla 还差；2nd order + pins（LongBench 0.1214、niah
  0.0467）也明显不如 2nd order 不加 pin。**根因大概率已经定位到代码层面**：训练走
  的 `_log_kv_train_lowmem_forward`（`model.py:724-725`）从头到尾没有任何 pin 相关
  代码；`_log_kv_select_pins`（`model.py:865`，`@torch.no_grad()`）只在推理路径里
  被调用（`model.py:1013-1020`，`cache.pin_size > 0` 时）。`arc_warmup.yaml` 虽然
  `config: base.yaml` 继承了 `log_kv_pin_size: 256`，但这个值对训练**完全没有效果**
  ——训练根本不构建 `LogStructuredKVCache`，不会走到 pin 选择逻辑。也就是说**模型从
  没在训练里见过"精确 pin token + 同一 token 被池化稀释的粗糙副本同时存在"这种输入
  结构**，pin 是纯推理期外挂（`log_kv_cache.py:357-361` 注释："pins only ADD
  duplicates, hierarchy still pools them, state trajectory bit-identical
  with/without pins"）。这个"多一个精确副本只会帮忙不会有害"的设计假设，被这组
  实测数据推翻了。第 4 节那条"训练用非 0 pin、推理改成 0 pin 是安全的，两者可以自由
  不一致"的备注需要重新审视——**这个备注反过来才是更接近真相的：train/eval 在 pin
  这件事上必须不一致，因为训练那条路径根本不存在"有 pin"这个状态**。
- ACC（常识推理）四组几乎不分伯仲（0.611–0.616），短程信息在所有压缩方案下都保留
  得不错，真正拉开差距的是长程检索类任务（LongBench、niah）。
- **重新解读第 2 节的核心问题**：LogKV vanilla（纯均值池化，不开二阶修正、不开 pin）
  在 32768 长度上 niah 已经是 0.032，相对 dense baseline 的 1.0 几乎全灭。二阶修正
  把它拉到 0.0827（+2.6×）是真实的正向贡献，但**绝大部分损失在纯均值池化阶段就已经
  发生**——32768 长度下大量 token 被压进远超"width≥8 开始失真"量级的槽（B=512，
  recent_size=1024，实际宽度可能是 64/128 甚至更宽），rank-1 近似哪怕完全精确，
  表达能力也不足以在这么宽的槽里救回一个孤立事实。这正是 pin 机制本来要解决的场景
  （`log_kv_cache.py` 注释："uniform 2:1 mean-pooling dilutes a distant
  low-redundancy fact ... pins"），但目前 pin 是负贡献而不是正贡献。**结论：比起
  继续抠 rank-1 精度，修好 pin 大概率是更高杠杆的下一步**（pin 本来就是为了兜住这个
  确切失败模式设计的）。

**pin 诊断代码已实现（2026-08-07，待在目标评测环境跑真实 checkpoint）**：
`litgpt/litgpt/log_kv_pin_diag.py` 会在真实 `generate_until` 路径 prefill 后读取每层
`_log_kv_pin_indices`，把它们转回原始 prompt token 坐标，并跟 NIAH prompt 里的真实
needle sentence token span 对比；`eval.py` 新增默认关闭的
`--log_kv_pin_diag_output` / `--log_kv_pin_diag_radius` /
`--log_kv_pin_diag_max_samples` / `--log_kv_pin_diag_include_indices`。

建议先跑少量样本：
`torchrun --nproc_per_node=1 eval.py --config exp/qwen1.7b-32k/eval.yaml --benchmark niah_single_1 --limit 4 --log_kv_pin_diag_output <out-dir> --log_kv_pin_size 256 --log_kv_pin_diag_max_samples 4`

如果已有 prompt dump，也可以跑：
`python unused/diagnose_log_kv_pins_jsonl.py --samples <samples.jsonl> --checkpoint_dir <ckpt> --tokenizer_dir <tok> --output <pin_diag.json> --limit 4`

输出 JSON 的 `pin_diag.verdict` 读法：
- `pins_often_hit_needle_or_neighborhood_check_train_infer_pin_mismatch` → pin 多数选中
  needle 或近邻，但指标仍差，更支持"训练从没见过 pin 态"的训推分布不一致假设。
- `pins_mostly_miss_needle_check_pin_selection` → pin 多数没选到 needle，优先查
  `_log_kv_select_pins` 本身（`log_kv_pin_obs_window=64`、`kernel_size=7` 聚类等）。

### 6.2【已解决，2026-08-07】warmup CPT 训练（1500 步，scale=0.2, warmup=100）已完整跑完

用户确认 6.1 表格里的数据就是完整 1500 步训练后的结果，不再是中间 checkpoint。

### 6.3（优先级低于 6.1 的 pin 排查，仅为设计参考）是否要做按 layer/width 差异化的 `second_order_scale`

6.1 的干净数据已经证明 width≥8（32768 长度）在匹配 scale 下确实有明显下游损害，但
6.1 结尾的重新解读认为**大部分损害发生在纯均值池化阶段**（vanilla niah 就已经
0.032），二阶修正只是部分缓解；真正对症的机制（pin）目前是负贡献。所以这里的判断
调整为：**先把 6.1 里 pin 的根因查清楚、pin 能不能修好，再回头评估 6.3 这个接口改动
的 ROI**——如果 pin 修好后 width≥8 的损害大幅缩小，那么 per-width scale 这个接口
改动的必要性会显著降低；如果 pin 修好后仍有明显损害，再考虑这个方向（3.1 节的诊断
门控已经验证过"只在 width≤N 生效"在数值上是可行的，接口改动主要是把这个门控从诊断
专用搬到生产路径，并想清楚训练时怎么处理这个新维度的超参）。**不要在 pin 排查有
结论之前动生产接口**。

### 6.4 pin 诊断真实数据 + 离线分析工具（2026-08-10）

**离线分析工具：`unused/pin_diag.py`（新增）** 直接读 `eval.py
--log_kv_pin_diag_output` 产出的 JSON，不用重新跑评测：

```bash
python unused/pin_diag.py ./pin_diag_smoke/pin_diag_xxx.json
python unused/pin_diag.py './pin_diag_smoke/*.json' --band 23-26   # 支持 glob，按文件名排序取最后一个
```

功能：①样本级"任意层/组至少命中一次"的比例；②同一比例限定在 `--band` 指定的层
区间内；③原样重打 JSON 里已有的 group 级 `overall`/`by_layer` 摘要；④用相同
`pin_size`/`recent_size`/`radius`、相同的真实 needle span，对候选区间做纯随机
采样算出的基线命中率（`--seed`/`--recent_size`/`--pin_size` 可覆盖，默认读 JSON
里落盘的 `config`）。代码审查结论（合成 JSON 跑通全部分支，含 0 样本等边界情况）：
**没有功能性 bug**，唯一小瑕疵是 `_load_json` 对 glob 结果按文件名字符串排序取
"最后一个"当作最新——同一 benchmark+checkpoint 反复重跑没问题（定长时间戳后缀能
正确排序），但混用不同 benchmark/checkpoint 名时不保证真的是最新，不常用不算
紧急。

**⚠️ 一次值得记住的教训：第一次跑诊断时 `log_kv_pin_size` 传成了 64，不是生产用的
256**——随机基线打印出来的 `pin_size=64` 才发现这个问题（工具会把它从 JSON 的
`config.log_kv_pin_size` 里如实读出来打印，这也是留这行打印的价值）。**pin_size
不同，真实命中率和随机基线都不可比**，纠正前的 13.2%/4.1% 这组数字已经作废，
下面是用正确 `log_kv_pin_size=256`（跟 base.yaml/eval.yaml 生产配置一致）重跑
120 个样本（niah_single_1/2/3，radius=16）后的数据。

**核心发现：exact 稳定超过随机基线，但 near 追平或干脆不如随机基线，且这个模式在
所有层（包括表现最好的层）都成立**：

| | exact_hit_rate | near_hit_rate |
|---|---:|---:|
| 随机基线（同 pin_size=256/recent_size=1024/radius=16）| 16.0% | 35.0% |
| 真实 pin（全部 26880 组 sample×layer×group 平均）| 17.2% | 17.4% |

按层看表现最好的几层（层号→exact / near，随机基线是 16.0% / 35.0%，对所有层都一样，
因为候选区间、pin_size、needle 位置不随层变化）：

| 层 | exact | near | exact 相对随机 | near 相对随机 |
|---|---:|---:|---:|---:|
| 23 | 29.2% | 29.9% | 1.83× | 0.85×（不如随机） |
| 24 | 34.2% | 34.6% | 2.14× | 0.99×（追平） |
| 25（最佳）| 34.5% | 35.0% | 2.16× | **1.00×（打平）** |
| 26 | 32.2% | 32.6% | 2.01× | 0.93×（不如随机） |
| 27 | 19.2% | 19.2% | 1.20× | 0.55×（明显不如随机） |

浅/中层的倍数更低，多数层 exact 相对随机在 1.2-1.5× 左右、near 相对随机普遍在
0.6-0.85× 之间（不如随机）。**没有一层的 near_hit_rate 真正超过随机基线**——即便
是全模型表现最好的 Layer 25，也只是打平，没有反超。

**这彻底改变了之前"pin 选择机制约等于随机"或"选择机制整体在起正向作用"两种猜测**，
指向一个更具体、更可操作的机制性解释：`_log_kv_select_pins`（`model.py:868-928`）
用 `salience.topk(n_pin)` 直接取分数最高的 256 个候选，中间只有
`max_pool1d(kernel_size=7)` 做局部平滑——**没有任何促使被选中的 256 个位置在空间上
分散开的机制**。如果显著性分数本身在少数"泛泛而显著"的位置（重复模板句、标点、
attention sink 一类的 token）上抱团扎堆，topk 会把预算浪费在挤在一起的少数几个
"热点"里，而不是像随机采样那样天然均匀撒开——这样即使真正的 needle 恰好不在这些
热点附近，纯随机采样反而因为"撒得更开"而更容易蒙对附近，因此 near_hit_rate 反而
不如随机。这也能同时解释为什么 exact_hit_rate 却稳定超过随机：深层确实学到了一部分
"这个 token 是 needle" 的真实信号，topk 会精确抓住它，但抓不住的时候，剩下的预算
分布方式比随机差。

**样本级"任意层/组至少命中一次"（96.7%，120 个样本里 116 个）需要打折扣看**：
每个样本有 28 层 × 8 组 = 224 次独立机会，只要单次 near_hit_rate 有 35%（哪怕是
纯随机基线的水平），`1-(1-0.35)^224` 已经和 1 没有可辨识的差距——**这个 96.7% 本质
上是"机会数够多"的数学必然结果，不能当作 pin 机制样本级可靠的证据**，换成随机基线
去跑大概率也是接近 100%。这个指标要么按更小的机会数（比如只看单层）算，要么直接
弃用。

**结论（比 6.1 末尾"训推分布不一致"的猜测更进一步）**：现在有证据表明
`_log_kv_select_pins` 的选择质量本身就是一个独立、可定位的问题——不是"训练没见过
pin 态"单独就能解释的（那个解释针对的是"模型不知道怎么用已经选对的 pin"，但现在
看来很多时候 pin 压根没选对，甚至不如瞎选）。两个机制可能同时成立、互相独立地拖累
指标：①`_log_kv_select_pins` 的 top-k 策略缺乏空间分散性，near-hit 覆盖率比随机
还差；②即使某次真的选中了 needle，训练路径从没见过这种"精确 pin + 池化稀释副本
并存"的输入，模型不一定会用。**下一步优先级更新为**：(a) 抽几条样本把 256 个真实
pin 的位置摊开看分布，确认"抱团"猜想是否成立；(b) 如果确认，尝试给 `_log_kv_select_pins`
加一个简单的空间分散约束（比如挑 topk 之后对选中位置做最小间距抑制，或者把候选区间
分桶、每桶限额），比"先做训推一致性"更容易在选择层面直接见效；(c) 6.1 末尾"训推
分布不一致"的假设仍然值得保留和验证，但排在 (a)(b) 之后。

### 6.5 假设：CPT 训练可能在侵蚀 pin 打分依赖的稠密检索能力（2026-08-10，未验证）

**背景**：`_log_kv_select_pins` 算 salience 用的是原始、未压缩的 prefill K（候选区间
`[0, C)` 是压缩发生之前的完整精确 token）做标准点积注意力——`attn = softmax(obs_q @
k.mT * scale)`，没有任何专门为 pin 训练过的参数。所以 salience 分数好不好，本质上
问的是"模型的原生**稠密**注意力，能不能在长上下文里精准聚焦到相关内容"，这是一个跟
LogKV 压缩本身无关的能力。

**风险**：`CausalSelfAttention.forward`（`model.py:723-724`）显示，只要
`self.training_log_kv` 为真且是训练调用（`input_pos is None`），就无条件路由到
`_log_kv_train_lowmem_forward`（压缩/流式路径）——**warmup CPT 训练全程不会执行到
稠密的 `scaled_dot_product_attention`（`model.py:792`）**，demo.py 文件头也写明
"This script has no dense route"。也就是说，CPT 训练从头到尾都在优化"怎么通过压缩
后的层级读信息"，从未有梯度信号去维持"稠密全量注意力下怎么精准定位相关 token"这个
能力——而这正是 pin 打分依赖的能力。持续做这种纯压缩路径的 CPT，理论上有可能让模型
预训练阶段具备的稠密检索能力逐渐漂移/退化（也可能影响很小，或者 base 模型本来就一般，
这是一个待验证的假设，不是结论）。

**怎么验证（成本可控，不需要新训练）**：对比 base（CPT 之前）checkpoint 和 warmup
CPT 之后 checkpoint，在**稠密**注意力（不开任何 LogKV 压缩）下的 NIAH 表现。障碍：
`eval.py` 的 `LogKVLM` 封装目前**只会调用 `set_log_kv_cache`（压缩路径），完全没有
走稠密 `set_kv_cache` 的分支**（`eval.py:475` 注释："There is no dense fallback"），
需要小改一下加个"dense 模式"开关，或者绕开这套封装直接用 litgpt 自带的通用
`generate` 脚本跑一个简化版 NIAH。底层 `GPT` 类本身已经有标准稠密 `KVCache`
（`model.py:1958`）和 `set_kv_cache`（`model.py:300`），不需要新写压缩相关代码。

**如果验证成立的后续方向**：CPT 训练配比里混入一部分稠密注意力的批次，防止稠密检索
能力被压缩训练目标"挤掉"；如果不成立（base 模型本来就一般，不是 CPT 的锅），提升
salience 质量就要靠更大投入的专门检索监督或检索类数据继续预训练，优先级低于修选择
算法（见 0 节"下一步方向"阶段 3(f)(g)）。

### 6.6 抱团扎堆已实锤 + `pin_diag.py` 新增的空间分布分析（2026-08-10）

**`unused/pin_diag.py` 新增了两块功能**（用户实现）：
- `print_pin_distribution_summary`：用 `--log_kv_pin_diag_include_indices` 记录的
  完整 `pin_indices`，对每个 (sample, layer, group) 算直方图占用 bin 数、归一化熵、
  相邻 pin 间隔、以及若干固定宽度（64/128/256/512 token）滑动窗口内的最大 pin 密度，
  并打印密度最高的 top-N group 的 ASCII 直方图。
- `print_compressed_slot_summary`：读取 `needle_covering_slots`（见下面的已知问题），
  统计 needle 落在压缩 level slot 还是 recent 精确窗口。

**空间分布结果证实了"抱团扎堆"猜想，而且是典型现象，不是极端个例**（23-26 层，
3840 个 group）：32 个 bin 的占用比例中位数只有 25%（约 8/32），单个最热 bin 中位数
占了 38.3% 的 pin，最密 512-token 窗口中位数占比 35.5%，归一化熵中位数 0.471（1.0
才是均匀分布）。极端例子（如 `sample=104 layer=23 group=4`）里，256 个 pin 中
250 个（97.7%）挤在一个 512-token 窗口里，而这个 group 的"span"却高达候选区间的
98.7%——span 这个指标会被少数跑到候选区间另一端的离群点带偏，看聚集程度应该看
`dense{window}`/`max_bin_frac` 这几列，不要只看 span。**结论：阶段 1(b) 确认通过，
可以直接进入 1(c) 给 `_log_kv_select_pins` 加空间分散约束**，不需要更多证据。

**已知问题（不阻塞，之后再查）：`print_compressed_slot_summary` 的输出全是 0%/n/a，
大概率是 bug，不是真实结论**。`group-level compressed coverage: 0/0` 说明每个
group 的 `needle_covering_slots` 都是空列表——连"needle 落在 recent 精确窗口"都是
0%，这不合理（`_cache_slot_layout`，[log_kv_pin_diag.py:221-291](litgpt/litgpt/log_kv_pin_diag.py#L221-L291)，
理论上会把当前缓存里所有 level + recent 累计拼起来，应该完整覆盖 needle 所在位置，
不管它被压缩到多深）。两个怀疑方向：①`record()` 是在整个 `generate_until` 生成流程
跑完之后才读缓存状态，不是紧跟在 prefill 之后——此时缓存已经因为解码新 token 又做过
若干次进位合并，跟 pin 刚选出来那一刻的快照不是同一个状态；②`_cache_slot_layout`
自己的坐标/计数有 bug——它已经把自检字段 `layout_matches_token_count`
（[log_kv_pin_diag.py:289](litgpt/litgpt/log_kv_pin_diag.py#L289)）记进了每个
group 的 `slot_layout` 字段，但 `pin_diag.py` 当前没有打印它，下次要查这个问题时
第一步就是把这个字段打出来看是不是 False。这个功能不影响阶段 1(c) 的判断，先不修。

### 6.7 阶段 1(c)：给 `_log_kv_select_pins` 加 NMS 空间分散约束（2026-08-10，已实现并通过审查）

**改动**（六个文件，都已提交，无未提交改动）：
- `litgpt/model.py`：`GPT.set_log_kv_cache()` 新增 `pin_min_distance` 参数；
  `CausalSelfAttention` 新增 `log_kv_pin_min_distance` 属性；新增静态方法
  `_log_kv_select_nms_indices()`——按显著性从高到低贪心选 pin，每选一个就抑制周围
  `min_distance` 范围内的候选，间距不够时用剩余最高分点回填以保证 `pin_size` 不变；
  `_log_kv_select_pins()` 里 `log_kv_pin_min_distance <= 1` 走旧 `topk`，`> 1` 走
  NMS。
- `eval.py`/`demo.py`/`eval.sh`/`majob.sh`/`unused/diagnose_log_kv_pins_jsonl.py`：
  新增 `log_kv_pin_min_distance` 参数，从 CLI/YAML 一路串到
  `model.set_log_kv_cache()`，默认值全链路统一是 `0`（等价旧行为）。
  `majob.sh` 的 `read -r` 位置变量列表和 Python 端 `print()` 的输出顺序同步加了
  新字段，避免后面的 `LOG_KV_SECOND_ORDER_SCALE`/`SAVE_CKPT` 等字段被串位。

**审查结论：核心算法正确，已用合成数据验证**（不是只读代码，实际跑了一遍）：
构造一个"抱团"显著性分布（4000 候选里塞一个 50-token 宽的超高分热点，模拟真实
观测到的"250/256 挤在一个窗口"），top-k 在 100-token 窗口内集中度 79.7%，加了
`min_distance=32` 的 NMS 降到 4.7%；相邻 pin 最小间距实测精确等于设定值；
`min_distance` 大到无法满足间距时触发回填，验证过不会死循环、不会重复索引、
最终数量精确等于 `n_pin`；`n_pin == C` 边界正常。六个文件的参数默认值、
`majob.sh` 的位置变量顺序都逐一核对过，没发现功能性 bug。

**唯一非阻塞的顾虑：性能**。`_log_kv_select_nms_indices` 是纯 Python 嵌套循环
（每个 batch×group 一次，候选列表可达 3 万+），没有向量化、跑在 CPU 上——而且
"高分挤在一起"恰恰是最容易让 NMS 需要多走几步候选列表的输入分布，也正是这次要修的
场景。28 层 × 8 组每次 prefill 都要跑一遍，建议在真实评测上留意一下墙钟时间有没有
明显变慢，不用现在就优化。

**下一步（阶段 1(d)）**：用非 0 的 `--log_kv_pin_min_distance`（比如先试 32 或 64）
重新跑一次跟 6.4/6.6 同样配置的 pin 诊断 + `unused/pin_diag.py` 随机基线对比，
确认 near_hit_rate 有没有真正甩开随机基线的 35%（不是像之前那样打平或更差），
以及"Pin spatial distribution"那几个聚集指标（bin 占用比例、熵、密度窗口）有没有
明显改善。确认后再进入阶段 2（跑真实 niah/longbench 看下游指标是否好转）。

### 6.8 阶段 1(d)：`log_kv_pin_min_distance=64` 验证通过，near-hit 反超随机基线（2026-08-10）

**确认参数**：`log_kv_pin_min_distance=64`（`exp/qwen1.7b-32k/diag_warmup.yaml:33`）。
其余配置跟 6.4/6.6 一致（`log_kv_B=512`、`log_kv_recent_size=1024`、
`log_kv_prefill_block=1024`、`log_kv_pin_size=256`、`log_kv_pin_obs_window=64`，
niah_single_1/2/3，32768 长度，`limit=40`，80 个有效样本）。

**结果，对比 6.4/6.6 的无约束 top-k**：

| | exact_hit_rate | near_hit_rate |
|---|---:|---:|
| 随机基线（不变）| 15.8% | 34.3% |
| 无约束 top-k（6.4/6.6）| 17.2% | 17.4%（**不如随机**）|
| + NMS min_distance=64（本次）| **50.6%** | **65.1%**（**1.9× 随机**）|

空间分布也从"典型 group 就已聚集"（32-bin 占用中位数 25%、熵 0.471）变成接近均匀
（占用中位数 96.9%、熵 0.960，Top 12 最密 group 全部只有 `8/256 (3.1%)` 落在最密
的 512-token 窗口里，不再有异常聚集）。层间模式也从"只有 23-26 层管用"变成
"9-27 层普遍 exact 50-72%、`median_dist=0`"，连最差的层（Layer 5）也比修之前
所有层的最好成绩都高。工具内置的 `verdict` 从
`pins_mostly_miss_needle_check_pin_selection` 翻成了
`pins_often_hit_needle_or_neighborhood_check_train_infer_pin_mismatch`。
**阶段 1 到此可以认为完成**：选择质量已经明确超过随机基线，不再是主要瓶颈。

**更精确的机制解读（用户指出，纠正了 6.6 的表述）**：抱团本身不是问题，抱团发生在
needle 附近才是好事；真正的问题是"抱团发生在离 needle 很远的地方，而且这个错误的
热区宽到能吃光全部 256 个名额"——旧数据里 Top 12 最密 group 动辄
`count=250/256 (97.7%)` 挤在一个 512-token 窗口，说明那个假热区本身至少有几百个
token 的分数都异常偏高，宽到足以让 256 个名额完全填不到别处，needle 所在区域因此
一个名额都拿不到。NMS 起效的原理跟"知道 needle 在哪"无关，它只是不允许任何一个
区域垄断全部预算——全局最高分那个点在 NMS 里永远最先被选中（跟旧 top-k 一样），
只有排名靠后、落在同一热区里的候选会被分流到别处，所以 NMS 对"少量名额、真的挨着
needle"的情况几乎没有伤害，只掐掉"大量名额、不挨着 needle、还抱团"这种情况。

**关于假热区成因的一个具体猜想（待验证，不阻塞后续工作）**：RULER 风格的 NIAH
任务经常在 haystack 里埋多个用同一模板写的"干扰句"（只有一处的数值是真正问的那个），
如果显著性机制认出的是"这个句式重要"而不是"这个具体数值是答案"，会对所有干扰句
都打高分；`max_pool1d(kernel=7)` 会把某一条干扰句的峰值进一步平滑加宽。修 NMS 后
Top 12 最密 group 的密集窗口位置分散在候选区间的很多不同地方（不是同一个固定位置），
支持"每条样本有各自的干扰句位置"而不是"某个全局固定伪影"。验证方法：把修 NMS
*之前*那份 JSON 里最夸张的抱团例子（如 `sample=104 layer=23 group=4`）对应的
token 区间在原始 prompt 里抠出来，看是不是跟 needle 同模板的干扰句。

**下一步（阶段 2，已启动）**：用这套 `log_kv_pin_min_distance=64` 配置跑真实
niah/longbench 下游评测（`majob.sh`，`base.yaml` 已加上这一行），确认"+pin"是否
从"比 vanilla 还差"变成有正贡献——这个评测本次会话结束时已经在跑，结果见 6.9
（负面）。

### 6.9 阶段 2 结果：pin 全系列（含 NMS 各档、极小 pin_size）均跑输 vanilla，确认是剂量效应（2026-08-11）

**五组配置的真实下游对比**（同一 warmup CPT checkpoint，`second_order_scale=0.2`，
其余压缩参数一致，只切换 pin 相关三个参数；niah 是 niah_single_1/2/3 在 32768
长度上的均值）：

| 配置 | pin_size | min_distance | LongBench | LongBench_e | niah（均值）|
|---|---:|---:|---:|---:|---:|
| 不用 pin | 0 | – | **0.1716** | **0.1918** | **0.0827** |
| top-k（无 NMS），pin=4 | 4 | 0 | 0.1507 | 0.1624 | 0.0727 |
| top-k（无 NMS），pin=256 | 256 | 0 | 0.1214 | 0.1331 | 0.0467 |
| NMS，pin=256 | 256 | 16 | 0.1327 | 0.1457 | 0.0113 |
| NMS，pin=256 | 256 | 64 | 0.1313 | 0.1429 | 0.0200 |

（后两行 min_distance=16/64 是用 6.7/6.8 修好的 NMS 选点算法产出的配置；"不用 pin"
和 "top-k pin=256" 两行数字分别就是 0 节表格/6.1 表格里的"+2nd order, no pins"和
"+2nd order + pins"——同一份数据，这里把 pin 那一档进一步拆成四个变体做对照。）

**核心发现 1：剂量效应，不是选择精度问题**。三个 benchmark 上排名完全一致：不用
pin 最好，pin=4 稳居第二且始终离第一名最近，四个 pin=256 变体（不管 NMS 怎么调）
都排在后面、离第一名更远。6.7/6.8 已经确认 NMS 能把选点的 exact/near-hit 率从
不如随机修到 1.9× 随机，但这组下游数据显示**选点选得越准，pin=256 也没有更接近
vanilla**——说明选择质量从来不是（或不再是）下游变差的主因，真正起决定作用的是
"插入了多少个精确槽"：插得越多、伤害越大，哪怕插得很少（4 个）也还是负收益，只是
负得更少，从没转正。

**核心发现 2：NMS 对两类任务的影响方向相反**。LongBench/LongBench_e 上 NMS
（min_distance=16/64）比无约束 top-k(256) 好；niah 上反过来，无约束 top-k(256) 比
两个 NMS 版本都好（0.0467 > 0.02 > 0.0113，NMS-16 是 niah 上最差的配置）。跟 6.8
"抱团在需要的地方是好事"的解读一致：LongBench 类任务的相关信息本来就分散在文档
多处，NMS 强制打散预算是真的在帮忙；niah 只有一个精确 needle，最优策略是把预算
冗余地砸在那一个点上，NMS 恰恰在拆散这种该抱团的地方。**结论：NMS 调的是"伤害往
哪类任务分配"，不是"要不要有伤害"**——两种任务下 pin=256 都追不上 vanilla，更追
不上 pin=4。

**一个能同时解释"越多越伤"和"哪怕很少也伤"的机制猜测（待验证，不阻塞下一步）**：
压缩层级的 pooled slot 由多 token mean-pool 得到，点积分数大概率被磨平（幅度/方差
偏小、偏集中）；pin 槽是未池化的原始精确 key，点积分数量级、分布很可能明显不同。
同一个 softmax 行里混入哪怕几个"分布外量级"的分数，会重新分配整行的注意力质量——
不需要 pin 位置有问题，只要它的分数尺度和其它槽不匹配，就足以扰乱这个 query 对其它
（本来该关注的）位置的正常注意力。这个假设预测"插得越多扰动越大"（吻合剂量效应），
也预测"哪怕只插 1~4 个也有伤害"（是尺度失配，不是实现 bug——剂量关系平滑单调，
不是 bug 常见的阈值性/不连续表现）。

**结论**：阶段 1（选择质量修复）是真实有效、已经确认的改动，但**它解决的问题不是
下游变差的瓶颈**——瓶颈已经转移到"pin 机制本身跟当前训练分布不匹配"。继续在
NMS/选点算法上调参预计不会再有实质性突破（min_distance 从 64 调到 16、再到"极小
预算"pin=4 这三个方向都试过，负收益的方向没有变过，只是幅度和在哪类任务上更明显
在变）。下一步方向见 0 节"阶段 4"。

### 6.10 阶段 3：dense-mode base vs warmup CPT 真实跑出结果（2026-08-12）

**改动**（上次会话已实现、代码审查无 bug）：`eval.py`/`litgpt/model.py` 新增
`log_kv_dense_mode` 开关，为真时 `LogKVLM._set_eval_cache()` 走
`self.model.set_kv_cache(...)`（标准 KV cache），完全绕开 LogKV 代码；启动时校验
跟 `log_kv_diag_mode`/`log_kv_pin_diag_output`/`log_kv_pin_size!=0` 互斥。新建了
`exp/qwen1.7b-32k/dense_niah_base.yaml`（`checkpoint_dir` 指向原始 Qwen3-1.7B-Base，
`log_kv_dense_mode: true`，`log_kv_pin_size: 0`），跟已有的 `arc_warmup.yaml` +
`DIAG_ARGS="--log_kv_dense_mode true --log_kv_pin_size 0"` 配合，分别对 base 和
warmup CPT checkpoint 跑同一套 niah_single_1/2/3。

**结果**：

| 指标 | base（未 CPT）| warmup CPT | 变化 |
|---|---:|---:|---|
| Common sense | 0.6151 | 0.6123 | 微降 ~0.3pp |
| LongBench | 0.013 | 0.2075 | 约 16 倍 |
| LongBench_e | 0.2606 | 0.4463 | +71% |
| niah（稠密）| 0 | 0.9353 | 从完全不会到接近满分 |

**解读**：
- **CPT 训练确实教会了模型长上下文检索能力**（稠密 niah 0→0.9353），长文档理解
  （LongBench/LongBench_e）也大幅提升，common sense 只有噪声量级的微降（持续预训练
  在长文档语料上续训导致的轻微"遗忘"，是常见、预期内的副作用，不是训练配方出了
  问题）。
- **把这个 0.9353 稠密上限跟 6.1 表格的"LogKV + 2nd order, no pins"（niah 0.0827）
  对比**：同一个 checkpoint，唯一变量是开不开压缩，niah 从 0.9353 掉到 0.0827——
  **压缩本身（完全不涉及 pin）就吃掉了 85 个百分点以上**，比整个 pin 系列实验
  （6.9，各配置在 0.011~0.047 之间来回摆）影响大一个数量级。这把"压缩本身的信息
  损失"重新确立为全链路里最大的单一瓶颈，pin 一直以来只是在争夺一个本身就很小的
  剩余空间。
- **重要限定（不影响上面的结论，但影响怎么解读这个数字本身）**：CPT 训练全程走
  `_log_kv_train_lowmem_forward`（`log_kv_train_block=16384` 的分块流式训练，
  block 之外的历史全部走压缩+detach），模型从没有在真正的稠密因果注意力（每个
  token 精确可见、梯度能穿透到任意历史 token）下训练过。所以 dense_mode 测出来的
  0.9353，严格说是"压缩训练出来的权重，泛化到一个训练时从未见过的、信息更完整的
  输入分布"的结果，不是一个可以直接当作训练目标的"干净上限"。这个混淆因素如果有
  方向性，应该是让这个数字偏保守（信息更完整通常只会帮忙，但分布不匹配本身是训练
  从没经历过的），所以真实的"如果从头到尾用稠密注意力训练"的上限很可能比 0.9353
  更高——不会推翻"压缩代价远大于 pin 代价"这个结论，只是提醒不要把 0.9353 直接当
  成一个精确、无偏的物理量。

### 6.11 新增独立诊断：`log_kv_pin_score_diag`（pin vs pooled 分数/mass 尺度对比，2026-08-12）

**动机**：6.9 结尾提出的机制假设——pooled slot 是多 token mean-pool 出来的，点积
分数被磨平；pin 槽是未池化的精确 key，分数尺度可能明显不同，混进同一个 softmax
行会扰乱整行的注意力质量分配，不需要 pin 位置选得准不准。已有的 `log_kv_diag.py`
明确要求 `pin_size=0`（slot→token 映射假设槽覆盖连续 token 区间，pin 是打散的
精确重复，破坏这个假设），不能直接拿来验证这个假设，所以新增了一个独立、更轻量
的旁路诊断，不复用、不修改 `log_kv_diag.py`。

**实现**（三个文件，代码审查无 bug，两条分支——GQA/MHA——都挂了钩子）：
- 新增 `litgpt/log_kv_pin_score_diag.py`：`PinScoreDiag` 全局单例，
  `capture_score_stats()` 在 `scores.mul_(scale)` 之后、二阶修正和 `log(w)` mass
  bias 加上去之前调用，拿到纯 `dot_pin_score`/`dot_pooled_score`；`record()` 在
  `attn = softmax(scores)` 算完之后调用，用同一份生产 buffer 拿到
  `final_pin_score`/`final_pooled_score`（加完修正、softmax 前）和按槽归一化的
  `pin_mass_per_slot`/`pooled_mass_per_slot`/`pin_to_pooled_mass_per_slot_ratio`
  （及其 log10 版本）。全程复用生产路径的张量，不额外重算，避免诊断和生产口径
  不一致。
- `litgpt/log_kv_cache.py` 的 `log_kv_slot_attention()` 新增可选的 slot 边界参数
  （`pin_slot_range`/`pooled_slot_range`），GQA（`nh != nkv`）和 MHA 两条分支都
  加了被 `LOG_KV_PIN_SCORE_DIAG.enabled` 门控的记录调用。
- `litgpt/model.py` 的 `_log_kv_training_forward` 只在 vectorized prefill block
  里、且只在 `cache.token_count == start`（fresh prefill，跟已有 `LOG_KV_DIAG`
  oracle 诊断共用同一个 precondition，天然排除 decode/pending 阶段）时计算
  `[pooled][pin][recent][causal_tail]` 的边界，并且只在 query 位置落在
  `log_kv_pin_score_diag_window_from_end`（默认 512，落地实验用了 1024）窗口内
  才记录——避免把生成阶段的 decode query 和 prompt 中段无关的 query 混进统计。
- `eval.py` 新增 `--log_kv_pin_score_diag_output`/`--log_kv_pin_score_diag_window_from_end`，
  启动时校验不能跟 `log_kv_dense_mode`/`log_kv_diag_mode` 同开、必须
  `log_kv_pin_size > 0`；多卡用 `dist.all_gather_object` 收集各 rank 的
  `state_dict()`，rank 0 落盘 JSON（`total`/`by_layer`/`by_layer_branch` 三级
  聚合）。

**第一次真实结果**（warmup CPT step_1400，`pin_size=256`/`min_distance=26`，
niah_single_1/2/3，`window_from_end=1024`，120 个样本，全部 28 层）：

- **典型情况（log10 均值，更抗离群值）：pin 反而略微"吃亏"**——
  `pin_to_pooled_mass_per_slot_log10_ratio` 在**全部 28 层都是负的**（-0.03 到
  -0.68，换算成倍数是 0.2~0.9 倍）。但纯点积 `dot_pin_score` 全部 28 层无一例外
  都比 `dot_pooled_score` 高——证实两者分数尺度确实不一样，`log(w)` mass bias
  在 27/28 层把这个优势基本抹平甚至反超（只有 layer 0 例外），说明现有的 mass
  bias 补偿机制在"典型情况"下工作得还不错。
- **但存在剧烈的偶发"塌缩"**：`pin_to_pooled_mass_per_slot_ratio` 算术均值 9.09，
  std 高达 36546，max 到 2.5 亿——分布极端右偏，`pin_mass_total.max ≈ 1.0`（某些
  query 上几乎 100% mass 全给了 pin）。这种塌缩不是少数样本的特例，**120 个样本
  全部都有**（每个样本在整个窗口里见过的最大 log10_ratio 从 2.77 到 8.41 不等，
  换算成倍数最小也有约 589 倍）——但塌缩集中发生在窗口内**某个位置**，不是发生在
  离生成最近的最后一个 query：`last_query` 的 log10_ratio 在 120 个样本里全部
  温和（-0.13 到 -0.56），跟"最极端时刻"完全不是一回事。

**结论**：塌缩是普遍存在的现象，但不集中在模型真正要用检索信息生成答案的那一刻，
且这个塌缩程度是否命中真实 needle，需要跟 `log_kv_pin_diag_output` 的命中率数据
交叉验证——见 6.12。

### 6.12 sample_id 跨诊断关联 + 大规模相关性分析（2026-08-12）

**动机**：6.11 的塌缩事件到底是"模型精准命中 needle 后合理全力押注"还是"跟
needle 位置无关的伪影"，需要把 `pin_score_diag`（塌缩程度）和 `pin_diag`（命中
真实 needle 与否）两份独立的诊断数据按同一个请求关联起来看，两者之前互不知道
对方的存在。

**实现**（三个文件，代码审查无 bug，重点核对过跨 rank 唯一性/多线程串号/内存量级/
join 语义四个风险点，逐一确认没问题）：
- `eval.py` 的 `generate_until()` 在真正调用 `litgpt_generate(...)` 前生成全局
  唯一 `sample_id`（`rank{r}|global_req{n}|task={task}|doc={id}`，`global_req`
  是从跨 rank 条带切分 `requests[dp_rank::dp_size]` 精确还原出的、在完整共享
  列表里的原始下标，本身已经全局唯一，`task=` 段用于跨 task 的 `generate_until`
  调用之间去重）；同一个 id 同时传给 `PinDiagRecorder.record(sample_id=...)` 和
  `LOG_KV_PIN_SCORE_DIAG.set_sample_context(sample_id)`，`finally` 里清空 context
  避免异常时串到下一条样本。整个循环单线程同步执行，不存在交错风险。
- `litgpt/log_kv_pin_diag.py` 的每条样本记录加了 `sample_id` 字段。
- `litgpt/log_kv_pin_score_diag.py` 新增按样本分桶的 `samples` 输出，每个样本只
  存几个标量（`max_pin_to_pooled_mass_per_slot_log10_ratio` 及其倍数、
  `last_query` 下七八个字段），不保留任何张量，内存量级是 O(样本数) 而不是
  O(query×slot)；多卡 gather 用"列表拼接"语义（不是 `_RunningStat` 那种累加
  语义），按 `sample_id` 防御性去重。
- 实测验证（1500 个真实样本）：`sample_id` 100% 唯一无重复，三个 task 各 500
  个、八个 rank 各分到 1/8，跟设计预期完全吻合。

**分析脚本**：`unused/pin_collapse_vs_hit.py`（纯标准库，无第三方依赖），按
`sample_id` join 两份 JSON，排除掉 `comparable_needle_count==0`（needle 被截断
出 prompt）的样本，按 `near_hit_rate`/`exact_hit_rate` 是否 >0 分 hit/miss 两组
对比塌缩程度，同时算 Pearson 相关系数、按 task 拆分、打印塌缩最猛的 top-N 样本，
支持 `--out-csv` 导出完整 join 表。

**核心结论（1500 个真实样本，3 个 niah 任务各 500 个）**：

| 命中质量指标 | 与窗口内最猛塌缩（max_log10_ratio）的相关系数 | 与最后一刻塌缩（last_query）的相关系数 |
|---|---:|---:|
| `near_hit_rate` | +0.089 | -0.173 |
| `exact_hit_rate` | +0.100 | — |
| `min_distance`（离针最近的 pin 有多远）| -0.093 | -0.022 |

命中率本身很健康（97.7% 的样本有 near-hit，`min_distance` 中位数为 0，即绝大多数
样本在某个 layer/group 上有精确命中），但**塌缩程度跟命中质量的相关系数全部在
噪声量级（\|r\|<0.2）**，HIT 组（1465 个）和 MISS 组（35 个）的塌缩均值只差
0.4 个 log10 单位（3.81 vs 3.39），相对组内标准差（0.66/0.46）不算有效差异；
按 task 拆开看，`near_hit_rate` 均值差得不少（0.67/0.46/0.59），但 `max_log10_ratio`
均值几乎一样（3.82/3.79/3.80）。**这推翻了"塌缩=正确检索的极端体现"这个乐观
猜测**（真是这样的话应该看到强正相关），支持 6.9 提出的"分数尺度失配"机制假设：
塌缩更像是跟位置的原始点积量级绑定的现象，跟这个位置是不是语义上正确的答案基本
无关。

**对下一步的意义**：这个结果让阶段 4（训练期 pin 注入，见 6.13）变得更有必要——
问题看起来确实是"模型没学会怎么正确校准精确槽和池化槽混合出现时的分数尺度"，而
不是"选点选得不够准"（阶段 1 已经证明选点精度可以修好，但 6.9 证明修好选点不解决
下游问题）或"塌缩本身就是有意义的信号"（这次证明塌缩和是否命中基本无关）。

### 6.13 阶段 4：训练期随机 pin 注入已实现并启动短续训（2026-08-11~12）

**实现**（四个文件，用户实现，代码审查无 bug，重点核对了 forward/backward 选点
一致性和 `pin_size`/`pin_train_max` 语义不会在 warmup 中途搞混两个高风险点）：
- `litgpt/log_kv_cache.py` 新增 `_sample_training_pin_positions()`（每个 chunk
  边界按 `pin_train_prob` 概率触发，随机挑 `[0, inject_start)` 范围内最多
  `pin_train_max` 个已经流过的历史位置，`inject_start` 严格小于当前位置，不会
  泄漏未来信息）和 `_install_training_pins()`（取这些位置的原始 K/V 调用已有的
  `cache.set_pinned()`）。`LogKVStreamTrainingAttention.forward` 用 `*pin_args`
  变长参数向后兼容旧的 7 参数调用，只采样一次并存进 `ctx`；`backward` 读
  （不重新采样）`ctx` 里存的选点结果，在 `with torch.no_grad()` 里原样重放同一次
  注入，保证 forward/backward 走完全相同的代码路径——这是这类"流式训练+backward
  重放"结构最容易出错的地方，已重点核对过。
- `litgpt/model.py` 新增 `GPT.set_log_kv_pin_training()`，`enable_log_kv_training()`
  新增 `pin_size`/`pin_train_max`/`pin_train_prob` 参数并做范围校验（含
  `pin_train_max > pin_size` 会报错，防止训练配置超过缓存容量）。
- `demo.py` 新增 `get_log_kv_pin_train_schedule()`（线性爬坡，模式跟已有的
  `get_log_kv_second_order_scale` 一致）和 `log_kv_pin_train_max`/`_prob`/
  `_warmup_steps` 三个新 YAML/CLI 参数，训练循环每步据此重设注入强度。
  `demo.py` 里 `enable_log_kv_training(..., pin_size=log_kv_pin_train_max, ...)`
  正确地把"目标/最大值"传给固定容量参数 `pin_size`、把"当前 warmup 进度值"传给
  `pin_train_max`——没有传反（传反会在 warmup 中途缓存容量不够时报错崩溃）。
- `exp/qwen1.7b-32k/base.yaml` 新增三个 `null` 默认的 YAML 字段（遵循 3.3 节的
  CLI 扫参约定）。

**实验配置**：`exp/qwen1.7b-32k/pin_train_shortft.yaml`——`resume_dir` 指向 warmup
CPT checkpoint 纯权重续训（不用 `auto_resume`，优化器/step 全部重新开始），
`learning_rate=1e-5`（比原 CPT 的 5e-5 低，避免破坏已收敛的长上下文能力）,
`log_kv_second_order_warmup_steps=0`（`second_order_scale` 保持恒定 0.2，不重新
爬坡，避免同时引入两个分布变化，方便把效果单独归因到 pin 注入上）,
`log_kv_pin_train_max=256`/`pin_train_prob=0.5`/`pin_train_warmup_steps=150`。

**当前状态**：已经用 `bash majob.sh exp/qwen1.7b-32k/pin_train_shortft.yaml` 启动，
`max_steps` 从最初设计的 300 上调到 1700（已跑到 step 1400，还剩 300 步）。**2026-08-12
修了两个 bug 才能让这次续训真正跑起来**（见 7 节）：①`resume_dir` 原来被设成指向
自己，但 `resume_dir` 无论指向哪都只做纯权重加载、`global_step` 永远从 0 开始——
已改成 `auto_resume: true`，才能真正从 step 1400 接着跑剩下的 300 步而不是重新跑满
1700 步；②`majob.sh` 的跳过训练判断原来只看 `checkpoint_exists`（裸文件存在性），
不看 `global_step` 有没有达到新的 `max_steps`，导致 `save_path` 下已有 step 1400
的 checkpoint 时会误判成"训练完成"直接跳到评测——已改成叠加 `checkpoint_finished`
判断。两处都改完了。**2026-08-12 又发现并修了第三个 bug**：`auto_resume` 修好之后，
用户反馈 loss 波动很大（0.3~2.4）——根因是 `get_log_kv_pin_train_schedule` 的
爬坡比例用的是绝对 `global_step` 算 `ratio = current_step / warmup_steps`，而
`log_kv_pin_train_warmup_steps` 当时写的是 150（按"从 0 开始的续训"设计的）。
`auto_resume` 修好后 `global_step` 正确地从 1400 起算，`1400/150 ≫ 1`，爬坡比例
从续训第一步起就被钳到 1.0——pin 注入从第一步就是满强度（256/0.5），完全没有
按设计意图爬坡，YAML 里"避免骤然引入新分布造成不稳定"这条注释写的初衷因此落空。
已改成 `log_kv_pin_train_warmup_steps: 1550`（= 1400 + 150，把爬坡窗口平移到
从实际续训起点开始算）。同时把 `learning_rate`/`min_lr` 从 `1e-5` 降到 `5e-6`，
给这次续训多留一点安全余量（注意 LR 调低不会让每一步的 loss 数值本身更平滑——
那取决于这一步的 batch 和有没有触发 pin 注入——只是降低单个噪声大的 step 把
权重带偏、造成级联不稳定的风险）。**具体这次续训跑到第几步、什么时候能跑完、
loss 波动有没有收敛，下次接续时需要向用户确认**，本文档写下时还不知道最终结果。

**跑完之后要做的唯一一件事**：用同一套 NMS pin 配置（256/64 或与 6.9/6.11 一致的
具体 min_distance）重新跑 LongBench/LongBench_e/niah，对比 vanilla
（0.1716/0.1918/0.0827）。看这一个结果就够了——追平/反超说明训推一致这条路有效，
继续投入；没用就说明问题更深，评估放弃 pin。**不再追加新的诊断实验**（诊断已经
做得够多，6.4~6.12 已经把机制层面能查的都查过一遍，每次跑评测都是真实 GPU 开销，
之后除非下游数字本身不好解释，否则不再为了"多看一眼机制"单独起新的诊断跑）。

**最终结果（2026-08-13，`pin_train_shortft.yaml` 续训到 step 1700 后评测）**：

| 配置 | common sense | LongBench | LongBench_e | niah |
|---|---:|---:|---:|---:|
| vanilla（无 pin）| 0.6113 | **0.1716** | **0.1918** | **0.0827** |
| pin=256, NMS-64，**未训练**（6.9）| – | 0.1313 | 0.1429 | 0.0200 |
| warmup CPT 基线（6.10，无 pin）| 0.6123 | – | – | – |
| pin=256, NMS-64，**训练期注入后**（本节）| 0.6126 | 0.1328 | 0.1484 | 0.0213 |

训练后的三项下游指标（0.1328/0.1484/0.0213）跟训练前的未训练版本（0.1313/0.1429/
0.0200）几乎完全一致，差异在噪声量级；common sense（0.6126）也跟 warmup CPT 基线
（0.6123）持平，说明这次短续训本身没有破坏模型原有能力，只是**没有起到设计预期的
作用**。跟 vanilla 相比仍然有明显差距（LongBench 差 0.039、LongBench_e 差 0.043、
niah 差 0.061）。

**结论：训推一致假设不成立，pin 这条线到此终止**。回顾整条验证链：选择质量（阶段
1，NMS 修复真实有效但不解决下游问题）→ 剂量效应（阶段 2/6.9，插得越多伤害越大，
插得再少也是负收益）→ 分数尺度机制（阶段 5/6.11/6.12，塌缩确认存在但跟命中质量
无关）→ 训推一致（阶段 4/本节，训练后跟不训练一样烂）。四个独立、互不依赖的方向
依次验证都是负结果，不是同一个假设的不同侧面被同一个 bug 污染——**没有再值得
尝试的、廉价的下一步假设了**。不建议继续在 pin 机制上投入（含"加长这次续训"这个
选项：虽然这次剂量偏轻——满强度注入只有 150 步、50% 概率、LR 5e-6——理论上不能
100% 排除欠训练，但四个方向一致指向同一结论，且阶段 3（6.10）已经证实压缩本身的
损失比 pin 影响大一个数量级，继续投入验证成本换回的信息价值不高）。后续如果还做
LogKV，重心应该转向压缩机制本身（槽宽、二阶修正、pooling 方式），而不是 pin。

### 6.14 压缩本身·方向 1：重要性加权池化已实现（2026-08-13，代码完成，未跑评测）

**动机**：6.10 证实压缩本身（不涉及 pin）吃掉 85+ 个百分点，比 pin 全系列实验大
一个数量级。看 `log_kv_slot_attention` 的打分公式：

```
score_s = scale·(q·k_s) + 0.5·scale²·sos·sigma2·(q·sigma_u)² + λ·log(w_s)
          ↑ 一阶：均值 key，needle 被稀释 1/width    ↑ 二阶：rank-1、只能加不能减
```

二阶修正确实有正贡献（niah 0.032→0.0827，2.6×，见 0 节表格），但它是 rank-1、只能
沿单一主方向修正，只有当一阶的均值 key 没把 needle 冲淡到"不参与竞争"时才救得回
来。`_compact_tokens`/`_flush_pairs` 里 `k_entry = k.mean(dim=2)` 这一行是均匀
1/n 平均——32768 长度、B=512 时槽宽可能到 64/128，needle 被稀释成 1/width。**杠杆
在一阶的 pooled key，不在继续加二阶的 rank**：把 `k.mean()` 换成按 per-token
重要性加权的均值，让池化 key 的内容和位置子通道都更指向重要 token，比继续抠二阶
精度更对症。

**为什么不是 pin 那条老路**：pin 是往压缩层级里混入一个量级迥异的精确 key（6.9/6.11
证实的"分数尺度失配"是 pin 失败的核心机制），而且训练路径完全没有 pin 存在
（`_log_kv_select_pins` 是 `@torch.no_grad()`，只在推理期跑，见 6.1）——训推不
一致是 pin 四个死因之一。重要性加权池化不引入新槽、不改变槽的数量或量级，只改变
"槽的内容怎么算出来"，而且这个加权是 k 的**确定性纯函数**（无新增可学参数），
训练和推理天然用同一份代码算出同一个值，从设计上就规避了 pin 系列的训推不一致
问题。

**设计（六个文件，均已实现并通过单测审查）**：核心是把"槽的 pooling 权重"和
"`log(w)` mass bias 用的 token 计数"解耦成两个独立追踪的量——`w`（计数）完全不变，
只服务 mass bias；新增 `imp`（重要性质量），只服务 pooling 的加权平均和 rank-1
统计量的 Chan 合并公式。默认关闭（`importance_pooling=False`）时两条路径完全不
接触新代码，逐字节复现改动前的行为。

- `litgpt/log_kv_cache.py`：
  - `_pair_rank1_stats()` 新增 `frac_a`/`frac_b` 参数（默认 0.5/0.5），把硬编码的
    `0.25` 泛化成 `frac_a*frac_b`——2 点集合的加权协方差有闭式解
    `Sigma = p_a*p_b*(k_a-k_b)(k_a-k_b)^T`，对任意 `p_a+p_b=1` 都精确成立（不是
    只在 0.5/0.5 时），推导见本节末尾；`p_a=p_b=0.5` 时退化成原公式，逐字节不变。
  - `compact()` 新增可选的 `imp1`/`imp2` 参数：给定时用重要性（不是 `w`）驱动
    pooling `alpha` 和 Chan 合并公式里的 `frac_a`/`frac_b`（协方差必须用跟 pooling
    相同的权重居中，否则统计量会不一致），返回值追加一个 `imp_total` 尾元素；
    省略时是原有 3-元组/8-元组返回，一行代码不多算。
  - `_compact_tokens()` 新增可选的 `imp` 参数（arbitrary-n 加权均值 + 加权
    rank-1 统计量，`sqrt(p_i)` 替换原来均匀的 `inv_sqrt_n`，`p_i=1/n` 时精确退化
    成原公式）。
  - `LogStructuredKVCache.__init__` 新增 `importance_pooling: bool = False`；每层
    新增 `level_imp_{ell}` buffer（形状同 `level_w_{ell}`，恒定 fp32——不参与任何
    跟激活值的 cat/matmul，只做比例计算，fp32 避免 bf16 精度模糊重要性比例）。
    `_get_level`/`_set_level`/`_clear_level`/`_add_compact_entry`/`_binary_carry`/
    `_flush_pairs`/`_append_level0`/`ingest_chunk` 都相应加了 `imp` 参数线程。
  - `_flush_pairs`（真实流式路径，唯一处理原始 token 的入口）里，重要性启发式是
    `imp_tok = k.float().norm(dim=-1)`（post-RoPE key 的 L2 范数）——不引入新的
    可学参数，纯 k 的函数，因此训练/推理路径自动一致。
- `litgpt/model.py`：`build_log_kv_cache`/`set_log_kv_cache`/`enable_log_kv_training`
  三处新增 `importance_pooling` 参数，一路透传到 `LogStructuredKVCache` 构造。
- `demo.py`/`eval.py`：新增 `log_kv_importance_pooling`（`bool`，两边都走
  `run_cli()` 的签名自省，`--log_kv_importance_pooling true` 自动生效，不需要
  手写 argparse），`demo.py` 的 `_run_eval()` 辅助函数也透传这个参数，保证
  `demo.py` 驱动的训练+eval 一条龙天然用同一个开关值。
- `eval.sh`/`majob.sh`：**没有改**——参照 `log_kv_dense_mode`（同类"还在验证阶段"
  的 bool 开关）的先例，它也没有进核心 `LOG_KV_ARGS` 列表，而是走
  `DIAG_ARGS="--log_kv_dense_mode true" bash eval.sh ...` 这种 opt-in 覆盖，本次
  新开关照此惯例处理，见 8.5 的调用命令。

**验证方式**：本次会话只有本地 Mac + CPU 环境，没有 GPU，无法跑真实 checkpoint。
已做的验证：①手推数学——2 点加权协方差闭式解、n 点加权协方差的 `sqrt(p_i)`
推导，并用单测核对（含"均匀重要性退化成原无权公式"的一致性检验，规避了"rank-1
截断对 n>2 不精确"这个陷阱——n>2 时不能拿真实协方差去核对 rank-1 truncation
的输出，只能核对"均匀 imp 退化成 imp=None 路径"这种自洽性）；②
`tests/test_log_kv_cache.py` 新增 `TestImportancePooling`（11 个用例：加权公式的
闭式解核对、`compact()`/`_compact_tokens()` 的 imp 驱动 alpha 而 w 不受影响、
`ingest_chunk`/`add_recent` 端到端流式路径含多级 carry），config `mineru`
conda env（`/Users/hourunli/anaconda3/envs/mineru`，Python 3.12）跑
`test_log_kv_cache.py` 全量 124/124 通过（113 条既有 + 11 条新增，逐条比对过
关闭改动前后 baseline 是 113 条不变）；③`git stash` 到干净版本重跑同一测试命令，
确认后 14 个 collection error（`litdata`/`jsonargparse` 缺失、pytest marker 未
注册）和这唯一的 1 条 warning（`requests`/`urllib3` 版本不匹配）都是环境本身
就有的、跟这次改动无关。**没有做的**：真实 checkpoint 上的 niah/LongBench 数字，
需要 GPU 环境接着跑。

**下一步（在有 GPU 的环境接着做）**：先跑 8.5 的纯 eval-time 决定性实验（不需要
重新训练，因为这是确定性启发式，直接套在已有的 warmup CPT checkpoint 上）——
如果 niah/LongBench 相对 vanilla（0.0827/0.1716/0.1918）有实质提升，说明这个方向
有戏，再考虑做一版可学权重（比如把 L2 范数换成一个小 MLP 门控）配合 CPT 训练；
如果没有提升，说明"稀释"不是（或不是主要）瓶颈，改评估 6.3 的按 width 差异化
scale 或阶段 3 提到的自适应槽宽/预算分配方向。

**数学推导备忘（2 点加权协方差精确闭式解）**：设 `m = p_a k_a + p_b k_b`
（`p_a+p_b=1`），则 `k_a - m = p_b(k_a-k_b)`、`k_b - m = -p_a(k_a-k_b)`，代入
`Sigma = p_a(k_a-m)(k_a-m)^T + p_b(k_b-m)(k_b-m)^T` 化简得
`Sigma = p_a p_b (k_a-k_b)(k_a-k_b)^T`——对任意 `p_a,p_b` 都精确 rank-1，不需要
`_rank1_psd_from_factors` 的迭代近似（只是复用同一个函数走统一代码路径，2 点输入
时该函数本身也精确收敛，见 `_dominant_eigvec_small` 的幂迭代对已经精确 rank-1 的
输入零误差这一事实——已在 6.7/6.8 NMS 那次会话验证过同一性质）。`p_a=p_b=0.5`
时代入得 `0.25·(k_a-k_b)(k_a-k_b)^T`，正是原来的无权公式。

## 7. 有用的坑 / 经验教训（给下次接续的自己看）

- `eval.sh` vs 直接 `torchrun --config <yaml> eval.py`：**语义不同**。`eval.sh` 会把
  YAML 展平成纯 CLI flag 自己拼命令，不传 `--config` 给 `eval.py`，所以 `_o()` 的
  "YAML 非 null 覆盖 CLI" 逻辑根本不会触发；直接用 `--config <yaml>` 时会触发。凡是
  新增会被扫参覆盖的字段，YAML 里必须写 `null`（见 3.3 节）。
- argparse "后出现的 flag 生效"：`utils.py` 的 `run_cli()` 用的是普通
  `argparse.ArgumentParser()`（不是 jsonargparse，是为了让 `--config` 能当一个普通
  参数存在），一行命令里如果同一个 flag 出现两次，以最后一次为准——这是 `majob.sh`/
  `eval.sh` 里 `DIAG_ARGS` 放在 `LOG_KV_ARGS` 后面、能正确覆盖 YAML 默认值的原因。
- `majob.sh` 如果 `save_path` 下已经存在 checkpoint，会**跳过整个训练阶段**直接进
  eval——这是当初新建 `arc_warmup.yaml` 必须用独立 `save_path` 的原因，不然会误判成
  "已经训练好了"，直接拿旧 checkpoint 去评测。**2026-08-12 修了一个相关 bug**：
  跳过判断原来只用 `checkpoint_exists`（裸文件存在性），不看 `checkpoint_meta.yaml`
  里的 `global_step`——`checkpoint_finished()` 函数其实早就写好了（正确读
  `global_step`/`max_steps` 比较），但从没被真正调用过。后果是想给一个已经训练过的
  `save_path` 调高 `max_steps` 续训更多步时，`majob.sh` 会误判"权重已存在"直接跳过，
  不会真的多训。已修：`majob.sh:299` 判断条件加上 `&& checkpoint_finished`。
- **`resume_dir` 和 `auto_resume` 不要选错**：想让训练从已有 checkpoint 精确接着跑
  （`global_step`/optimizer/LR schedule 全部延续），必须用 `auto_resume: true`，
  不能用 `resume_dir` 指向同一个目录——`resume_dir` 无论指向哪里都只做纯权重加载，
  `global_step` 永远从 0 开始，调高 `max_steps` 只会导致重新跑满新的总步数，而不是
  在原有基础上再训练"差额步数"。
- 修改本地代码前，遇到"看起来是 bug"的测试预期值，先去追代码的真实语义（本次是
  `_binary_carry`），不要想当然地"以测试为准"去改生产代码——3.2 节就是反过来，测试
  错了，代码是对的。
- 本地跑 pytest 用 conda env `mineru`（`/Users/hourunli/anaconda3/envs/mineru`,
  Python 3.12）；系统自带 Python 3.9.13 无法解析仓库里到处用的 `X | None` 类型注解。
- **`log_kv_dense_mode` 测出来的数字不是"干净的能力上限"**（见 6.10）：CPT 训练
  全程走 LogKV 分块压缩前向，从没真正用稠密注意力训练过，dense_mode 推理是把
  压缩训练出来的权重塞进一个训练时从未见过的输入分布里跑，本质上也是一种（跟
  pin 那个不是同一种、但同类型的）训推不一致。不影响"压缩代价远大于 pin 代价"
  这个定性结论（量级差太多），但引用这个数字时要说清楚它的含义，不要当成物理
  意义上精确的能力上限。
- **两份独立诊断要事后关联分析时，提前设计好共享 id，不要等跑完了再想办法对齐**
  （见 6.12）：`sample_id` 的设计要点——(a) 用能保证跨 rank/跨 task 唯一的组合
  （rank + 在完整共享请求列表里的原始下标 + task 名兜底），而不是局部计数器；
  (b) 同一个 id 字符串对象直接传给两边记录，不要各自独立生成再指望格式一致；
  (c) 需要跨样本聚合分析的诊断，从"只做全局 running stat"改成"额外按样本 id 分桶"
  时，每个样本只存几个标量（不存张量），内存量级天然可控，不用等真的爆内存了
  才发现问题。

## 8. 常用运行命令（2026-08-10 新增）

### 8.1 直接调用 Python 脚本（单机调试用）

`demo.py`（训练，logKV CPT，无 dense 分支）：

```bash
# 单卡调试
python demo.py --config exp/qwen0.6b-4k/debug.yaml

# 单机多卡
torchrun --nproc_per_node=8 demo.py --config exp/qwen1.7b-32k/arc_warmup.yaml
```

`eval.py`（评测）：

```bash
# 单卡，不用 YAML，纯 CLI（调试用）
python eval.py --checkpoint_dir ./checkpoints/Qwen/Qwen3-0.6B-Base --benchmark piqa

# 单机多卡，纯 CLI
torchrun --nproc_per_node=8 eval.py \
  --checkpoint_dir <SAVE_DIR> --benchmark "boolq,piqa,hellaswag" \
  --log_kv_B 512 --log_kv_recent_size 1024 --log_kv_pin_size 0

# 用 YAML（--config 会触发 3.3 节提到的"YAML 非 null 覆盖 CLI"逻辑，
# 新增字段默认值必须写 null，否则同名 CLI 参数会被静默吞掉）
python eval.py --config exp/qwen1.7b-32k/eval.yaml

# pin-vs-needle 只读诊断（3.6 / 6.1 节），单卡先跑小样本：
torchrun --nproc_per_node=1 eval.py --config exp/qwen1.7b-32k/eval.yaml \
  --benchmark niah_single_1 --limit 4 \
  --log_kv_pin_size 256 --log_kv_pin_diag_output <out-dir> \
  --log_kv_pin_diag_max_samples 4
```

### 8.2 `majob.sh`：训练→评测一条龙（ModelArts 作业入口）

```bash
bash majob.sh <config.yaml>
# 例：
bash majob.sh exp/qwen1.7b-32k/arc_warmup.yaml
```

行为：先看 `save_path` 下是否已有"finished"的 checkpoint（有就跳过训练直接评测，
见 7 节），否则跑 `torchrun ... demo.py --config <yaml>` 训练；训练完做文件锁
barrier 等所有节点就绪，再跑两次 `torchrun ... eval.py`（主 benchmark 列表一次，
niah_single_1/2/3 一次，两次共用同一个 `MASTER_PORT`——这点跟 8.3 的 `eval.sh` 不同）。

依赖的环境变量（ModelArts 会自动注入；本地单机手动跑时全部有默认值，不用手动设置）：

| 变量 | 作用 | 默认值 |
|---|---|---|
| `MA_NUM_GPUS` | 每节点卡数 → `GPUS_PER_NODE` | 8 |
| `MA_NUM_HOSTS` | 节点数 → `NUM_NODES` | 1 |
| `MASTER_ADDR` | rendezvous 地址 | `localhost`（多机必须显式可达，或靠 `MA_VJ_NAME`/`MA_TASK_NAME`/`MA_MASTER_INDEX` 拼出 ModelArts DNS 名） |
| `MASTER_PORT` | rendezvous 端口 | 6000 |
| `VC_TASK_INDEX` | 当前节点序号 → `NODE_RANK` | 0（多机每个节点必须不同，通常由调度器注入，不要手动瞎设） |

脚本开头硬编码 `source /home/ma-user/anaconda3/bin/activate torch218`——本地非
ModelArts 环境跑之前先确认这个 conda env 路径存在，或者手动改成本机的 conda 路径。

### 8.3 `eval.sh`：跳过训练，只跑评测（已有 checkpoint 时调参/重跑用）

```bash
bash eval.sh <training_yaml_or_eval_yaml> [benchmark_csv|none] [niah_csv|none]
# 例：
bash eval.sh exp/qwen1.7b-32k/arc.yaml                  # 默认全量 benchmark + 默认三项 niah
bash eval.sh exp/qwen1.7b-32k/arc.yaml piqa none        # 只跑 piqa，不跑 niah
DIAG_ARGS="--log_kv_diag_mode baseline --log_kv_diag_exact_from_layer 21" \
    bash eval.sh exp/qwen1.7b-32k/diag.yaml niah_single_1 none   # 透传诊断参数
```

跟 `majob.sh` 共用同一套环境变量约定（见 8.2 表格），但更严格：会校验
`GPUS_PER_NODE`/`NUM_NODES`/`NODE_RANK`/端口必须是非负整数，多机时 `MASTER_ADDR`
仍是 `localhost` 会直接报错退出（`majob.sh` 没有这个校验）。另外主 benchmark 和
niah 两次 `torchrun` **各用各的端口**（`MAIN_EVAL_MASTER_PORT` 默认 `MASTER_PORT`，
`NIAH_EVAL_MASTER_PORT` 默认 `MASTER_PORT + 1`），不会像 `majob.sh` 那样两次评测
复用同一个端口。`eval.sh` 把 YAML 展平成纯 CLI flag 传给 `eval.py`（不传
`--config`），所以 3.3 节"YAML 非 null 覆盖 CLI"的坑在这条路径上不会触发。

### 8.4 dense-mode base-vs-CPT 对比 / pin_score_diag + pin_diag 联合诊断（2026-08-12 新增）

**dense-mode 对比**（6.10）：base checkpoint 用新建的 `dense_niah_base.yaml`，CPT
checkpoint 复用 `arc_warmup.yaml` + `DIAG_ARGS` 覆盖 `log_kv_pin_size`：

```bash
bash eval.sh exp/qwen1.7b-32k/dense_niah_base.yaml none niah_single_1,niah_single_2,niah_single_3

DIAG_ARGS="--log_kv_dense_mode true --log_kv_pin_size 0" \
    bash eval.sh exp/qwen1.7b-32k/arc_warmup.yaml none niah_single_1,niah_single_2,niah_single_3
```

**pin_score_diag + pin_diag 联合跑**（6.11/6.12，两个诊断没有互斥关系，可以一次
跑出来，为了 hit/miss 分组对比样本量足够，不要加 `--limit`，也不要设
`log_kv_pin_diag_max_samples`，否则 `pin_diag` 那边会截断、join 后一部分
`pin_score_diag` 样本找不到对应项）：

```bash
SAVE_DIR=/home/ma-user/work/bucket-wulan-green/${MY_REAL_NAME:-default}/ckpt-new/qwen1.7b-32k-cpt-logKV-warmup

DIAG_ARGS="--log_kv_pin_score_diag_output ${SAVE_DIR}/pin_score_diag --log_kv_pin_score_diag_window_from_end 1024 --log_kv_pin_diag_output ${SAVE_DIR}/pin_diag" \
    bash eval.sh exp/qwen1.7b-32k/arc_warmup.yaml none niah_single_1,niah_single_2,niah_single_3
```

跑完之后用 `unused/pin_collapse_vs_hit.py`（纯标准库，训练环境不用装额外的包）
按 `sample_id` join 两份 JSON，输出 hit/miss 分组统计、按 task 拆分、Pearson
相关系数、塌缩最猛的 top-N 样本：

```bash
python unused/pin_collapse_vs_hit.py \
    --pin-score-diag pin_score_diag_..._TS.json \
    --pin-diag pin_diag_..._TS.json \
    --out-csv joined.csv
```

### 8.5 重要性加权池化决定性实验（2026-08-13 新增，见 6.14）

纯 eval-time 开关，不需要重新训练（确定性启发式，`--config` 后面覆盖同一份
`arc_warmup.yaml` 加载的 checkpoint 权重不变）。跟 `log_kv_dense_mode` 一样走
`DIAG_ARGS` opt-in，没有进 `eval.sh`/`majob.sh` 的核心参数列表：

```bash
DIAG_ARGS="--log_kv_importance_pooling true" \
    bash eval.sh exp/qwen1.7b-32k/arc_warmup.yaml none niah_single_1,niah_single_2,niah_single_3
```

对比对象是 0 节表格/6.1 表格里的"+2nd order, no pins"一行（同一 checkpoint，唯一
区别是加不加这个开关）：

| 配置 | LongBench | LongBench_e | niah |
|---|---:|---:|---:|
| vanilla（无 pin，均匀池化，现有基线）| 0.1716 | 0.1918 | 0.0827 |
| + importance_pooling（本次待验证）| ? | ? | ? |

三项都明显优于 0.1716/0.1918/0.0827 → 方向成立，值得做可学权重版本；没有实质提升
→ 说明稀释不是主要瓶颈，回到 6.3/阶段 3 提到的槽宽或自适应预算方向。
