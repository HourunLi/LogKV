# LogKV 项目工作记录（存档，更新于 2026-08-11）

> 本文件是给下次接续工作时用的存档，记录 LogKV（Fenwick-tree / O(log N) 显存 KV cache
> 压缩，rank-1 Σ_s/Γ_s 二阶修正）这条线目前做了什么、改了什么、卡在哪、下一步该干嘛。
> 代码层面的细节（怎么加插件、怎么跑服务）见仓库根目录上一级的 `~/AGENTS.md`；这份
> 只讲 LogKV 这一个专题。

## 0. 现状速览（2026-08-11 更新，只想快速接续就读这节，细节看后面对应章节）

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

**下一步方向（2026-08-11 更新——阶段 2 已出结果，是负面的，路线图据此重排，见 6.9）**：

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

*阶段 3 —— 稠密检索能力是否被侵蚀（见 6.5），优先级下调为可选补充证据：*
(f)(g) 6.5 的假设仍未验证，`log_kv_dense_mode` 开关已经实现并通过代码审查，但
阶段 2 的证据已经不需要靠它来解释下游变差——即使 salience 打分质量完美，只要
"精确槽混入池化槽"本身是分布外输入，一样会拖累指标。降级为可选的补充证据，不再是
主线；开关已就绪，真要跑的话成本很低（缺的只是真跑一次 base vs CPT checkpoint 的
NIAH 对比）。

*阶段 4（当前优先级最高）—— 训推一致：让模型在训练里见过"精确槽混入池化层级"这件事：*
(h) 不建议直接照搬最初设想的"重新设计训练前向 + 重新跑完整 CPT"（成本最高、风险
最大）。建议先做一个**最小可行版本**：从已有的 warmup CPT checkpoint 出发做一次
**短续训**（几百步量级，不是从头 1500 步），训练时用简单规则（比如随机挑
`_log_kv_train_lowmem_forward` 已经流过的一部分位置，复制一份精确副本混进当前
chunk 的槽序列）模拟"精确槽 + 池化槽共存"这种输入结构，不追求复刻真实
`_log_kv_select_pins` 的显著性打分逻辑（那是 no_grad、推理专用，训练侧模拟只需要
让模型见过这种**分布**，不需要位置选得多准）。跑完之后用同一套 NMS pin 配置重新
评测，看 pin=256 能不能追上甚至超过 vanilla。如果这个便宜的短续训验证有效，再考虑
要不要并入下一次完整 CPT；如果依然没用，说明问题比"分布没见过"更深，需要重新评估
是否彻底放弃 pin。

*继续排在 pin 这条线之后、暂不动的：*
(i) 6.3 提到的"按 layer/width 差异化 second_order_scale"接口改动。

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

## 7. 有用的坑 / 经验教训（给下次接续的自己看）

- `eval.sh` vs 直接 `torchrun --config <yaml> eval.py`：**语义不同**。`eval.sh` 会把
  YAML 展平成纯 CLI flag 自己拼命令，不传 `--config` 给 `eval.py`，所以 `_o()` 的
  "YAML 非 null 覆盖 CLI" 逻辑根本不会触发；直接用 `--config <yaml>` 时会触发。凡是
  新增会被扫参覆盖的字段，YAML 里必须写 `null`（见 3.3 节）。
- argparse "后出现的 flag 生效"：`utils.py` 的 `run_cli()` 用的是普通
  `argparse.ArgumentParser()`（不是 jsonargparse，是为了让 `--config` 能当一个普通
  参数存在），一行命令里如果同一个 flag 出现两次，以最后一次为准——这是 `majob.sh`/
  `eval.sh` 里 `DIAG_ARGS` 放在 `LOG_KV_ARGS` 后面、能正确覆盖 YAML 默认值的原因。
- `majob.sh` 如果 `save_path` 下已经存在一个"finished"的 checkpoint，会**跳过整个
  训练阶段**直接进 eval——这是当初新建 `arc_warmup.yaml` 必须用独立 `save_path` 的
  原因，不然会误判成"已经训练好了"，直接拿旧 checkpoint 去评测。
- 修改本地代码前，遇到"看起来是 bug"的测试预期值，先去追代码的真实语义（本次是
  `_binary_carry`），不要想当然地"以测试为准"去改生产代码——3.2 节就是反过来，测试
  错了，代码是对的。
- 本地跑 pytest 用 conda env `mineru`（`/Users/hourunli/anaconda3/envs/mineru`,
  Python 3.12）；系统自带 Python 3.9.13 无法解析仓库里到处用的 `X | None` 类型注解。

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
