# LogKV 项目工作记录（存档，更新于 2026-08-13）

> 本文件是给下次接续工作时用的存档，记录 LogKV（Fenwick-tree / O(log N) 显存 KV cache
> 压缩，rank-1 Σ_s/Γ_s 二阶修正）这条线目前做了什么、改了什么、卡在哪、下一步该干嘛。
> 代码层面的细节（怎么加插件、怎么跑服务）见仓库根目录上一级的 `~/CLAUDE.md`；这份
> 只讲 LogKV 这一个专题。
>
> **2026-08-13 note**：pin 相关的详细调试记录（原 6.4~6.13，约 500 行）已压缩进 6.4，
> 只保留结论链和关键数字，过程性叙事已删除。完整历史如果需要可以从更早的会话记录找。

## 0. 现状速览（2026-08-13 更新，只想快速接续就读这节，细节看后面对应章节）

**结论先说：pin 这条线已终止**。选择质量、剂量效应、分数尺度机制、训推一致四个
独立方向依次验证均为负（详见 6.4），不再往 pin 上投入。下一步的高价值方向是压缩
本身——阶段 3 揭示压缩本身（不涉及任何 pin）就吃掉 85+ 个百分点，比 pin 全系列
实验的影响大一个数量级。

**2026-08-13 新增、2026-08-14 跑出决定性实验结果：压缩本身这条线的第一个方向
——重要性加权池化——代码已实现（见 6.5），eval-time 决定性实验已跑出分裂结果
（ACC/LongBench/LongBench_e 小幅变好，niah 反而从 0.0827 掉到 0.0787），详细
数字和解读见 6.5/8.5。核心思路：把 `_flush_pairs`/`compact()` 里"槽的 pooling
权重"和"log(w) mass bias 用的 token 计数"解耦成两个独立量，pooling 权重按启发式
重要性（post-RoPE key L2 范数）加权而不是均匀 1/n，mass bias 继续完全不变地用
计数 `w`。是 k 的纯函数，无新增可学参数，训练/推理路径自动一致（这正是 pin 系列
失败的根因之一，这次设计上从一开始就规避掉）。默认关闭时（`importance_pooling=
False`）跟改动前逐字节相同，124/124 单测通过（113 条既有 + 11 条新增）。**结论：
不建议直接采用这版启发式**——niah 定向变差，说明"key L2 范数"这个显著性代理和
"needle-ness"不是一回事（很可能是 attention-sink 式高范数干扰 token 把 needle
的池化权重从均匀池化保证的 1/n 挤压下去了），下一步候选见 6.5 结尾。**

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
| + 2nd order, no pins, + importance pooling | 0.6157 | 0.1753 | 0.196 | 0.0787 |

**pin 调查结论摘要（完整证据链见 6.4）**：二阶修正本身是正贡献（niah 0.032→0.0827，
2.6×），但大部分下游损失在纯均值池化阶段就已经发生，rank-1 表达能力本身不够。
pin 本来是为了顶住这个失败模式设计的（往压缩层级里混入精确槽兜住被稀释的
needle），但选择质量修好之后（NMS 空间分散约束，near-hit 从不如随机修到 1.9×
随机）下游指标依然全面跑输 vanilla，且是单调剂量效应（插得越多伤害越大），根因
是"分数尺度失配"（pooled 槽点积被磨平、pin 槽是原始精确 key，混进同一 softmax 行
扰乱整行注意力分配），训练期注入也没能让模型学会校准这个尺度差异。四个独立方向
依次验证均为负，不是同一个 bug 的不同侧面，pin 这条线到此终止。

**下一步方向**：

*压缩本身·方向 1【已实现代码、已跑决定性实验、两个正交旋钮（lambda +
temperature）代码均已完成，待 GPU 侧扫描，见 6.5/8.5】—— 重要性加权池化：*
lambda=1.0（纯重要性）的结果是 ACC/LongBench/LongBench_e 小幅变好、niah 定向
变差（0.0827→0.0787），推翻了"稀释是 niah 主瓶颈、加权池化能救回来"这个核心
机制假设的直接验证——key L2 范数是个还行的通用显著性代理，但和"needle-ness"
不是一回事。**不建议直接采用 lambda=1.0**。已实现两个正交旋钮：
`importance_pooling_lambda`（往均匀份额整体混合）和
`importance_pooling_temperature`（只压缩启发式自身的动态范围，更针对性地
压制离群高范数 token），代码/单测均已完成，可以在 GPU 上并行跑两条扫描（命令见
8.5）：如果存在中间点同时保住 LongBench 增益、niah 不再倒退，这个方向值得继续
投入；如果两条曲线都只是端点间插值、没有任何中间点两头都好，说明 needle 保护
和这类从 k 范数出发的确定性启发式结构性冲突，转 6.2 方向。

*继续排在后面、暂不动的（都没有新代码，需要先讨论范围再决定值不值得写）：*
(i) 6.2/6.3 提到的"按 layer/width 差异化 second_order_scale"生产接口改动——**唯一
例外**：诊断门控本身（3.1 节，`log_kv_diag_second_order_max_width`/
`_max_layer`，已经在 `eval.py` 里接好、从未接入生产路径）零代码就能跑，只是
测的是 oracle 输出误差而不是真实 LongBench/niah 分数，命令见 8.5，可以作为
"值不值得做这个接口改动"的免费前置信号，跟方向 1 的两条扫描一起并行提交。
(ii) 压缩本身·方向 2（稠密→压缩自蒸馏，让压缩前向对齐同序列稠密前向）：天花板更
高但需要改训练循环（多跑一遍稠密 teacher 前向 + KD loss），工程量和出 bug 的
风险都明显更大，且这次没有 GPU 能自己先跑一轮验证正确性，不建议在没有更多信号
前贸然写——想推进的话应该先讨论清楚 loss 形式和触发时机，再动手实现。
(iii) 压缩本身·方向 3（rank-2/rank-r 槽统计）：跟方向 1 是替代关系（加权池化让
均值 key 已经带上 needle 内容后，需要的残差 rank 天然更低），需要把 Chan-merge
和 rank-1 截断的数学（`_rank1_psd_from_factors` 等）推广到 rank-r，改动面大、
正确性依赖需要在 GPU 上迭代验证，原因同 (ii)，不建议这次盲写。

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

### 3.6 pin-vs-needle 只读诊断工具（历史实现，pin 已终止，工具仍可复用）

为排查 pin 拖累指标的根因，实现了只读诊断：`litgpt/litgpt/log_kv_pin_diag.py`（定位
NIAH prompt 里的 needle token span，汇总每层/每组 pin indices 是否 exact/near hit）、
`eval.py` 的 `--log_kv_pin_diag_output` 等参数（真实 `generate_until` 路径 prefill 后
记录）、`unused/diagnose_log_kv_pins_jsonl.py`（离线入口，复用同一套逻辑跑已 dump 好的
prompt）。诊断不改变模型输出、不参与训练。pin 这条线虽然终止了，但这套"某种槽选择/
标注是否命中已知目标位置"的诊断基建具备通用性，以后如果要诊断别的选择类机制（比如
自适应槽宽的预算分配）可以直接复用或参考这个模式。

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

### 6.1【已解决】"width≥8 + scale 匹配训练值(0.2) + 生产 B=512"的干净数据

warmup CPT（scale=0.2, warmup=100 步，1500 步训练已完整跑完）产出了完整的下游指标
对比，走的是 `majob.sh`，四组配置用的是**同一个 checkpoint + 同一套评测命令模板**，
只切换 `log_kv_pin_size` / `log_kv_second_order_scale`，可比性已确认（`majob.sh` 调
`eval.py` 时不传 `--config`/`--limit`，全量跑，niah 数字取的是 32768 档、
niah_single_1/2/3 三个子任务的均值，不存在浅层污染或截断问题）：

| 配置 | ACC | LongBench | LongBench_e | niah（均值）| niah_single_1 | niah_single_2 | niah_single_3 |
|---|---|---|---|---|---|---|---|
| 完整 transformer（dense baseline，CPT 后）| 0.6158 | 0.2463 | 0.2715 | 1.0 | – | – | – |
| LogKV vanilla（无二阶修正、无 pin）| 0.6146 | 0.1626 | 0.1803 | 0.032 | 0.024 | 0.046 | 0.026 |
| LogKV + importance pin | 0.6146 | 0.1363 | 0.1441 | 0.0313 | 0.044 | 0.03 | 0.02 |
| LogKV + 2nd order + pins | 0.6113 | 0.1214 | 0.1331 | 0.0467 | 0.068 | 0.032 | 0.04 |
| LogKV + 2nd order, no pins | 0.6113 | 0.1716 | 0.1918 | 0.0827 | 0.07 | 0.114 | 0.064 |

**解读**：
- 二阶修正（scale=0.2，与训练目标匹配）在**不加 pin** 时是四个压缩变体里下游指标
  最好的：LongBench 0.1716 > vanilla 0.1626，niah 0.0827 > vanilla 0.032（约 2.6×）。
  说明 warmup CPT 训练出来的二阶修正在下游任务上确实有正向作用。
- **加 pin 是负向的**，且和是否叠加二阶修正无关。根因见 6.4 的 pin 调查总结。
- ACC（常识推理）四组几乎不分伯仲（0.611–0.616），短程信息在所有压缩方案下都保留
  得不错，真正拉开差距的是长程检索类任务（LongBench、niah）。
- LogKV vanilla（纯均值池化）在 32768 长度上 niah 已经是 0.032，相对 dense baseline
  的 1.0 几乎全灭——**绝大部分损失在纯均值池化阶段就已经发生**，这是 6.5 重要性
  加权池化想直接对症的病灶。

### 6.2 是否要做按 layer/width 差异化的 `second_order_scale`（暂不动）

6.1 的数据证明 width≥8（32768 长度）在匹配 scale 下确实有明显下游损害，但大部分
损害发生在纯均值池化阶段，二阶修正只是部分缓解。判断：**先看 6.5（重要性加权池化）
出不出结果，再回头评估这个接口改动的 ROI**——如果加权池化能显著缓解 width≥8 的
损害，那么 per-width scale 这个接口改动的必要性会显著降低。3.1 节的诊断门控已经
验证过"只在 width≤N 生效"在数值上是可行的，接口改动主要是把这个门控从诊断专用
搬到生产路径，并想清楚训练时怎么处理这个新维度的超参。**不要在没有更多信号之前
动生产接口**。

### 6.3 dense-mode 对比：压缩本身的信息损失是最大瓶颈（2026-08-12）

`eval.py`/`litgpt/model.py` 的 `log_kv_dense_mode` 开关（为真时走标准 `set_kv_cache`，
完全绕开 LogKV 代码）跑了 base（未 CPT）vs warmup CPT 的稠密 NIAH 对比：

| 指标 | base（未 CPT）| warmup CPT | 变化 |
|---|---:|---:|---|
| Common sense | 0.6151 | 0.6123 | 微降 ~0.3pp |
| LongBench | 0.013 | 0.2075 | 约 16 倍 |
| LongBench_e | 0.2606 | 0.4463 | +71% |
| niah（稠密）| 0 | 0.9353 | 从完全不会到接近满分 |

**解读**：CPT 训练确实教会了模型长上下文检索能力（稠密 niah 0→0.9353）。把这个
0.9353 稠密上限跟 6.1 表格的"LogKV + 2nd order, no pins"（niah 0.0827）对比——
同一个 checkpoint，唯一变量是开不开压缩，niah 从 0.9353 掉到 0.0827——**压缩本身
（完全不涉及 pin）就吃掉了 85 个百分点以上**，比整个 pin 系列实验的影响大一个
数量级。这把"压缩本身的信息损失"确立为全链路里最大的单一瓶颈，pin 一直以来只是
在争夺一个本身就很小的剩余空间。**重要限定**：CPT 训练全程走分块压缩前向，从没
真正用稠密注意力训练过，所以 0.9353 不是"干净的能力上限"，而是"压缩训练出来的
权重泛化到陌生输入分布"的结果——如果有偏，大概率让这个数字偏保守（真实上限可能
更高），不影响"压缩代价远大于 pin 代价"这个结论。

### 6.4 pin 系列调查存档（已终止，2026-08-10~13）

pin（SnapKV 式显著性钉扎：prefill 时给全前缀打分，top-P token 以精确 w=1 槽形态
钉在压缩层级之外）本来是为了顶住 6.1/6.3 揭示的均值池化稀释问题设计的。完整调查
分四个独立方向，依次验证均为负，详细数据留档如下：

**阶段 1（选择质量）**：诊断发现 `_log_kv_select_pins` 的原始 `topk` 选择缺乏空间
分散性，256 个 pin 名额会抱团挤在少数"泛泛显著"的热点里（离线工具 `unused/pin_diag.py`
测得典型 group 的 near-hit 率只有 17.4%，比同参数下的纯随机基线 35.0% 还差）。修复
方案：给 `_log_kv_select_pins` 加 NMS 空间分散约束（`log_kv_pin_min_distance` 参数，
`litgpt/model.py`），贪心选点时抑制已选点周围的候选，验证后 near-hit 率从"不如随机"
修到 65.1%（1.9× 随机基线）。**选择质量修复真实有效**，但见阶段 2。

**阶段 2（选择修好后是否真解决下游问题）**：用修好的 NMS 配置重新跑真实
LongBench/niah，结果全系列 pin 变体（含极小 pin_size=4 对照组）都跑输不用 pin
的 vanilla：

| 配置 | pin_size | min_distance | LongBench | LongBench_e | niah（均值）|
|---|---:|---:|---:|---:|---:|
| 不用 pin | 0 | – | **0.1716** | **0.1918** | **0.0827** |
| top-k（无 NMS），pin=4 | 4 | 0 | 0.1507 | 0.1624 | 0.0727 |
| top-k（无 NMS），pin=256 | 256 | 0 | 0.1214 | 0.1331 | 0.0467 |
| NMS，pin=256 | 256 | 16 | 0.1327 | 0.1457 | 0.0113 |
| NMS，pin=256 | 256 | 64 | 0.1313 | 0.1429 | 0.0200 |

三个 benchmark 排名一致：不用 pin 最好，pin=4 次之且离第一名最近，pin=256 各变体
（无论 NMS 怎么调）都更差——**确认是剂量效应**（插入的精确槽越多伤害越大），
不是选择精度问题：选点选得再准，pin=256 也没有更接近 vanilla。

**阶段 3（分数尺度机制）**：新增独立旁路诊断 `log_kv_pin_score_diag.py`，直接在
生产 `log_kv_slot_attention` 里记录 pin 槽 vs pooled 槽的原始点积、加完 mass bias
后的最终分数、各自吃到的 softmax mass。核心发现（120 个真实样本，全部 28 层）：
纯点积 `dot_pin_score` 全部层都比 `dot_pooled_score` 高（证实两种槽分数尺度确实
不同），且存在剧烈的偶发"塌缩"现象（某些 query 上 pin 吃掉远超比例的 mass，120
个样本全部都有，最猛时超过 500 倍）。进一步用 `sample_id` 关联 pin 命中率数据
（1500 个真实样本）发现：塌缩程度跟这个 pin 是否真的命中 needle，相关系数只有
±0.09~0.17，基本不相关——**推翻"塌缩=模型精准命中后合理押注"的乐观猜测**，支持
"分数尺度失配"机制假设：塌缩更像是某些位置的原始点积天生偏高（跟语义相关性
无关），不是选点准不准的问题。

**阶段 4（训推一致，最后一个假设）**：怀疑问题根源是训练路径完全没有 pin
（`_log_kv_select_pins` 是 `@torch.no_grad()`，只在推理期跑），模型没在训练里
见过"精确槽混入池化层级"这种输入分布。实现了训练期随机 pin 注入
（`_sample_training_pin_positions`/`_install_training_pins`，`litgpt/log_kv_cache.py`，
配合 `pin_train_prob`/`pin_train_warmup_steps` 线性爬坡），从 warmup CPT checkpoint
短续训到 step 1700 后评测：

| 配置 | common sense | LongBench | LongBench_e | niah |
|---|---:|---:|---:|---:|
| vanilla（无 pin）| 0.6113 | **0.1716** | **0.1918** | **0.0827** |
| pin=256, NMS-64，未训练 | – | 0.1313 | 0.1429 | 0.0200 |
| pin=256, NMS-64，训练期注入后 | 0.6126 | 0.1328 | 0.1484 | 0.0213 |

训练后跟训练前几乎完全一致，差异在噪声量级——**训推一致假设也不成立**。

**最终结论**：选择质量（真实有效但不解决下游问题）→ 剂量效应（插得越多伤害越大，
插得再少也是负收益）→ 分数尺度机制（塌缩确认存在但跟命中质量无关）→ 训推一致
（训练后跟不训练一样烂）。四个独立、互不依赖的方向依次验证都是负结果，不是同一个
假设的不同侧面被同一个 bug 污染——**pin 这条线到此终止，不建议继续投入**（含"加长
续训"这个选项：这次续训剂量偏轻，理论上不能 100% 排除欠训练，但四个方向一致指向
同一结论，且压缩本身的损失比 pin 影响大一个数量级，继续投入验证成本换回的信息
价值不高）。诊断基建（`log_kv_pin_diag.py`/`pin_diag.py`/`log_kv_pin_score_diag.py`/
`pin_collapse_vs_hit.py`）仍在代码库里，具备通用性，需要时可以复用或参考。

### 6.5 压缩本身·方向 1：重要性加权池化——已实现、已跑决定性实验，分裂结果（2026-08-13 代码 / 2026-08-14 结果，见本节末尾和 8.5）

**动机**：6.3 证实压缩本身（不涉及 pin）吃掉 85+ 个百分点，比 pin 全系列实验大
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

**为什么不是 pin 那条老路**：pin 是往压缩层级里混入一个量级迥异的精确 key（6.4
证实的"分数尺度失配"是 pin 失败的核心机制），而且训练路径完全没有 pin 存在
（`_log_kv_select_pins` 是 `@torch.no_grad()`，只在推理期跑）——训推不一致是 pin
四个死因之一。重要性加权池化不引入新槽、不改变槽的数量或量级，只改变"槽的内容
怎么算出来"，而且这个加权是 k 的**确定性纯函数**（无新增可学参数），训练和推理
天然用同一份代码算出同一个值，从设计上就规避了 pin 系列的训推不一致问题。

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

**结果（2026-08-14）**：8.5 的决定性实验已跑出，是分裂结果而不是预想的"整体
提升"或"整体无提升"——ACC/LongBench/LongBench_e 相对 vanilla（0.6113/0.1716/
0.1918）小幅提升到 0.6157/0.1753/0.196，但 niah 相对 vanilla（0.0827）反而降到
0.0787。这否定了"稀释是 niah 的主瓶颈、加权池化直接能救"这个假设的最简版本
（如果对，niah 应该最先受益，结果却是唯一变差的指标），但也不是"稀释完全不是
瓶颈"的干净否定（毕竟其它三个指标确实都在变好，说明这个杠杆点本身不是无效的，
只是当前这版启发式——key L2 范数——对 needle 类 token 是负相关而非正相关代理）。
下一步不是直接冲可学权重（MLP 门控）：2026-08-14 决定跳过"诊断 needle 范数分布"
这一步，直接实现了 `importance_pooling_lambda` 混合权重（部分走加权、部分走均匀，
代码/单测均已完成，见本节末尾），下一步是在 GPU 上扫一遍 lambda（命令见 8.5）——
如果混合权重能找到 sweet spot（LongBench 增益还在、niah 不再倒退），说明问题
出在"过度偏离均匀"而不是"方向整体错了"，才值得投入可学权重；如果混合权重也
救不回 niah，说明 needle 保护和这类显著性代理天然冲突，应该转 6.2 的按 width
差异化 scale 或自适应槽宽/预算分配方向。

**数学推导备忘（2 点加权协方差精确闭式解）**：设 `m = p_a k_a + p_b k_b`
（`p_a+p_b=1`），则 `k_a - m = p_b(k_a-k_b)`、`k_b - m = -p_a(k_a-k_b)`，代入
`Sigma = p_a(k_a-m)(k_a-m)^T + p_b(k_b-m)(k_b-m)^T` 化简得
`Sigma = p_a p_b (k_a-k_b)(k_a-k_b)^T`——对任意 `p_a,p_b` 都精确 rank-1，不需要
`_rank1_psd_from_factors` 的迭代近似（只是复用同一个函数走统一代码路径，2 点输入
时该函数本身也精确收敛，见 `_dominant_eigvec_small` 的幂迭代对已经精确 rank-1 的
输入零误差这一事实）。`p_a=p_b=0.5` 时代入得 `0.25·(k_a-k_b)(k_a-k_b)^T`，正是
原来的无权公式。

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
  "已经训练好了"，直接拿旧 checkpoint 去评测。**相关 bug 修复**：跳过判断原来只用
  `checkpoint_exists`（裸文件存在性），不看 `checkpoint_meta.yaml` 里的
  `global_step`——`checkpoint_finished()` 函数其实早就写好了（正确读
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
  `source activate mineru` 有时会解析到另一个装了同名 env 的 anaconda（Python
  3.9），不确定时直接用绝对路径 `/Users/hourunli/anaconda3/envs/mineru/bin/python`
  最保险。
- **`log_kv_dense_mode` 测出来的数字不是"干净的能力上限"**（见 6.3）：CPT 训练
  全程走 LogKV 分块压缩前向，从没真正用稠密注意力训练过，dense_mode 推理是把
  压缩训练出来的权重塞进一个训练时从未见过的输入分布里跑，本质上也是一种训推
  不一致。不影响"压缩代价远大于 pin 代价"这个定性结论（量级差太多），但引用这个
  数字时要说清楚它的含义，不要当成物理意义上精确的能力上限。
- **两份独立诊断要事后关联分析时，提前设计好共享 id，不要等跑完了再想办法对齐**：
  `sample_id` 的设计要点——(a) 用能保证跨 rank/跨 task 唯一的组合（rank + 在完整
  共享请求列表里的原始下标 + task 名兜底），而不是局部计数器；(b) 同一个 id 字符串
  对象直接传给两边记录，不要各自独立生成再指望格式一致；(c) 需要跨样本聚合分析的
  诊断，从"只做全局 running stat"改成"额外按样本 id 分桶"时，每个样本只存几个
  标量（不存张量），内存量级天然可控，不用等真的爆内存了才发现问题。

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
DIAG_ARGS="--log_kv_dense_mode true --log_kv_pin_size 0" \
    bash eval.sh exp/qwen1.7b-32k/arc_warmup.yaml none niah_single_1,niah_single_2,niah_single_3
```

跟 `majob.sh` 共用同一套环境变量约定（见 8.2 表格），但更严格：会校验
`GPUS_PER_NODE`/`NUM_NODES`/`NODE_RANK`/端口必须是非负整数，多机时 `MASTER_ADDR`
仍是 `localhost` 会直接报错退出（`majob.sh` 没有这个校验）。另外主 benchmark 和
niah 两次 `torchrun` **各用各的端口**（`MAIN_EVAL_MASTER_PORT` 默认 `MASTER_PORT`，
`NIAH_EVAL_MASTER_PORT` 默认 `MASTER_PORT + 1`），不会像 `majob.sh` 那样两次评测
复用同一个端口。`eval.sh` 把 YAML 展平成纯 CLI flag 传给 `eval.py`（不传
`--config`），所以 3.3 节"YAML 非 null 覆盖 CLI"的坑在这条路径上不会触发。
`DIAG_ARGS` 是给还在验证阶段、没进核心参数列表的 bool 开关用的 opt-in 覆盖机制
（`log_kv_dense_mode`/`log_kv_importance_pooling` 都走这条路，见上面示例和 8.5）。

### 8.4 dense-mode base-vs-CPT 对比（2026-08-12，见 6.3）

```bash
bash eval.sh exp/qwen1.7b-32k/dense_niah_base.yaml none niah_single_1,niah_single_2,niah_single_3

DIAG_ARGS="--log_kv_dense_mode true --log_kv_pin_size 0" \
    bash eval.sh exp/qwen1.7b-32k/arc_warmup.yaml none niah_single_1,niah_single_2,niah_single_3
```

（pin_score_diag/pin_diag 联合诊断的命令已随 6.4 的 pin 调查存档一起精简掉；工具
本身——`log_kv_pin_score_diag.py`/`log_kv_pin_diag.py`/`unused/pin_collapse_vs_hit.py`
——还在代码库里，需要时可以照着 `eval.py --log_kv_pin_score_diag_output`/
`--log_kv_pin_diag_output` 这两个参数的docstring 重新拼命令。）

### 8.5 重要性加权池化决定性实验（2026-08-13 新增，见 6.5）

纯 eval-time 开关，不需要重新训练（确定性启发式，`--config` 后面覆盖同一份
`arc_warmup.yaml` 加载的 checkpoint 权重不变）。跟 `log_kv_dense_mode` 一样走
`DIAG_ARGS` opt-in，没有进 `eval.sh`/`majob.sh` 的核心参数列表：

**2026-08-13 修复**：`arc_warmup.yaml`/`diag_warmup.yaml` 原来把 `log_kv_pin_size`
写死成 `256`（历史遗留，pin 调查期间需要）。pin 已确认终止（见 6.4），但这两份
yaml 还是 8.5/6.5 决定性实验和其它默认 eval 用的主力配置——不改的话，下面这条
命令会悄悄带着 pin=256 跑，跟"不用 pin"的 vanilla 基线（0.1716/0.1918/0.0827，
本身是当年专门加 `--log_kv_pin_size 0` override 才测出来的）不可比，属于一个真实
的踩坑点。已把两份 yaml 的默认值改成 `log_kv_pin_size: 0`（训练不受影响，只影响
这两份 yaml 触发的 eval 步骤），下面的命令现在不需要额外覆盖 pin 就是对的：

```bash
DIAG_ARGS="--log_kv_importance_pooling true" bash majob.sh exp/qwen1.7b-32k/arc_warmup.yaml
```

**注意（曾经踩过）**：早前这里写的是
`bash eval.sh ... none niah_single_1,niah_single_2,niah_single_3`——把主 benchmark
参数设成 `none`。`majob.sh` 的 `DEFAULT_BENCHMARKS`（主列表）和 `NIAH_BENCHMARKS`
（niah_single_1/2/3）是两个独立变量，**LongBench/LongBench_e 的数字来自主列表里的
`longbench_*` 子任务，不来自 NIAH_BENCHMARKS**——`none` 会把 LongBench/LongBench_e
整段跳过，只测出 niah，凑不出下面这张表。用 `majob.sh` 不传 `BENCHMARKS`/
`NIAH_BENCHMARKS`（吃默认值）就是当年跑基线用的同一套方法，最保险；如果只想要
LongBench/LongBench_e/niah、不想陪跑常识任务，改用
`DIAG_ARGS="..." bash eval.sh exp/qwen1.7b-32k/arc_warmup.yaml <逗号分隔的 longbench_* 任务列表> niah_single_1,niah_single_2,niah_single_3`，
任务列表从 `majob.sh` 的 `DEFAULT_BENCHMARKS` 里摘 `longbench_*` 那部分（20+ 个），
手动摘容易打错，不确定就用 `majob.sh` 全量。

对比对象是 0 节表格/6.1 表格里的"+2nd order, no pins"一行（同一 checkpoint，唯一
区别是加不加这个开关）：

| 配置 | ACC | LongBench | LongBench_e | niah |
|---|---:|---:|---:|---:|
| vanilla（无 pin，均匀池化，现有基线）| 0.6113 | 0.1716 | 0.1918 | 0.0827 |
| + importance_pooling（2026-08-14 实测）| 0.6157 | 0.1753 | 0.196 | 0.0787 |

**结果（2026-08-14，已跑出）：分裂结果，不是干净的赢或输**。ACC/LongBench/
LongBench_e 都小幅变好（+0.3~0.4pp），niah 反而变差（0.0827→0.0787，相对 -4.8%）。
"vanilla"这行是本次重新跑出来复现的，跟 6.1 存档数字逐位一致，确认 checkpoint/
评测流程本身没有漂移，这次的差异是开关本身造成的，不是噪声或环境问题。

**这打破了 6.5 提出这个方向时的核心机制假设，而不只是"没达到预期"**：6.5 的
motivation 是"均匀池化把 needle 稀释成 1/width，二阶修正救不回一个一阶就被冲淡
到没法参与 softmax 竞争的槽"——如果这个机制假设对，importance pooling 应该优先
救 niah（因为 niah 是最典型的单点稀释受害者），结果却是 niah 单独变差、恰恰是
其它任务在变好。合理的解释：post-RoPE key L2 范数是一个还不错的**通用显著性**
代理（对 LongBench 这类"整体内容摘要/理解"任务有帮助），但**不是"needle-ness"
的代理**——niah 的 needle 是随机插入的事实，没有理由天然具有更高的 key 范数；
而已知的"attention sink"现象（少数 token 因为位置/句法原因具有异常高的范数，
和语义重要性无关）意味着这些高范数 token 可能在加权平均里系统性地把 needle
的权重从均匀池化保证的 1/n 挤压到更低——均匀池化对 needle 是"保底"的，重要性
加权反而可能撤掉这个保底。这也解释了为什么不是"没变化"（模型没见过这个分布，
不会用）而是特意在 niah 上定向变差：如果纯粹是分布偏移噪声，应该四个指标同向
或至少无规律，不会恰好精准打在这个方向假设最依赖的那个指标上。

**结论**：不建议直接采用当前这版（key L2 范数、lambda=1.0）重要性池化——它换来的
ACC/LongBench 小提升不能抵消 niah 的定向回退，尤其 niah 一直是本项目最受关注的
压缩质量信号（2nd order 修正当年就是靠 niah 0.032→0.0827 论证有效的，见 6.1）。

**2026-08-14 决定跳过诊断步骤，直接实现混合权重（代码已完成）**：新增
`importance_pooling_lambda`（默认 `1.0` = 纯重要性，向后兼容；`0.0` 数值上等价于
均匀池化）。六个文件都已改完并通过单测：

- `litgpt/log_kv_cache.py`：`compact()`/`_compact_tokens()` 新增 `imp_lambda`
  参数，`>=1.0` 时走原表达式（逐字节不变），否则把 alpha/份额线性插值到
  count-based 份额——`alpha = lambda*alpha_imp + (1-lambda)*alpha_w`；Chan-merge
  用的 `frac_a`/`frac_b` 用同一个 lambda 同步插值（协方差居中权重必须跟 pooling
  权重一致，见 6.5 原文）。`_flush_pairs`（真实流式路径）单独实现同样的插值——
  原始 token 的 count 恒为 1，均匀份额就是 0.5，所以是
  `frac_a = lambda*frac_a_imp + (1-lambda)*0.5`。`LogStructuredKVCache.__init__`
  新增 `importance_pooling_lambda: float = 1.0`，显式校验落在 `[0, 1]`。
- `litgpt/model.py`：`build_log_kv_cache`/`set_log_kv_cache`/
  `enable_log_kv_training` 三处新增 `importance_pooling_lambda` 参数，透传方式
  跟 `importance_pooling` 完全一致。
- `demo.py`/`eval.py`：新增 `log_kv_importance_pooling_lambda: float = 1.0`
  （走 `run_cli()`/`_o()`，`--log_kv_importance_pooling_lambda 0.3` 直接生效），
  透传到 `set_log_kv_cache`/`enable_log_kv_training`，打印语句和 eval 的 JSON
  输出 metadata 里都加了这个字段。
- `tests/test_log_kv_cache.py`：新增 `TestImportancePoolingLambda`（9 个用例：
  取值校验、`compact()`/`_compact_tokens()` 的插值闭式解核对、lambda=0 数值上
  等价于纯均匀池化、lambda=1 逐字节等价于改动前的纯重要性代码路径、端到端流式
  路径里 lambda=0.5 的槽内容确实落在 lambda=0 和 lambda=1 之间）。全量
  133/133 通过（124 条既有 + 9 条新增）。

**下一步（在有 GPU 的环境接着做）**：对同一个 warmup CPT checkpoint 扫一遍
`lambda`，寻找"LongBench 增益还在、niah 不再倒退"的甜点：

```bash
for LAM in 0.7 0.5 0.3 0.15; do
    DIAG_ARGS="--log_kv_importance_pooling true --log_kv_importance_pooling_lambda ${LAM}" \
        bash majob.sh exp/qwen1.7b-32k/arc_warmup.yaml
done
```

对比对象还是上面那张表（vanilla 0.6113/0.1716/0.1918/0.0827，lambda=1.0
0.6157/0.1753/0.196/0.0787）。解读：
- 如果存在某个 `lambda` 使 niah ≥ 0.0827（不再倒退）且 LongBench/LongBench_e
  仍高于 vanilla，说明"部分信息"确实有用、问题出在"极端偏离均匀"，这个方向值得
  继续投入（比如训一版可学权重）。
- 如果 niah 随 `lambda` 从 1 降到 0 单调地从 0.0787 爬回 0.0827、且爬升过程中
  LongBench 的增益也跟着线性消失（即整条曲线只是在两个端点间插值，没有任何
  中间点同时优于两个端点），说明"needle 保护"和"这类显著性代理"是结构性冲突，
  不存在两全的 lambda，应该转 6.2 的按 layer/width 差异化 scale 方向。

**2026-08-14 又追加了第二个正交旋钮：`importance_pooling_temperature`（代码已
完成，理由见下）。** lambda 是"整体往均匀分布混合"——即使某个 token 的范数只是
略高（不是离谱的 attention-sink 式异常值），lambda<1 也会连带把它的显著性信号
一起打折。如果 niah 的问题具体是"少数极端高范数 token 抢走份额"，更对症的手术是
只压制那些极端值，不动中等显著性的部分——这正是 temperature 做的事：
`imp = raw_imp.clamp_min(eps) ** temperature`，`temperature=1.0`（默认）不变，
`<1.0` 压缩动态范围（比如 0.5 就是开平方，把 9 倍的范数差压缩成 3 倍），
`temperature -> 0` 极限下所有 token 权重都趋于相等（等价于均匀池化），和 lambda
数学上是两条不同的插值路径，可以叠加使用（先 temperature 重塑，再 lambda 混合）。
六个文件的改法跟 lambda 完全对称：

- `litgpt/log_kv_cache.py`：`__init__` 新增 `importance_pooling_temperature:
  float = 1.0`，显式校验 `> 0`。在两处计算原始启发式的地方（`_flush_pairs` 的
  `imp_tok = rk.float().norm(...)`、`ingest_chunk` 的 `imp = k.float().norm(...)`）
  各加一行 `if temp != 1.0: imp = imp.clamp_min(eps) ** temp`——`compact()`/
  `_compact_tokens()` 完全不用改，它们已经是"消费任意 imp 张量"的通用接口，不
  关心这个张量是怎么来的。
- `litgpt/model.py`/`demo.py`/`eval.py`：三处 `build_log_kv_cache`/
  `set_log_kv_cache`/`enable_log_kv_training` 新增 `importance_pooling_temperature`
  参数，透传方式跟 `importance_pooling_lambda` 完全一致；`demo.py`/`eval.py` 新增
  `log_kv_importance_pooling_temperature: float = 1.0`（走 `_o()`）。
- `tests/test_log_kv_cache.py`：新增 `TestImportancePoolingTemperature`（6 个
  用例：取值校验、默认值 1.0 逐字节不变、`ingest_chunk` 内部 reshaping 跟手算
  `k.norm()**temp` 再喂给 `_compact_tokens` 完全一致、temperature<1 确实让一个
  9 倍范数比压缩成更接近 0.5 的份额、端到端流式路径里 temperature=0.5 的槽内容
  落在 temperature=1（原始启发式）和 lambda=0（约等于均匀）之间）。全量
  139/139 通过（133 条既有 + 6 条新增）。

**下一步**：

```bash
for TEMP in 0.7 0.5 0.3 0.15; do
    DIAG_ARGS="--log_kv_importance_pooling true --log_kv_importance_pooling_temperature ${TEMP}" \
        bash majob.sh exp/qwen1.7b-32k/arc_warmup.yaml
done
```

**关于"并行"的一个真实踩坑点，写在这里免得周末浪费 GPU 时间**：
`majob.sh`/`eval.sh` 的 eval 输出路径是 `${save_path}/evaluate/eval_results_<秒级
时间戳>.json`，`save_path` 只由 yaml 决定（`arc_warmup.yaml` 没有按 lambda/
temperature 分目录），**不会**按超参数自动分开——多个值不同的运行会把结果都堆到
同一个 `evaluate/` 目录下，靠文件名里的时间戳互不覆盖，具体是哪次跑的靠 JSON
内部的 `log_kv_importance_pooling_lambda`/`_temperature` 字段区分（已经加进
metadata 了，见上面的实现清单）。**推荐串行跑**（就是上面 for 循环那样，一个
跑完再跑下一个）——这样零冲突风险，"周末不用盯着"和"串行"并不矛盾，反正每个
任务本身要跑较久，你只是不需要在两次之间手动敲命令。如果真的要在多台机器/多个
GPU 节点上同时跑：①**千万不要为了"隔开结果"去改 `MY_REAL_NAME`**——这个变量
同时决定 checkpoint 的 `save_path`，换一个新值会让 `majob.sh` 找不到已训练好的
checkpoint，从而误判成"没有现成权重"去重新跑 1500 步训练（几个小时的 GPU 时间
白费，而且训出来也不是这次要测的东西）；②确认没有其它进程占用同一批 GPU
（`majob.sh`/`eval.sh` 会读 `MASTER_PORT` 等环境变量，同机同端口的两个 torchrun
会互相冲突），不同物理节点之间没有这个问题。

同一张对比表，同样的"存在中间点两头都好 vs 端点间纯插值"判读逻辑。如果 lambda
和 temperature 两条扫描曲线形状相似（都是端点间单调插值、没有中间甜点），说明
问题不是"精确的插值方式"，而是这整类"从 k 范数出发的确定性启发式"跟 needle
保护结构性冲突，两条路都不用再细调，直接转 6.2 方向；如果其中一条（尤其
temperature，因为它更针对性地只压制离群值）找到了甜点而另一条没有，说明"精确
压制离群值"和"整体打折"这两种数学操作对这个任务不等价，值得针对表现好的那条
再细化（比如更小的 temperature 步长，或者两者联合网格）。

**额外一个零代码、零风险的任务：6.2 的 width/layer 门控诊断**（3.1 节，早就
接好在 `eval.py` 里，从未接入生产路径）。这个不是跑 lm-eval 拿真实分数，是拿
`log_kv_diag_mode` 那套 oracle-误差诊断，回答"如果二阶修正只在 width≤N 且
layer≤M 时生效，逐 token 输出误差会怎么变"——用来在真正去改生产接口（6.2 里
明确说的"先看有没有更多信号再动"）之前，先低成本看一眼这个方向值不值。跟上面
两个 lambda/temperature 扫描完全独立，可以一起排队跑：

```bash
DIAG_ARGS="--log_kv_diag_mode baseline --log_kv_diag_second_order_max_width 4 --log_kv_diag_second_order_max_layer 14" \
    bash eval.sh exp/qwen1.7b-32k/diag_warmup.yaml niah_single_1,niah_single_2,niah_single_3 none
```

（`diag_warmup.yaml` 已经把 pin 关掉、`log_kv_diag_mode` 留了 null 能被 CLI
覆盖，见文件头注释；`second_order_max_width`/`max_layer` 随便换组合扫，这条不
产出 ACC/LongBench 分数，产出的是 `err_width_gated` 这类误差指标，解读方式见
`log_kv_diag.py` 文档字符串，不是这张主对比表能直接拼进去的数字，先当独立参考。）

**三条任务加起来的执行顺序建议**：lambda 扫描、temperature 扫描、这条诊断三者
互相独立，可以任选顺序、也可以分给不同 GPU 节点各跑一条；同一条内部（比如同一个
`for LAM in ...` 循环）按上面说的串行执行最安全。

### 8.6 第四条任务（可选、需要先看完前三条的结果）：短续训 importance_pooling

**为什么不是一开始就做**：eval-only 扫描（8.5 上面两条）测的是"CPT 权重从没见过
这个池化分布，能不能直接受益"；如果扫描完全无效（niah 在所有 lambda/temperature
下都没恢复），更可能是"信息在池化那一步就被结构性丢弃了"（比如 needle 的内容被
高范数干扰 token 在加权平均里挤出去），这种情况续训救不回来——训练只能让下游层
更好地利用"槽里已经有的信息"，不能凭空找回从未进入槽的信息。反过来，如果扫描
找到了部分有效但没完全打平 vanilla 的点，"模型没适应这个新分布"就是一个合理
解释，续训才值得投入（这正是 6.3 里"base 模型没 CPT 时 dense 检索也是 0，CPT
后才行"那个先例）。**结论：先跑完 8.5 的两条扫描，挑一个最有希望的具体
lambda/temperature 值，只续训这一个配置，不要对着整个网格续训**——续训比纯
eval 贵得多，没有先验信号的情况下对着网格挨个续训是在浪费 GPU。

准备好的配置文件：`exp/qwen1.7b-32k/arc_warmup_importance_continue.yaml`。
用法：先把文件里 `log_kv_importance_pooling_lambda`/
`log_kv_importance_pooling_temperature` 改成扫描里选中的具体数值（这两个字段
必须是非 null 的具体值——`majob.sh` 的训练阶段只传 `--config`，不经过
`DIAG_ARGS`，跟 eval 阶段的调用方式不同，见文件内注释），然后：

```bash
bash majob.sh exp/qwen1.7b-32k/arc_warmup_importance_continue.yaml
```

**这份 yaml 的安全设计（已经读过 `demo.py` 的 resume 逻辑才这样写，别自己改
`save_path`/`resume_dir` 的关系）**：
- `save_path` 是一个全新目录，跟 `arc_warmup.yaml` 的 `save_path` 不同——续训
  绝不能写回原始 checkpoint，那份 checkpoint 是所有对比表格（0.6113/0.1716/
  0.1918/0.0827 等）的锚点，一旦被覆盖这些数字就再也复现不出来了。
- 用 `resume_dir`（只读原 checkpoint 的模型权重）而不是 `auto_resume`（会去读
  `save_path` 下自己的完整训练状态）——`auto_resume` 语义是"续自己"，跟这里
  "读别人的权重、存到新地方"不匹配。代价：Adam 的 momentum/variance 从零重新
  累积，不是逐 bit 精确续训，但对几百步的短续训这是标准做法。
- `max_steps: 300` 是"续训步数"不是"目标 step"——`resume_dir` 只读权重不读
  step 计数，`global_step` 从 0 开始算。
- `log_kv_second_order_warmup_steps` 设成 `0`（不是原来的 `100`）：加载进来的
  权重已经在 `second_order_scale=0.2` 收敛，重新 warmup 100 步等于人为制造一次
  跟 importance_pooling 无关的分布扰动，会污染这次实验的结论。

跑完之后照常用 `eval.sh`/`majob.sh` 评测这个新 checkpoint（`checkpoint_dir` 指向
新 `save_path`），跟 vanilla（0.6113/0.1716/0.1918/0.0827）和同一 lambda/
temperature 下的 eval-only 数字三方对比：如果续训后的 niah 明显高于 eval-only
版本、逼近或超过 vanilla，说明"训练里见过这个分布"确实关键，值得投入更多步数或
换成可学权重；如果续训后跟 eval-only 版本差不多，说明问题不在"模型没适应"，
是这版启发式本身在结构性丢信息，回到 8.5 结尾的"转 6.2 方向"结论。
