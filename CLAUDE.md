# LogKV 项目工作记录（存档，更新于 2026-08-07）

> 本文件是给下次接续工作时用的存档，记录 LogKV（Fenwick-tree / O(log N) 显存 KV cache
> 压缩，rank-1 Σ_s/Γ_s 二阶修正）这条线目前做了什么、改了什么、卡在哪、下一步该干嘛。
> 代码层面的细节（怎么加插件、怎么跑服务）见仓库根目录上一级的 `~/CLAUDE.md`；这份
> 只讲 LogKV 这一个专题。

## 0. 现状速览（2026-08-07，只想快速接续就读这节，细节看后面对应章节）

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

**怀疑的问题（按当前置信度从高到低）**：
1. **pin 是负贡献，且大概率是训练/推理分布不一致导致的**——训练路径
   `_log_kv_train_lowmem_forward` 完全不涉及 pin（`_log_kv_select_pins` 是
   `@torch.no_grad()`、只在推理时调用），所以不管 yaml 里 `log_kv_pin_size` 写多少，
   模型从没在训练里见过"精确 pin token + 同一 token 被池化稀释的粗糙副本同时存在"
   这种输入结构。还没做的验证：pin 选的位置到底准不准（见 6.1 末尾的诊断计划）。
2. 二阶修正本身是正贡献（niah 0.032→0.0827），但**大部分下游损失在纯均值池化阶段就
   已经发生**（vanilla niah 就已经比 dense 掉了 97%），所以第 2 节"rank-1 在宽槽下
   失真"未必是当前最大的病灶——32768 长度下槽宽可能远超 width=8，rank-1 表达能力
   本身就不够，这正是 pin 该顶上的场景。
3. ACC（常识推理）四组几乎无差异，问题集中在长程检索类任务，短程信息保留良好。

**下一步方向**：优先级从高到低——(a) 写只读诊断脚本，对比 `_log_kv_select_pins`
选中的 pin 位置和 niah 真实 needle 位置，判断 pin 选择准不准（还没做，等待拍板）；
(b) 视 (a) 结果决定是修 `_log_kv_select_pins` 本身，还是要不要让训练也引入 pin 态；
(c) 6.3 提到的"按 layer/width 差异化 second_order_scale"接口改动，明确排在 pin 排查
之后，pin 修好后再评估还有没有必要做。

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

### 3.6 本次会话（2026-08-07 续，纯排查，未改动任何生产/测试代码）

上一次会话（3.1–3.5）改了代码；这次会话只做了三件事，**都没有touch生产代码**：
1. 帮用户核对了 warmup CPT（scale=0.2）五组下游指标数据，确认口径（ACC 笔误、niah
   取 32768 档均值的约定）——结果见 6.1 表格。
2. 读代码定位了"pin 拖累指标"的大概率根因：`model.py:724-725`
   （`_log_kv_train_lowmem_forward`，训练路径）与 `model.py:865/1013-1020`
   （`_log_kv_select_pins`，推理路径 `@torch.no_grad()`）之间完全没有交集——训练从不
   构建 `LogStructuredKVCache`，`log_kv_pin_size` 这个训练配置字段是死参数。
3. 读了 `majob.sh:377,396-408` 和 `unused/parse_lmeval_table.py:338`，确认 6.1 的
   niah 数据没有踩中 6.1 原先担心的"`limit` 截断到浅层 width"的坑（`majob.sh` 调
   `eval.py` 时不传 `--config`/`--limit`，全量跑；niah 数字按约定取的是 32768 档）。

产出的诊断脚本（6.1 末尾"下一步"里提的 pin-vs-needle 对比）**还没写**，等用户决定
要不要做。

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

**下一步（尚未执行，等待决定要不要做）**：写一个只读诊断，对着现有 checkpoint 跑
几条 niah_single_1 样本，把 `_log_kv_select_pins` 选中的 `_log_kv_pin_indices`
（`model.py:924`）跟数据集里真实的 needle 插入位置对比：
- 如果 pin 选得准但指标还是差 → 印证"训练从没见过 pin 态"的分布不一致假设，下一步
  要决定要不要让训练也引入 pin 态（工程量较大）。
- 如果 pin 选得不准 → 问题在 `_log_kv_select_pins` 本身（`log_kv_pin_obs_window=64`、
  `kernel_size=7` 聚类之类），修起来便宜很多，不涉及重训。

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
