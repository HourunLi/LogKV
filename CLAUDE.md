# SemanticLogKV 开发指导文档（存档，随讨论推进持续更新）

> 本文件是 `semanticLogKV` 分支的**入口文档**，记录"语义簇 + 簇级抽象位置"这条压缩
> 方案的动机、核心设计和变更历史。**每次讨论出现突破或进展，都要写回文档**（变更
> 记录一律加到本文件 §14，细节改到对应章节所在的文件），不要另开新文档。
>
> 原 `logKV`/`claude/semantic-cluster-log-compression-8ai32a` 分支上的 CLAUDE.md 是
> 另一条独立技术路线（位置分桶 + rank-1 二阶修正 + pin 机制）的进展存档，与本分支
> 无关，不在这里维护——那条线的结论只在 §1、§10 里作为背景锚点和教训引用。

## 文档地图

内容按章节拆到四个文件，**全局章节编号连续**，所以 `§5.19`、`§11-A`、`§2.3` 这类
交叉引用在任何一个文件里都指向同一处。

| 文件 | 章节 | 内容 |
|---|---|---|
| **CLAUDE.md**（本文件）| §0–§4、§9、§10、§14 | 现状速览、背景、**核心设计**、捞针论证、最坏情况兜底、相关工作、另一分支的教训、**变更记录** |
| [`docs/algorithm-spec.md`](docs/algorithm-spec.md) | §5 | 算法规格与实现方案：记号、分簇、簇内压缩、buffer 清单、文件级改动、实现顺序、**tips 与易错点**、**复用边界**、**开工前必须定死的五个决定** |
| [`docs/experiments.md`](docs/experiments.md) | §6–§7 | 实验协议（Stage 0–3，含决策门）、消融表 |
| [`docs/risks-and-open-questions.md`](docs/risks-and-open-questions.md) | §8、§11–§13 | 风险与对策、**技术难点清单**、未决问题、压缩机制的剩余空间 |
| [`docs/glossary.md`](docs/glossary.md) | —（速查） | **术语表**：结构层次、维度记号、参数、废弃记号、代码符号、外部概念 |

**先读顺序**：想知道"为什么这么设计"读本文件 §2；想动手实现读
`algorithm-spec.md` 的 §5.19/§5.20 加 `risks-and-open-questions.md` 的 §11；
想知道"值不值得做"读 `experiments.md` 的 Stage 0 决策门；**看不懂某个词或符号就查
`glossary.md`**（尤其 `B` 在代码里有两个含义这个坑）。

## 0. 现状速览（2026-08-14）

**这是什么**：LogKV 现有压缩机制（`litgpt/log_kv_cache.py`）按**位置**做 Fenwick 树式
分层压缩——固定 2:1 合并，越老的 token 槽越宽、压得越狠。已验证的结论（另一分支的
存档）：这套方案的最大瓶颈是**槽内异质性**——语义无关的 token 被硬塞进同一个位置槽
做均值池化，needle 稀释成 1/width，niah@32768 从稠密上限 0.9353 掉到 0.0827。

**本分支的方案**：把"按位置分桶"换成"按语义分簇"——语义相近的 token 才共享槽，
needle 因为和周围内容语义不相关而倾向于自成一簇，槽内不再异质，稀释问题从结构上
消失。配套需要解决两个新问题：①位置信息不能再靠 post-RoPE 均值池化表示；②簇数
不能固定，需要由内容驱动、预算仍是次线性的自适应机制。

### 设计迭代史（四轮，v3.1 是当前版本）

| 版本 | 位置表示 / 路由 | 为什么被取代 |
|---|---|---|
| v1 | 逐频率复数统计量 `z`（特征函数）+ 标量 `β` 插值 gain | β 把"往哪转"和"多可信"耦合进一个标量，且 β=0 时对最不可信的频率通道施加最大重物化力度，方向性错误 |
| v2 | `z` + 恒定满模长旋转 + 独立的 `κ·log ρ` 置信项 | 修好了"不确定性污染内容"，但 **z 本身仍是加权和，相位相消依然发生**，只是被诚实报告出来而已 |
| v3 | 真实锚点位置集 `(p_lo, p_hi, p_mid)`；路由加时序门控，打断即 **fork 新簇** | 锚点表示保留；但 fork 把**簇身份**和**存储局部性**绑死，高频复现实体会炸掉 K（§2.1 反例一）|
| **v3.1（当前）** | 锚点集不变；路由拆成 **cluster（语义身份）/ segment（时序段）两层** | — |

**v3 的核心转变（保留）：不再试图"把多个位置压成一个数"，而是不做任何平均。** 每个
entry 存它覆盖范围内**真实成员**的边界与集中锚点，读出时在这些真实位置上各做一次
标准 `apply_rope`。每个虚拟槽的 `ρ` 恒为 1，**相位相消从数学上根本不会发生**。

**v3.1 的核心转变**：时序打断不再 fork 出新的**簇**，而是在同一个簇下开一个新的
**segment**。簇只负责语义身份（一个 centroid，用于路由），segment 负责存储局部性。

**当前阶段：设计与算法规格已完成（§5），代码尚未开始写。**

> **务必读清楚这句话字面的意思，不要被下面大段的伪代码/公式/`raise
> ValueError(...)` 片段误导。** `litgpt/`、`tests/` 里不存在
> `log_kv_semantic_clusters`、`op_log`、`current_segment` 等任何本文档描述
> 的字段或分支——可以用 `grep -rn log_kv_semantic_clusters litgpt/ tests/`
> 自行验证，应该零匹配。`LogKVStreamTrainingAttention` 这个类名确实在
> `litgpt/log_kv_cache.py`/`litgpt/model.py` 里存在，但那是它在语义簇设计
> 之前、位置分桶时代就有的版本，本文档在它之上设计的改动尚未落地。变更记录
> （§14）里反复出现的"更正""这一轮修的""P0/P1"，改的都是**这份规格文本
> 自身的逻辑漏洞**——两处描述互相矛盾、一条公式在某个边界条件下算错、一个
> 反例说明某条规则不成立——不是已经在跑的代码里发现的 bug。**连 Stage 0
> 的 dump 脚本（§5.21-5，本项目该写的第一行代码）都还没有写**，Stage 1
> （生产实现）完全没有开始。

**下一步（按优先级）**：
1. **S0.0（§7）：扫 `(g_max, ℓ_block)`。** 全课题最根本的实验——一端是纯语义聚类，
   另一端退化成"连续性约束语义分段"，扫它等于直接回答"收益来自语义分组本身，还是
   仅仅来自更好的分段边界"。纯 CPU 可测，**排在所有事情之前**。
2. Stage 0 其余离线证伪实验（§7）——一次 dump + CPU 分析，决定方案值不值得往下做。
3. `litgpt/log_kv_position.py` 纯函数 + 单测（§5.14），CPU 可测，不依赖 dump 结果。
4. 视 Stage 0 结果决定是否继续 Stage 1（生产代码）。**动手前先读 `algorithm-spec.md`
   的 §5.21（开工前必须定死的五个决定）、§5.19/§5.20，以及 `risks-and-open-questions.md`
   的 §11。**

> **实际的第 0 步是 Stage 0 的 dump 脚本**（规格见 `experiments.md`）——它是上面第 1、2
> 项全部结论的输入，应当是本项目写的第一段代码。

## 1. 背景：为什么从位置分桶转向语义分簇

（完整证据链在另一分支的 CLAUDE.md 里，这里只摘录作为本方案设计依据的结论性数字。）

- 稠密 attention 上限（CPT 后模型）：niah@32768 = **0.9353**。
- 现有 LogKV（位置分桶 + rank-1 二阶修正，无 pin）：niah = **0.0827**，LongBench =
  **0.1716**，LongBench_e = **0.1918**，ACC = **0.6113**。压缩本身吃掉 85+ 个百分点。
- 机制诊断（D1 自检）：width≥8 时 rank-1 近似失真明显；根因是均值池化把 needle
  稀释成 1/width，二阶修正是 rank-1、只能加不能减，尤其救不回**读出侧**。
- 已经试过的邻近方案（重要性加权池化，不改变槽成员划分，只改池化权重）在 niah 上
  定向变差（0.0827→0.0787）——说明问题出在"哪些 token 同槽"，不是"槽内怎么加权"。

对照锚点：

| 配置 | ACC | LongBench | LongBench_e | niah@32768 |
|---|---:|---:|---:|---:|
| 稠密（CPT 后）| — | — | — | 0.9353 |
| 现有 LogKV vanilla（位置分桶）| 0.6113 | 0.1716 | 0.1918 | 0.0827 |
| SemanticLogKV（本分支目标）| ? | ? | ? | 目标 ≫ 0.0827 |

## 2. 核心设计

### 2.1 路由：cluster（语义身份）/ segment（时序段）两层

四层结构，从新到旧：

| 层 | 是什么 | 位置表示 | 规模 |
|---|---|---|---|
| recent window | 精确 token，不压缩 | 标准 RoPE | `recent_size`（1024）|
| cluster | 一个 centroid，纯语义身份，**只用于路由** | 无 | `K ≤ K_max` |
| segment | 该 cluster 一段不被打断的连续访问 | 继承自它的 entry | 每 cluster 若干 |
| entry | ladder 条目（**每 cluster 一条 ladder**）| `(p_lo, p_hi, p_mid)` | `B′ × L_alloc` per cluster |

#### 为什么必须拆成两层（v3 → v3.1，反例一）

v3 把"时序被打断"实现成"fork 新簇"，在**高频复现实体**上炸预算：一份文档里"Alice"
被提 50 次、每次都被打断，就产生 50 个语义相同的簇。三个后果：①`K` 正比于话题切换
次数而非 `log n`；②**内容越重要越会复现，于是越贵**，激励方向反了；③撞 `K_max` 后
Ward 合并的必然又是那几个 Alice 簇，绕一圈回到原点。

根因是把**簇的身份**（Alice 是谁）和**簇的存储局部性**（Alice 这次出现在哪）绑死了。
拆开之后：cluster 只有一个 centroid，`K` 不因复现而增长；segment 只标记时序段，用于
约束合并（§5.11），**不各自分配存储**。

**副产品（值得写进论文）**：segment 就是"语义制导的连续段"，于是最早被降级的
"连续性约束 Ward 合并"作为**存储层**回来了，与语义路由层是组合关系而非竞争关系。

#### 时序约束为什么是前提而不是补丁

[DeltaKV](https://arxiv.org/abs/2602.08005) 实测**超过 60% 的语义相似 token 在位置上
相距 16 以上**。纯语义聚类下"成员时间上散得很开"是常态。没有时序约束，entry 的
`(p_lo, p_hi)` 会例行覆盖大半个文档，位置信息退化成"这个簇全文都有"。

**约束必须作用在路由上，ladder 补救不了**：簇内成员按到达顺序进 ladder，而到达顺序
即位置顺序，所以 Fenwick 合并的本来就是"位置相邻的成员"。但如果成员本身散布，即使
level-1 entry（只合并 2 个成员）跨度也可能上千。**唯一的杠杆在路由。**

**理论定位**：[distance-dependent CRP](https://arxiv.org/pdf/0910.1022)（Blei &
Frazier）——ddCRP 让落座概率同时依赖距离函数 `f(d_ij)`，`d` 取时间距离即得本方案。

#### 簇内 ladder 是承重墙，不是装饰

只有一层平面簇的话，needle 一旦并入几千成员的大簇就被稀释成 1/几千——比现在的 1/32
更糟。ladder 保证新近成员保持细粒度。**ladder 的 level 0 存单 token（`w=1`）**，
不是现有方案的"2 token 合并"。这条改动是 §3 needle 论证和 §5.9 的承重点。

#### 读出端不感知簇/段结构

`log_kv_slot_attention` 只看到一个扁平槽池加有效位掩码。cluster/segment 都只是写
路径的路由元数据。这是保持张量矩形的关键（§5.17）。

### 2.2 位置表示（v3 起）：真实锚点集，不做任何平均

**病因**：v1/v2 的 `z_f = Σ w_j e^{iθ_f p_j} / Σ w_j` 本质是把多个振荡量叠加求平均。
成员位置一散布，高频通道相位就在圆上转到四面八方而抵消——数学事实，任何"求平均"式
的位置统计量都逃不掉。v2 的 `log ρ` 只是让模型**诚实地知道**相消发生了。
**要结构性杜绝相消，唯一的办法是不做平均。**

**定义**：每个 entry 存该范围内**真实成员**的位置锚点（整数，不是统计量）：

```
p_lo    = 该 entry 内最早成员的真实位置           ← 真实位置
p_hi    = 该 entry 内最晚成员的真实位置           ← 真实位置
sum_wp  = Σ_j w_j · p_j  （int64 累加器）
p_mid   = clamp( (2·sum_wp + w) // (2·w), p_lo, p_hi )  ← 加权均值，round-half-up 的
                                                          整数实现；读出时才算
```

`p_mid` 是**质心而非真实成员位置**。这一点曾经反过来（见下面的更正框），但抗相消性
**不依赖锚点是真实位置**——它只依赖"每个虚拟槽在**单一确定位置**上做一次 RoPE"，
该位置有没有真的出现过 token 无关紧要，`ρ` 照样恒为 1。

键仍存 **pre-RoPE 内容均值** `k̄_raw`（内容侧的均值没有振荡问题）。

**读出**：对同一个 `k̄_raw` 在每个锚点上分别做**标准、未经修改的 `apply_rope`**：

```
k_eff_a = apply_rope( k̄_raw, cos_cache[p_a], sin_cache[p_a] )     a ∈ {lo, mid, hi}
```

每个虚拟槽都在**单一确定位置**上旋转（`p_lo`/`p_hi` 是真实成员位置，`p_mid` 是质心），
`ρ ≡ 1`，相消不可能发生。不需要复数 buffer、gain 公式、`κ`、频率汇总——整套机制就是
原封不动的 `apply_rope`，只是在几个确定坐标上各调用一次。**"和 token 级 RoPE 一样优美"在这里是字面意义上的：它就是 token 级
RoPE。**

**合并规则**（幂等半格运算，比加权圆均值更贴合 Fenwick）：

```
p_lo   = min(A.p_lo, B.p_lo)          # 幂等半格
p_hi   = max(A.p_hi, B.p_hi)          # 幂等半格
sum_wp = A.sum_wp + B.sum_wp          # 整数加法：精确、可结合
```

三者都与合并顺序和层级无关，零累积误差。`sum_wp` 用**未归一化的整数累加器**而不是
存 `p_mid` 本身，是为了让结合律**逐位成立**（只在读出时除一次，不累积舍入）。

> **一处曾经写错的规则，务必别再犯**：早期版本写的是
> `p_mid = A.p_mid if A.w >= B.w else B.p_mid`（"继承权重更大一侧的真实成员位置"）。
> **这条规则不满足结合律**：三个等权 token 下，`merge(merge(A,B),C)` 得 `p_A`，
> 而 `merge(A,merge(B,C))` 得 `p_B`。
>
> 更糟的是它在实践中**恒等退化**：`_binary_carry` 的进位路径上每一次 `compact`
> 都是两个**等宽**块相merge，于是 `w_A == w_B` 每次都成立、`>=` 每次都选左侧，
> 归纳可得**平衡路径上 `p_mid ≡ p_lo` 恒成立**——第三个锚点贡献的信息严格为零，
> 去重后 `M` 恒为 2。只有不等宽合并（顶层饱和 §5.12、簇 Ward 合并 §5.6）才会
> 产生 `p_mid ≠ p_lo`，而那是罕见事件。
>
> **代价要诚实记一笔**：这个 bug 的副作用恰好压低了 `M`。修好之后 `p_mid` 通常
> 严格落在 `(p_lo, p_hi)` 内，`M` 从 2 变成 3，**直接顶到 §4 那张表里"最坏
> E[M]=3 ⇒ 4984 槽 ⇒ 超过 vanilla 的 3584"那一行**。所以 S0.6 从"验证性测量"
> 升格为**真正的决策门**：若实测 `E[M]` 逼近 3，退路是砍掉第三锚点只留
> `(p_lo, p_hi)`——反正按修复前的实现它本来就等于 `p_lo`，砍掉相对**现状**零损失。

### 2.3 打分接入：复用现有 kernel + 一处必须做的 mass bias 修正

每个 entry 展开成 1~3 个虚拟槽（去重后，`p_lo == p_hi` 的年轻 entry 自动只剩 1 个）：

```
score_{s,a} = scale·(q · k_eff_{s,a}) + λ·log( w_s / M_s )   ← 注意分母
value_{s,a} = v̄_s
```

**`/M_s` 是必须做的正确性修正。** 若一个 entry 展开成 `M_s` 个虚拟槽而每个都带完整的
`w_s`，`+λ·log(w_s)` 会被计入 `M_s` 次，softmax 里该 entry 实际拿到 `M_s · w_s ·
exp(score)`——一个静默放大大跨度簇权重的 bug。**这个坑是 v3 引入的，写实现时必须
首先处理。**

不再需要 `κ`/`log ρ` 置信项——没有任何虚拟槽在"撒谎"。

**v3 相对 v2 的诚实代价**：

| 维度 | v2（z 统计量）| v3（锚点集）|
|---|---|---|
| 相消问题 | 诚实报告 | **结构性不存在** |
| 存储/槽 | ~128 float | 2~3 个整数 |
| 额外超参 | `κ`，需 CPT 校准 | 无 |
| 命中"典型/中心"位置 | 好（圆均值是 MMSE 点估计）| **同等好**——`p_mid` 是位置的加权均值，且在该位置做完整旋转，没有 v2 的相消代价 |
| 命中"边界"位置 | 弱（平均会模糊边界）| **精确**（`p_lo`/`p_hi` 是真实成员）|
| 槽数开销 | ×1 | 去重后 ×1~3（**S0.6 必测**）|

z 保留为 §8 的正式消融对照。

### 2.4 开放问题：均值池化无法表达 supersession（未定，S0.7 判定）

**问题是真的**：均值池化无法表达"后面的信息作废前面的"。"考了 100 分"和"更正为
90 分"平均成 95。[Gated DeltaNet](https://arxiv.org/pdf/2412.06464) 的
`S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ` 里的 Householder 项干的正是
"先擦掉当前 key 方向上已有的关联、再写入新值"。

**但"每簇维护一个 gated delta state"暂不采纳**，三条理由（第 2 条决定性）：

1. **KV cache 不该替模型做遗忘决定。** DeltaNet 擦除是因为它的 state 是**固定大小**
   的。本方案的前提是"次线性但会增长"的内存，**付得起**把 100 和 90 都以更粗的
   分辨率留着。擦除是不知道未来 query 时做的不可逆承诺。
2. **读出机制不兼容，且不兼容的方向恰好是捞针。** 线性 state 的 `qᵀS` 是所有写入的
   叠加，没有选择性。更尖锐的是：**簇内 key 高度相似恰恰是线性 state 容量的最坏
   情况**。needle 若落进 haystack 大簇，均值池化下至少贡献 1/w，delta rule 下会被
   后续写入**彻底擦除**——代价从"稀释"恶化成"抹除"。
3. **训练路径代价。** delta state 要让梯度穿过 recurrence，是训练路径重写。

**已知边界（反例二）：segment 只解决远距离更正，不解决相邻更正。** "我考了 100
分——不对，是 90 分"中间没有被打断，照样进同一个 entry 被平均。缓解（非解决）：相邻
token 在 recent window 里长期精确保留，真正合并时也是 level 0/1 的小 `w`。
**不能宣传成"解决了 supersession"。**

**能保留这个洞察的落地方式**：现有 rank-1 Γ **结构上已经是一个秩 1 线性 state**——
`read_s = v̄_s + scale·γ_s·(q·γa_s)·γb_s` 就是 `qᵀ(γa γbᵀ)`。所以只需改 `Γ` 的
**构造方式**：从二阶矩改成 delta-rule 式更新。均值 `v̄` 作为不可擦除的底座保留、
softmax 选择性读出保留、改动局限在 `_pair_rank1_stats`/`compact` 内部。
**这条路径对相邻更正同样有效**，是目前唯一能覆盖反例二的方案。

**判定实验 S0.7**：统计簇内成员 value 是**互相并存**还是**系统性作废**，**分相邻/
远距离统计**。靠实验解决，不靠辩论。

### 2.5 悬而未决的根本性质疑：key 空间未必是语义空间

我们说"语义簇"，但实际聚类的是 **attention key 空间**，而 key 是"这个头做匹配需要的
东西"，不一定是主题语义。不同头分工差异很大：有的做位置匹配，有的做句法角色匹配。
对句法头来说"He/She/It"可能聚成一簇而跟指代对象分开——这时"簇有持续话题身份"这个
前提**根本不成立**。层间也一样：早期层的 key 接近 token embedding 加位置。

**但这也是自洽的**：per-head 聚类下，若某头的 key 本就按位置组织，它的簇天然时序
连续，segment 规则自动变成 no-op，不造成伤害。所以这不是 bug，而是意味着
**整套方案的价值在不同头、不同层之间差异会很大**。

**操作后果（硬性要求）**：Stage 0 的所有统计**必须逐层 × 逐头分开报告**。"平均看起来
还行"可能是少数中层语义头很好、多数头完全无效的叠加，而这两种情况对应完全不同的
下一步。

## 3. 为什么能解决捞针

**稠密**：`score = scale·(q·k_needle)`，`value = v_needle`。

**现有池化（needle 在宽度 w 的槽里）**：

```
k_slot   = ( k_needle + Σ_{j≠n} k_j ) / w
q·k_slot = (q·k_needle)/w + (w-1)/w · (q·k̄_hay)
v_slot   = ( v_needle + Σ_{j≠n} v_j ) / w
```

分数侧和读出侧**同时**被稀释 w 倍。`+λ log w` 补的是计数质量而非内容；即使赢得注意力，
返回的 `v_slot` 也已是 haystack 均值，而 Γ 是 rank-1。这解释了为什么二阶修正把 niah
从 0.032 提到 0.0827（2.6×）却依然离 0.9353 极远：**它修的是分数，修不了读出。**

**SemanticLogKV**：needle 与所有 centroid 距离 `> λ_new` ⇒ 自成一簇 ⇒ 成员数 ≤ B′ ⇒
**ladder 永不填满 ⇒ entry 永不合并**（§5.9）；且 `p_lo = p_hi = p_mid` ⇒ `M = 1`：

```
score = scale·(q · k_needle) + λ·log(1/1) = scale·(q · k_needle)   ← 与稠密逐位相同
value = v_needle                                                    ← 与稠密逐位相同
位置  = 在 p_needle 上的标准 RoPE                                    ← 与稠密逐位相同
```

**分数、读出、位置三者同时恢复到稠密**，而且这个论证**不依赖任何超参**。

代价只有一个槽——haystack 语义同质、`K_eff` 小、压得很狠，省下的预算转移给异常点。
**这就是为什么自适应预算不是锦上添花而是唯一机制**：固定均匀预算在信息论上无法同时
做到"压缩 haystack"和"保留 needle"。

**失败模式（Stage 0 必须证伪）**：`λ_new` 太大 → needle 并入 haystack 簇 → 退回稀释。

## 4. 最坏情况的双重兜底

**表现的地板**：内容不可压缩时 `K` 增长 → 槽变多 → 质量向稠密平滑逼近。`K` 撞
`K_max` 时强制 Ward 合并，把结构推回**"少数大簇 + 位置序 ladder"**——这是"今天
LogKV 的组织原则"（少数槽、按位置排序合并），**不是数值上等同于今天的 LogKV**：
level 0 单 token（不是 2-token 合并）、真实锚点位置表示（不是 z 统计量/expected
RoPE）这些改动是设计里不可选的（§2.1、§5.6 的 `K_max=1` 讨论已经验证过这一点，
即使 `K_max` 压到最小，分数也不会复现旧数字）。退化的终点是**同一类结构**，不是
比现状更差，但也不是逐位复现现状——这个区分在写论文对照实验时要保持精确。

**性能的天花板**：内存**硬性有界**（§5.12 的预分配）。这里必须区分两笔账（§11-C）：

| 口径 | vanilla | SemanticLogKV @32k | 用途 |
|---|---:|---:|---|
| **持久 cache 内存**（entry 数）| 3584 | **2344** = 1024 + 15×8×11 | memory-matched 公平性对比 |
| 读出时槽池（entry × M，瞬时）| 3584 | ~3004（E[M]=1.5）| 计算量/峰值内存 |
| 同上，最坏 E[M]=3 | 3584 | 4984 ✗ | **S0.6 的决策门** |

**锚点展开的 `M` 只影响读出时的瞬时槽池，不影响持久 cache 内存**——每 entry 存的是
3 个整数，不是 3 份键值。

**第三笔账，此前漏记：训练期 `op_log` 重放元数据，约 448MB/28 层@32k，**且是
每个尚未执行 `backward()` 的 in-flight forward 各付一份**（`algorithm-spec.md`
§5.21-2 的推导）。这笔账只属于训练侧，不属于上面两笔里的任何一笔，也不进
memory-matched 对比**——`op_log` 是给 backward 重放用的操作日志（§11-A），
serving/推理路径不做反向传播，**不分配、不持有这块内存**；这不是"推理时反正
用不上所以顺便不管"的隐式结果，是共享的路由/flush 逻辑显式接收一个
`record_op_log` 开关，只有训练专用的 `LogKVStreamTrainingAttention.forward()`
传 `True`，推理用的 `LogStructuredKVCache.forward()` 传 `False`，见
`algorithm-spec.md` §5.21-2 新增的"`op_log` 只能在训练路径分配"一节。上面
两笔账比较的是"cache 里究竟存了多少 entry"和"读出时瞬时展开多大"，两者都是
serving 也会付的代价，`op_log` 不是。

**"448MB" 不是训练期的固定开销，是每个 in-flight forward 的单价**：只要
下一次 `forward()` 在这次 forward 对应的 `backward()` 跑完之前发生，两份
448MB 就会同时存活，训练峰值因此是 `448MB × 同一时刻并存的 in-flight
forward 数`。本仓库 `litgpt/pretrain.py` 的梯度累积循环每个 microbatch 都
立即调用 `fabric.backward()`（只有 `optimizer.step()` 被推迟），所以在这条
训练循环下 in-flight 数恒为 1，"448MB"就是准确的峰值；但"累积 loss、只在
最后统一调一次 `backward()`"或 pipeline 并行的 microbatch 调度会让这个数字
按并发 forward 数相乘，完整分析、以及哪些模式安全/哪些需要重新核算，见
`algorithm-spec.md` §5.21-2 新增的"『448MB』只是每个 in-flight forward 的
代价"一节——训练时这笔额外的显存是真实成本，但它和"压缩率""memory-matched
公平性"这两个 serving 侧的论证是两件独立的事——放进同一张表比较会把训练
开销和推理内存预算混为一谈，所以单独列出，不进上面那张表。

> **这笔账中途差点被算错一次，教训值得留着**：为了让 `op_log` 安全地活过
> `backward()`（不被下一次 `forward()` 的 reset 清空），中间一版实现是
> `ctx.save_for_backward(cache.op_log.detach().clone(), ...)`——这个 `clone`
> 让训练峰值一度变成 448+448=896MB，因为 cache 自己的持久 448MB 和 ctx 里的
> 克隆同时存在。**最终方案不是克隆，是让 `op_log`/`op_log_len` 不再走"持久
> buffer + `reset_parameters()` 原地清零复用"这条路**——它们改成每次
> `forward()` 开头重新绑定成全新分配的张量，直接存进 `ctx`，不需要克隆，
> 训练峰值因此回到单份 448MB，就是本节这笔账的数字，不是 896MB。完整推导见
> `algorithm-spec.md` §5.21-2"训练峰值显存"更正框。

**关于"期望 O(log n)、最坏 O(N)"这个契约**：作为**论文的空间复杂度主张**它成立，
而且比"期望"更强——CRP 簇数是独立 Bernoulli 之和，Chernoff 给出多项式小的尾概率，
是**高概率界**。但作为**实现契约**不能照字面做，因为预分配必须按最坏情况，最坏
`O(N)` 就等于稠密。正解是**把最坏情况从内存挪到质量上**：内存硬性有界，多样性超标时
`K_max` 绑定 → Ward 合并 → 平滑退回"少数大簇 + 位置序 ladder"这一类结构（不是数值
上复现现有 LogKV，理由同上）。对 serving 而言这比"最坏 O(N) 内存"是**更好**的契约，
因为永不 OOM。真要做内存也浮动的版本，标准做法是 shared block pool + 动态分配
（vLLM PagedAttention 的结构），推迟到工程化阶段（§5.17）。

> **论文最大的攻击面**：自适应预算意味着 niah 变好可能仅仅因为多用了内存——而 niah
> 恰恰最容易把 K 推高。**从第一个实验开始就必须报告两笔账**，并准备 memory-matched
> 对照。留到最后补，整套实验要重跑。


## 9. 相关工作定位（2026-08 快照，务必用 search 复核最新情况）

| 工作 | 做了什么 | 与本方案的区别 |
|---|---|---|
| SemantiCache (2603.14303) | 按分隔符切语义块 + 贪心种子聚类合并 | **先分段再聚类**；本方案**先聚类再分段**。且它固定 budget、无簇内分辨率层级、frozen model |
| Multipole Attention (NeurIPS 2025, 2506.13059) | key 做 k-means，远处用 centroid 近似打分 | **保留全量 KV（O(n) 显存）**，centroid 只是索引，不是压缩 |
| SeKV (2606.31145) | entropy 引导语义 span，GPU summary + CPU SVD 重建 | offload + 检索系统，什么都不丢，非次线性 |
| ClusterAttn (ACL 2025) | 密度聚类自适应簇数 | 稀疏注意力，固定 1024 token 预算 |
| GVote (ICLR 2026, 2509.03136) | 蒙特卡洛采样未来 query 免手工设预算 | eviction/selection 框架，预算仍是 O(1) 固定池 |
| DeltaKV (2602.08005) | 残差编码 KV，利用长程 token 相似性 | 不是次线性；**但它实测的"60% 相似 token 相距 >16 位置"是 §2.1 时序约束的实证依据** |
| Gated DeltaNet (2412.06464) / GDN-2 (2605.22791) | 门控 + delta rule 的固定大小线性 state | 固定容量下的覆盖式记忆；§2.4 详述为何不直接采用 |
| ddCRP (0910.1022) | 距离依赖的 CRP，落座概率含距离函数 | §5.3 join cost 的理论出处 |
| **CompressKV** ([2508.02401](https://arxiv.org/abs/2508.02401) / [2606.24467](https://arxiv.org/abs/2606.24467)) | 识别 **Semantic Retrieval Heads**（同时抓首尾 token 与中段语义证据），据其选择保留哪些 KV；层间预算按"压缩前后 attention 输出的 Frobenius 差"分配 | **eviction 而非压缩**——被丢的 token 不可恢复；预算是 O(n) 的固定比例，不是次线性。**但它的数字必须正视**，见下方 §9-A |
| **ChunkKV** ([2502.00299](https://arxiv.org/pdf/2502.00299)) | 保留**连续 token 块**而非离散 token，以维持局部连贯 | 同为 eviction；**但"连续块优于离散选择"这个实证结果是 S0.0 的旁证**——它从 eviction 侧支持"分段边界本身有价值" |

### 9-A. 必须正视 CompressKV 的数字，以及本方案的立足点在哪

CompressKV 报告：LongBench 用 19% 预算保住 99% 满 cache 性能、3% 预算保 97% QA；
**NIAH 用 0.7% 存储达到基线的 90%**。0.7% × 32k ≈ 229 个 token，而本方案的预算是
2344 entry ≈ 7%。**用十分之一的预算拿到远高于我们目标的 NIAH 分数**，这对
"压缩是必要的，因为 eviction 会丢信息"这个动机是直接冲击，不能装作没看见。

三条区分是真实的，但要说清楚各自的适用边界：

1. **它是 eviction，本方案是压缩。** 它靠 prefill 期观察到的注意力模式挑该留谁——
   这在"问题已在 prompt 尾部"时极其有效，而 NIAH 恰好就是这个形态。**这正是本项目
   pin 系列的死因之三（观察窗口的架构性局限，§10）**：pin 当初在原理上也很漂亮，
   死于同一个假设。所以这条区分不是托词，是我们自己付过学费的。
2. **渐近不同。** 0.7% 的 32k 是 229，0.7% 的 1M 是 7340；本方案 32k→1M 是
   2344→3424。固定比例再小也是 `O(n)`，长到一定程度会反超。
3. **不可逆承诺。** eviction 在不知道未来 query 时就丢弃，多轮对话里后续轮次问到
   被丢内容就没了——这与 §2.4 拒绝 delta rule 的理由同源，且对 eviction 更强。

**但诚实的结论是：单针 NIAH 已经不是能证明本方案价值的实验。** 真正有区分度的是
**multi-needle、多轮、以及 query 不在 prompt 尾部的设定**——前两个 §7 已经列了，
第三个应当补进消融。若在这些设定下仍打不过一个 3% 预算的 eviction 方法，那么
"次线性压缩"的动机需要重新论证，而不是继续往下堆工程。

**另一处需要让路的新颖性主张**：§13.2 说"每层固定 `B′` 个 entry 这条分配曲线从没被
论证过"，但 CompressKV 的**层间预算按 Frobenius 误差分配**在方法论上就是同一件事
（测量驱动的率失真分配），只是分配轴是"层"而不是"ladder 层级"。§13.2 的提法要改成
"**沿 ladder 层级**的分配曲线未被论证"，并引 CompressKV 作为跨层版本的先例。

空着的生态位：**非参先验决定簇数 + 严格次线性空间 + 语义身份与时序局部性两层分离 +
不做平均的真实锚点位置表示**。多次搜索没有找到任何工作用非参贝叶斯给簇数一个先验、
同时保持严格次线性显存。

## 10. 复用另一分支的基础设施与教训（精简版）

**RoPE 布局约定**：`apply_rope`（`litgpt/model.py:2074`）是"前后半重复"布局，
`cos[f]==cos[f+d/2]`，partial rotary 只旋转前 `rope_n_elem` 维、尾部内容通道不旋转。
§5.14 的锚点物化直接依赖这个约定。

**pin 系列四条死因（设计规避清单）**：
1. 训推不一致——本方案路由同样 `no_grad`，但训练和推理走**同一条** cache 更新路径。
   **注意 §11-A 和 §11-E 是这条的新变体，必须显式处理。**
2. 分数尺度失配——本方案槽种群完全同质（都是池化槽，包括单点簇）。
3. 观察窗口的架构性局限——本方案路由在 flush 时逐 token 发生，不需要看到整段 prompt。
4. 剂量效应——本方案不新增槽种类，只改变成员划分。锚点展开形式上像"插入更多槽"，
   但它们共享同一个 `v̄` 和被摊薄的 mass bias，与 pin 的剂量效应不同源。

**工程习惯**：
- `eval.sh` 把 YAML 展平成纯 CLI flag（不传 `--config`），`_o()` 的"YAML 非 null
  覆盖 CLI"逻辑不会触发；直接用 `--config <yaml>` 会触发。新增会被扫参覆盖的字段，
  YAML 里必须写 `null`。
- `majob.sh` 如果 `save_path` 下已有 checkpoint 会跳过训练直接 eval——新实验必须用
  独立 `save_path`。
- 本地跑 pytest 用 conda env `mineru`（`/Users/hourunli/anaconda3/envs/mineru`，
  Python 3.12）；系统自带 Python 无法解析仓库里到处用的 `X | None` 类型注解。

## 14. 变更记录

> 每次讨论产生突破或进展,在这里加一条,新的在最上面。只记"改变了什么结论/设计",
> 不重复已经写进正文的细节——细节改到对应章节,这里留指针和一句话动机。

- **2026-08-19｜第十三轮核实：把 Phase 3a 仿射 scan 的 `γ` 定义域从 `(0,1]`
  改成 `[0,1]` 并说明 `γ=0` 时公式仍精确成立、把 S0.8 的 Ward 事件比较从
  "逐一比较候选对"细化成可执行的对齐算法、把最终 cache/readout 差异拆成
  "只在 token 集合完全相同的簇上比较字段"（诊断）和"attention 读出 L2
  误差"（唯一决策门）两个子项、并把本地缓冲"约 8KB"的内存账目标注清楚是
  per-(batch,KV group) 而不是整层。** 动机：用户对照已合并的 PR #19（HEAD
  `e231662`）逐条核实上一轮的四处修复，指出三处 P1、一处 P2——都属于"公式
  本身没错，但定义域/对齐算法/统计口径还留了缺口"。逐条结论：
  ① **P1：Phase 3a 仿射 scan 写的 `d[t]∈(0,1]`、"`A` 只会变小不会为 0"，
  和 §5.5 明确支持 `γ=0`（"归零"分支）矛盾。** 核实后确认这不是公式本身
  的 bug——把 `γ=0` 代入仿射 scan 的合并公式（`A=A_后·A_先` 等）逐项验证，
  在 `A=0` 处全部良定义：`n_eff_new`/`centroid_new` 精确退化成
  `Bn_last`/`S_last/n_eff_new`，即"只看衰减之后的内容"，和 §5.5 的
  "没有历史可混，直接替换"语义完全一致，不需要任何特判分支。**问题纯粹
  出在文字表述**：三处写法（`d[t]∈(0,1]`、"`A`只会变小不会为0"、"`γ∈(0,1]`
  时结果通常仍非零"）都隐含排除了 `γ=0`，容易诱使实现者写出
  `assert A>0`、`log(A)`/`1/A` 之类只在 `γ>0` 时安全的"优化"，在 `γ=0`
  处静默崩溃或算错。**修法**：定义域改成 `[0,1]`，三处都补充说明
  `γ=0` 时 `A` 精确变成 `0` 是符合语义的正确结果，不是数值边界；同时
  纠正了一处过度引申——`γ=0` 时单看 `n_eff` 确实退化成类似 `PAD_INSERT`
  的固定重置，但 `centroid`（`μ`）不会，它依然需要真正累积段内成员的
  内容而非只知道"过了几步"，所以"不能直接套用 steps_since"这条结论
  对 `μ` 在任何 `γ` 下都成立，不能因为 `n_eff` 在 `γ=0` 这个特例上像
  固定重置就反过来说"那就可以用 steps_since 了"。新增要求：
  §5.4"必须补的单测"第 3 条的对拍数据必须覆盖 `γ=0` + 本批内同一簇连续
  多次 `NEW_SEGMENT` 这个最容易让"`A` 恒为正"假设现形的场景。
  ② **P1：S0.8 的 Ward 事件比较说"逐一比较候选对"，但没有给出可执行的
  对齐算法。** 两条路径触发 Ward 合并的次数、触发时刻、候选对用的槽号/
  `epoch` 都可能因为批量近似而不同（谁先把 `K_max` 填满这件事本身可能
  发生在不同 token 上），"逐一比较"字面读容易被实现成按下标顺序两两
  对齐，产出的差异其实是伪差异。**修法**：给出具体对齐算法——用触发
  这次合并的 orphan 的**绝对 token 位置**（两条路径共有、不受路由近似
  影响的锚点）做主键对齐两条路径的事件列表；候选对不用槽号表示，改用
  合并前 `keep`/`free` 两个簇各自的**原始 token 绝对位置集合**表示
  （`scan_op_log` 扫到那一步即可得到），触发位置对上的事件比较这两个
  集合的 Jaccard 相似度；触发位置只在一条路径出现的事件不强行配对、
  不平均掉，单独计为 `ward_event_inserted`/`ward_event_deleted`——"事件
  没有对应物"和"配对上但决策不同"是两种不同的失效模式，混在一起会
  互相掩盖。
  ③ **P1：最终 cache/readout 差异的"对齐到同一组 token"缺具体算法，且
  和第 1 项的成对一致率（只给标量，不给簇的一一映射）实际上对不上。**
  聚类若发生真实分歧（不只是槽号错位），"这个槽对应那个槽"可能根本没有
  良定义的答案（比如批量路径合并了两个簇、严格串行参考没合并，找不到
  唯一对应的严格串行簇），此时逐字段比较 ladder 会产出没有意义的数字。
  **修法：拆成 3a/3b 两个子项，决策门只挂在 3b。** 3a（诊断，不参与决策）
  只在"两条路径的簇 token 集合完全相等"（不是重叠/相似）的子集上逐字段
  比较，落不进这个精确匹配子集的 token 单独报告成 `unmatched_token_ratio`，
  需要更细粒度诊断时可选用 Jaccard/匈牙利算法做软匹配但不进正式数字；
  3b（决策依据，唯一硬性门槛，< 5%）直接比较两条路径最终 cache 状态给出
  的 attention 读出相对 L2 误差——这一项不需要任何簇对齐，两条路径的簇
  怎么编号、有没有一一对应完全不影响它是否良定义，是唯一在聚类真实分歧
  时依然能给出干净数字的一项。`experiments.md` §6 的 S0.8 决策门措辞同步
  从"第 3 项"改成"3b"。
  ④ **P2：本地缓冲"约 8KB"的内存账漏了作用域，容易被读成整层或全局的
  数字。** `local_op_cap=4×flush_granularity` 默认 512 行 int32，
  `512×4×4B≈8KB` 说的是**单个 `(batch 元素, KV group)` 切片**的大小（和
  §5.13 buffer 表"前两维都是 `(B,G)`"的约定一致，本地缓冲完整形状是
  `(B,G,local_op_cap,4)`），不是整层、更不是整模型的数字。`B=1,G=8`
  时一层约 `8KB×8≈64KB`，28 层约 **1.8MB**——仍然远小于持久 `op_log` 的
  448MB，结论不变，但补上完整展开的数字，避免又把一笔小账错读成全局账。

- **2026-08-19｜第十二轮核实：厘清 `record_op_log=False` 下本地 op 缓冲的
  构建/持久化边界、把 Phase 2 的 orphan 判定钉死为"批内冻结、不随 Phase 3a
  的 metadata 更新重判"、把 S0.1 的 Ward 场景参考实现从"只吃主操作"改成
  "吃完整 op 序列"、并给出 Phase 3a 的 `n_eff`/`centroid` 向量化更新此前
  缺失的仿射扫描推导。** 动机：用户对照 `algorithm-spec.md`（HEAD
  `3aca2dc`）逐条核实，指出三处 P1、一处 P2——均属于"接口边界和实现公式
  还没钉死"，不是大方向错误。逐条结论：
  ① **P1：`record_op_log=False` 和"本地 op 缓冲"此前自相矛盾。** §5.4
  通篇（Phase 1/2/3 的伪代码）假设本地缓冲无条件存在、由 Phase 3a/3b
  消费去驱动 metadata 更新、Ward 合并候选选择、ladder 物理写入；但
  §5.21 的 `record_op_log` 门控代码块把 `append_ops_to_local_buffer`
  整体挂在 `if record_op_log` 之下，字面读会让推理路径的 Phase 3a 无 op
  可读，路由机制直接失效。**决定：本地缓冲（含它驱动的全部路由、
  metadata 更新、ladder 物理写入）在训练/推理两条路径上无条件构建和
  消费，不受 `record_op_log` 影响；`record_op_log` 唯一门控的是这一批
  处理完之后，要不要把已经写满的本地缓冲整体提交进跨批持久的
  `op_log`（供 backward 重放）——这一步在推理路径上跳过，本地缓冲随即
  被丢弃/被下一批复用，不留任何持久状态。** 本地缓冲自身很小
  （`local_op_cap=4×flush_granularity`，默认 512 行 int32、约 8KB），
  两条路径都构建它的开销可忽略，不会重新引入 448MB 那笔训练专属账目。
  §5.4"本地缓冲 vs 持久 op_log"一节补一条更正框，§5.21 的门控伪代码从
  "整体挂 if record_op_log"改成"路由/metadata/ladder 写入 unconditional，
  只有『提交进持久 op_log』这一步挂 if record_op_log"。
  ② **P1：Phase 3a 更新 metadata 后，Phase 2 是否重新判断 orphan 此前
  未钉死。** Phase 1 用批前冻结的 centroid 算出 `c*[t]`/`s*[t]`，据此把
  token 分成 direct/orphan；Phase 3a 随后更新 centroid；若 Phase 2 处理
  orphan 时用更新后的 centroid 重新判断，可能让某个 orphan 相对新
  centroid 已经 `≤λ_new`——但允许它转投一个本批 Phase 1 也在写的既有簇 X，
  会把这条 `JOIN` 排到 X 在本地缓冲里全部 Phase 1 成员的物理顺序**之后**
  （因为"Phase 1 整组先写、Phase 2 整组后写"），直接违反"同一逻辑簇的
  主操作必须按真实到达顺序出现"这条契约，并击穿§5.4"为什么 Phase1 先于
  Phase2 满足三条契约"那节证明依赖的前提。**决定：冻结。** `s*[t]`/
  `c*[t]`/direct-orphan 划分只在 Phase 1 计算一次，Phase 3a 更新后的
  metadata 只喂 Phase 2 的 Ward 代价计算和 Phase 3b 自己的在线更新，绝不
  重新拿去判断某个 orphan 该不该转投现有簇——这是"Phase 1 冻结 centroid
  是一个已知近似，S0.8 测"这条既有框架的直接延伸，不是新的近似类别。
  §5.4 的 Phase 1/3a/2 伪代码块、"为什么满足三条契约"一节分别补一句
  钉死这一点。
  ③ **P1：S0.1 的 Ward 场景参考实现描述不完整，只喂"主操作序列"验证不了
  它本该验证的东西。** 原文让参考实现"把本批最终本地缓冲里原样的主操作
  序列喂给一个……逐簇顺序重算的参考实现"，但 `WARD_MERGE` 是结构操作、
  不在"主操作"之列——参考实现看不到它，就既无法把被合并簇（`free_slot`）
  批内收到的贡献并入 `keep_slot`（断言 (i) 需要），也无法在复用槽位建新
  簇前清空旧内容（断言 (ii) 需要），而验证这两条正是这条测试存在的全部
  意义。**修法**：参考实现改为按物理顺序走完整本地缓冲（主操作 +
  `WARD_MERGE`，`PAD_INSERT`/`CARRY` 对 metadata 无影响可跳过），遇到
  `WARD_MERGE(keep,free)` 时对自己的 scratch 状态套用和 `ward_merge_only`
  步骤 1 完全相同的合并公式，再把 `free` 的 scratch 清零——不需要引入
  `derive_final_cluster` 那套 `(slot,epoch)` 版本化身份，因为这是一次
  真正按时间顺序执行的模拟（不是跳过时间顺序的离线并查集派生），"遇到
  NEW_CLUSTER 就清零"天然处理了槽位复用。本质上是把 backward 重放循环
  （§5.21-2）的 `append_to_ladder`/`ward_merge_only` 换成对 metadata 的
  操作，同一个模式。
  ④ **P2：Phase 3a 的 `n_eff`/`centroid` 向量化公式此前只说"复用
  pad-count 的 segmented reset scan 技巧"，这个类比是错的，补上实际
  推导。** `PAD_INSERT` 的重置落在一个固定值（mod 计数器精确回到 1），
  "距离上一次重置几步"这个无状态相对量就够了；但 `n_eff` 的 `γ` 衰减是
  把历史打折扣、不是清零（`n_eff_pre←γ·n_eff`），衰减后的值依然依赖
  衰减前的完整历史，"steps_since"这类单层查询在这里不成立。**补上的
  推导**：先识别出五个待更新字段里 `n_total`/`p_hi_c`/`current_segment`
  其实是平凡的分组归约（不需要扫描，直接复用已有的 `rank`/`nsg_incl` 取
  组内最后一个成员）；只有 `n_eff`/`centroid` 需要真正的扫描——把逐成员
  递推看成仿射变换的复合（`x↦d[t]·x+常数项`，`d[t]=γ` 或 `1`），复合
  满足结合律，可以用固定 `⌈log₂flush_granularity⌉` 轮的 Hillis-Steele
  掩码扫描算出（三元组 `(A,Bn,S)` 及合并公式已给出），和 §5.21-3 的
  `_binary_carry` 是同一类"结合律换并行"技巧，只是幺半群从二进制进位
  换成仿射复合。**特别指出**：这个仿射复合不是现成库原语（不同于
  `nsg_incl` 依赖的原生 segmented cumsum/max），需要手写扫描；也指出了
  为什么不用负指数闭式解——病态批次会算出 `γ^{-128}` 量级的中间值，
  逼近 fp32 溢出边界，扫描全程只做有界量的乘加，规避了这个风险。不需要
  新增单测，§5.4 已有的 Phase 3a/3b 元数据对拍单测（③修完之后）天然
  覆盖这条公式对不对。

- **2026-08-19｜第十一轮核实：解决"Phase 3 批末运行"这个此前一直没被注意到的
  P0——它和"Ward 合并不受限、op_log 永不改写"这两条上一轮才证明成立的设计
  互相冲突；同时把训练期 `op_log` 显存账目精确到"per in-flight forward"、
  把它的分配显式限定到训练路径、在入口文档补一条不容错过的"未实现"声明、
  并把 S0.8 的分歧率从一个模糊标量拆成三条可执行指标。** 动机：用户对照最新
  远端逐条核实，指出一处 P0、两处 P1、一处 P2、外加一条工作流建议。逐条结论：
  ① **P0：Phase 3 若像此前写的那样"批末统一运行"，会和上一轮才证明成立的
  两条设计互相冲突，导致 Ward 合并读到 stale metadata、token 内容被错误地
  计入错误的簇。** 具体反例：Phase 1 把 direct token `tok0` 路由到既有簇 X、
  写下 `JOIN(X, seg, tok0)`；Phase 2 处理某个 orphan 时 `K_max` 已满，Ward
  代价矩阵选中 `(keep_slot=C, free_slot=X)`——X 恰好是本批刚被 Phase 1 触碰
  过的簇（Ward 尺寸加权对"批内刚建立、暂时还小"的簇的偏好，上一轮已经用
  几乎一样的场景证明过这不是边界情形）。若 Phase 3 严格等 Phase 1、Phase 2
  都跑完才统一 walk 本批 ops、按裸 `op.cluster` 分组更新
  `centroid`/`n_eff`/`n_total`/`p_hi_c`，`ward_merge_only(C,X)` 执行那一刻
  X 的 metadata 仍是批前快照（不含 `tok0`），于是 `tok0` 对 X 的内容贡献在
  合并这一步被丢弃；紧接着 X 被复用建立一个全新的簇，Phase 3 用裸槽号回头
  处理 `JOIN(X,seg,tok0)` 时，又会把 `tok0` 错误地计入这个和它毫无关系的
  新簇。**根因**：本节此前已经确立"Phase 1 整体（含 ladder 物理写入）必须
  先于 Phase 2 完整跑完"这条原则，理由是 Ward 合并需要看到真实、最新的
  ladder——但同一条逻辑对 metadata 同样成立，而"Phase 3 批末运行"这个设计
  从未被拿去对照这条原则重新检查，是一个被漏掉的推论，不是一个新的独立
  问题。**修法**：把"Phase 3"从"批末的第三个步骤"改成按内容来源拆开、各自
  在能拆的最早时刻执行——**Phase 3a**（向量化，紧跟 Phase 1 完成之后、
  Phase 2 开始之前）批量更新 Phase 1 产出的主操作对应的 metadata；**Phase
  3b**（内联，逐 token）在 Phase 2 本就是串行的循环内部，每个 orphan 主操作
  写入本地缓冲后立即更新它的 metadata，不再等批末。这样任何一次
  `ward_merge_only` 执行时，它要读的两个槽的 metadata 都已经是本批目前
  为止的真实值，"stale metadata"这类输入结构性不可能出现。这不是重新引入
  上一轮刚推翻的"批内重定向"——`op_log` 依然纯追加、永不改写，改变的只是
  "消费这些 op 去更新 metadata 的时机"，一个纯调度问题，不触及 op_log 顺序
  契约本身。相应修正了：§5.4 的三阶段伪代码、"Phase 3 是唯一写入点"那段
  论证里"Ward 总是先于 Phase 3"这句被证明是错的话、"Phase 3 的具体做法"
  拆成 3a/3b 两段、"执行时机"的表述、`current_segment` buffer 的写入点
  说明、`ward_merge_only` 步骤 1 里"和 Phase 3 不冲突"那句话补上它依赖的
  调度前提、以及 S0.1 单测第 3 条补上必须覆盖这个反例场景（且要双向验证：
  `keep_slot` 精确包含被合并簇的批内贡献、复用槽建立的新簇精确不包含）。
  ② **P1：训练期 `op_log` 448MB 这个数字只对"任意时刻最多一个 in-flight
  forward"成立，不是训练期的固定开销。** `ctx.save_for_backward` 只要
  下一次 `forward()` 在这次 forward 对应的 `backward()` 跑完之前发生，两份
  448MB 就会同时存活，峰值因此是`448MB × 同一时刻并存的 in-flight forward
  数`。核对本仓库 `litgpt/pretrain.py:351-355` 后确认，梯度累积循环每个
  microbatch 都立即调用 `fabric.backward()`（`no_backward_sync` 只跳过
  DDP all-reduce，不推迟 backward 本身），只有 `optimizer.step()` 被推迟——
  这条训练循环下 in-flight 数恒为 1，"448MB"是准确的。但"累积 loss、只在
  最后统一调一次 `backward()`"（峰值 `microbatch 数 × 448MB`）和 pipeline
  并行的 microbatch 调度（峰值 `pipeline depth × 448MB`）会让这个数字按
  并发相乘，必须显式排除或预算，不能假设"训练期就是 448MB"对它们也成立；
  activation checkpointing 不属于这两类（不推高 in-flight 数），但会让
  `op_log` 多分配一次可被立即回收的 throwaway 448MB，是效率问题不是峰值
  问题。`algorithm-spec.md` §5.21-2 新增专门一节写清楚这个换算规则和这三类
  模式各自的结论，CLAUDE.md §4 的第三笔账同步更新措辞。
  ③ **P1：推理是否分配 `op_log` 的规格不一致，必须把"不分配"从隐式默认行为
  改成显式 gate。** CLAUDE.md 说 serving"不分配、不持有"，但 §5.13 的
  buffer 表只说"每次 `forward()` 开头重新绑定"，没说是哪一个
  `forward()`——`LogStructuredKVCache` 有两个独立入口（推理用的
  `forward()`，训练用的 `LogKVStreamTrainingAttention.forward()`），字面
  读容易让实现者把 448MB 的分配也带进推理路径。**修法**：两个入口共享的
  路由/flush 逻辑显式接收 `record_op_log: bool`，训练侧传 `True`、推理侧传
  `False`；不能靠 `torch.is_grad_enabled()` 判断，因为路由决策本身（DP-means
  距离比较、`argmin`）在训练和推理下都恒定 `no_grad`（CLAUDE.md §10 死因
  1），这个信号在两条路径上是一样的，真正的区分点（这次 forward 之后会不会
  有对应的 backward）只有调用方知道，必须显式传。`algorithm-spec.md`
  §5.21-2、§5.13 的 `op_log` 行、CLAUDE.md §4 三处同步补上这条 gate 的
  说明。
  ④ **P1（工作流问题，非规格漏洞）：入口文档没有一句话能让人在读到一半时
  就确认"这一切都还没写成代码"。** 上一轮已经在 §0 写了"代码尚未开始写"，
  但整篇文档充满可直接复制的 Python 伪代码、`raise ValueError(...)` 片段、
  精确到字段名的 buffer 表，读者很容易在读了十几轮"这一轮修的""P0/P1"
  之后，把"规格文本的逻辑漏洞被反复修正"误读成"代码在跑、bug 在修"。核对
  `litgpt/`、`tests/` 确认零匹配 `log_kv_semantic_clusters`/`op_log`/
  `current_segment` 等任何本设计的字段（`LogKVStreamTrainingAttention` 类名
  确实存在，但那是语义簇设计之前、位置分桶时代的版本）。在 CLAUDE.md §0 和
  `algorithm-spec.md` 顶部/§5.1 参数表前都补了不容错过的"未实现"声明，附
  `grep` 自证命令，并明确"更正/这一轮修的"指的是规格文本自身的逻辑漏洞。
  ⑤ **P2：S0.8"分歧率 < 5%"是一个未定义统计口径的单一标量，拆成三项**：
  ①cluster assignment divergence（含 Ward 事件）——用已有的
  `scan_op_log`/`resolve_final_slots` 解析出每个 token 的 `(slot,epoch)`
  最终身份，因为槽号在两条路径下不保证对齐，改用成对共簇一致率
  （pairwise co-assignment agreement）而不是直接比较槽号，Ward 事件的
  合并候选对单独比较、不并进这个一致率里被平均掉；②segment/PAD
  overhead——`segment_count_ratio`/`pad_entry_ratio`，预期方向确定
  （`p_hi_c` 批内冻结只会让批量路径多开 segment，不会少开），报告"大多少"
  而非"有没有偏差"；③最终 cache/readout 差异——两条路径的 ladder 在按
  ①的身份对齐之后比较相对误差，或直接比较一次 attention 读出的相对 L2
  误差，这是唯一直接回答"近似值不值得用"的一项，**决策门只挂在这一项**
  （<5%）。①②是诊断信号，用于在③超标时决定修法：cluster assignment/Ward
  事件分歧主导时收紧 Phase 1 近似（如缩小 flush 粒度）；segment/PAD
  overhead 主导且①本身一致率高时按 §5.4"批量路由留下的一个未解决风险"
  那节的方向，给 `p_hi_c` 加按簇分组的前缀扫描，而不是缩小 flush 粒度
  （对这类分歧没有针对性，代价却是实打实的）；`γ` 更保守只在分歧确实由
  centroid 冻结导致漂移过大时对症，不是对所有分歧类型都有效的旋钮。
  `experiments.md` §6 的 S0.8 行与决策门、`algorithm-spec.md` §5.4 里
  引用 S0.8 的地方同步更新。

- **2026-08-14｜第十轮核实：解决"op_log 跨 Phase 顺序"这个此前一直没钉死的
  根本问题——主操作改成显式携带 `token_idx`，不再靠隐式位置对应原始 token；
  同时钉死 Ward 合并的物理执行顺序、拆开 `derive_final_cluster` 避免分段
  结果过期、把 Phase 1 的向量化公式显式限定在 direct 子序列上、并把训练期
  `op_log` 显存从"克隆导致翻倍"改成"重新分配、不翻倍"。** 动机：用户指出
  两处 P0——本质都是"批量化路由（Phase 1 向量化 + Phase 2 串行）产生的
  op_log，和 backward 重放/Phase 3 assumed 的东西不是同一个顺序"这一个更
  根本问题的不同表现——外加三处 P1。这是目前为止改动面最大的一轮，因为
  两处 P0 的修法（显式 `token_idx`）触及 op 格式、Phase 1/2 的全部伪代码、
  重放循环、以及 `derive_final_cluster`。逐条结论：
  ① **P0：`op_log` 的顺序契约自相矛盾**——§5.21-2 早就承认"op_log 的线性
  顺序是逻辑顺序，不是 forward 真实物理顺序"，但 backward 重放的
  `token_ptr` 是一个从 0 开始的隐式递增计数器，悄悄假设了"op_log 第 i 条
  就对应原始第 i 个 token"。具体反例：批次 `[direct τ0, orphan τ1,
  direct τ2]`，Phase 1（向量化）先写 `τ0`/`τ2` 的 op，Phase 2（串行）再写
  `τ1` 的 op——本地缓冲顺序是 `[τ0, τ2, τ1]`，不是 `[τ0, τ1, τ2]`，
  `token_ptr` 会把 `τ2` 的 `JOIN` 错误地喂上 `τ1` 的数据。**修法**：三类
  主操作的 `arg2`（此前恒为 `-1`，未使用）改存显式 `token_idx`，重放/
  `derive_final_cluster`/Phase 3 一律用 `op.token_idx` 索引原始
  `(k_raw,v,pos)`，不再依赖隐式位置。同时把顺序契约精确改写为三条（同簇
  内部按真实到达顺序、结构操作紧邻其主操作、跨簇顺序不重要），并证明
  "Phase 1 全体先写、Phase 2 全体后写"满足这三条——关键前提是"同一个逻辑
  簇的主操作在一个 flush 批内只可能整个来自 Phase 1 或整个来自 Phase 2，
  不会两边都有"（orphan 的语义距离恒 `>λ_new`，不可能 JOIN 一个 Phase 1
  也在写的既有簇；Ward 合并只改写 `keep_slot` 的聚合元数据，不产生新主
  操作，不会把 Phase 2 的东西混进 `keep_slot` 的主操作序列）。
  ② **P0：Ward 合并的物理执行顺序此前没有明确定义**——如果"Phase 1 只记
  日志、物理写入推迟到批末"，Ward 合并看不到 Phase 1 已分配的 token；如果
  "Phase 1 提前把全部 direct token 物理写入"，Ward 合并又会提前看到本该
  更晚到达的内容——两种读法都会让 forward 真实发生的事和 op_log 记录的
  顺序对不上。**修法**：钉死"Phase 1（路由决策 + 向量化物理写入）作为一个
  原子步骤完整跑完，Phase 2（串行，含 Ward 合并）才开始"这一条实现契约，
  不是"日志 vs 物理写入"的分界，是"Phase 1 整体 vs Phase 2 整体"的分界。
  这个顺序和①确定的 op_log 物理顺序完全对应，所以"严格按 op_log 顺序
  重放"自动等价于 forward 的真实执行，不需要另外证明。
  ③ **P1：`derive_final_cluster` 分段串联会让早期 token 的 final slot
  过期**——第 1 段扫完返回的 `final_slot` 一旦被调用方当作最终答案存起来，
  第 2 段里如果发生 `WARD_MERGE` 继续合并第 1 段某个 token 所在的槽，那个
  存起来的值就悄悄过期了，且没有任何机制通知调用方。**修法**：拆成
  `scan_op_log`（只累积 `token_identity`/`epoch`/`parent`，从不解析"最终
  归属"）和 `resolve_final_slots`（对累积下来的**全部** `token_identity`
  统一解析一次，只应该在扫完全部你关心的段之后调用一次）两个函数——
  中间任何一次 `scan_op_log` 的返回值都不包含"最终"这个概念，也就没有
  "过期"的可能性。
  ④ **P1：Phase 1 的向量化公式（segment id / PAD count / 本地缓冲索引
  分配）没有排除 orphan**——公式按全批 `c*[t]` 分组，但 Phase 1 只该处理
  `s*[t]≤λ_new` 的 direct token，orphan 的 `c*[t]` 是"离哪个既有簇最近"
  而非"它真的会去哪"，混进分组会污染真实 direct token 所在簇的统计。
  **修法**：显式定义 `direct[τ]=(s*[τ]≤λ_new)` 和压缩后的 `direct_idx`，
  下面所有公式的下标 `t` 改指压缩后 direct 子序列（`t=0..m_direct-1`），
  orphan 完全不出现在这些公式里，也不占本地缓冲的任何一行。
  ⑤ **P1：训练期 `op_log` 显存少算了一份 clone**——上一轮修的
  `.detach().clone()` 解决了别名 bug，但 cache 自己的持久 448MB 和 ctx 里
  的克隆同时存在，训练峰值实际是 896MB，`CLAUDE.md` 的"约 448MB"因此过时
  了。**没有选择接受翻倍**：重新审视后发现 `op_log`/`op_log_len` 是本表
  唯一"必须活过 `reset_parameters()`"的 buffer（其它 buffer 的内容 backward
  从不直接读，靠重放 `op_log` 重建，不需要活过自己所在的 forward() 调用）
  ——这个独有的需求，本就不该套用其它 buffer"预分配一次、原地清零复用"
  的模式。**修法**：`op_log`/`op_log_len` 改成每次 `forward()` 开头重新
  绑定成全新分配的张量（不是原地 `zero_()`），直接存进 `ctx`，不需要
  克隆——下一次 `reset_parameters()` 只会让属性名指向另一块全新存储，
  完全不触碰 `ctx` 里这次调用留下的对象。训练峰值显存回到单份 448MB，
  代价是分配频率变高，但 PyTorch 显存缓存分配器在稳态训练循环里能把这个
  代价压得很低，远小于峰值翻倍的代价。

- **2026-08-14｜第九轮核实：修掉 `ctx.save_for_backward` 存裸引用而非快照的
  bug、Phase 1 reset-scan 的 off-by-one、变长本地缓冲的索引分配公式、`K`
  未满/冷启动路径与 Ward 分支的初始化不一致，并澄清 `current_segment` 合并后
  的语义边界与更新 `glossary.md` 的漂移。** 动机：用户逐条核实上一轮的四处
  修复都已生效，同时指出一处 P0、四处 P1/P2 级的实现缺口。逐条结论：
  ① **P0：`ctx.save_for_backward(cache.op_log, cache.op_log_len)` 存的是
  裸引用，不是快照**——PyTorch 的 `save_for_backward` 不会自动 deep copy，
  而本项目一贯用预分配 buffer + `reset_parameters()` 原地 `zero_()` 复用
  （不是每次重新分配），下一次 `forward()` 的 reset 会直接清空 `ctx` 里
  "存"的那个引用指向的底层数据——`backward()` 读到的可能已经是被清空的
  `op_log`。虽然 PyTorch 的 saved-tensor 版本计数器可能会让这种误用在
  `backward()` 访问时报错，但这不是能依赖的安全网。**修法**：
  `ctx.save_for_backward` 时对 `cache.op_log`/`cache.op_log_len` 显式
  `.detach().clone()`；`q`/`k_raw`/`k_roped`/`v` 不需要同样处理，因为它们
  是每次 forward 新建的激活张量，不是会被 cache 对象在未来原地清零复用的
  buffer，不存在这个别名风险。
  ② **P1：Phase 1 的 PAD_INSERT reset-scan 有一处 off-by-one，且两个分支
  会以同一个方向同时出错**——`steps_since` 如果直接算成"组内下标减最近一次
  new_seg 的下标"（不减 1），对"已有上一次新段"和"本批还没开过新段"两个
  分支都会多算 1 步，因为 `new_seg=True` 那个 token 自己那一步已经被"计数器
  重置到 1"这条化简吸收掉了，不能再计入距离。补了从头展开递推
  `M_r = 1 + (r-1-r')` 的完整推导，钉死统一公式
  `steps_since[t] = rank[t] - last_new_rank_before[t] - 1`（哨兵 `-1`
  代表"本批还没开过新段"，代入后两个分支自动给出正确结果，不需要分别记两条
  公式），并强调这个 `-1` 是这条公式唯一容易做错、又不容易被单测覆盖到的
  地方（需要专门测"本批第一次开新段"这个边界）。
  ③ **P1：Phase 1 写入本地缓冲的变长展开缺一个索引分配公式**——`PAD_INSERT`
  必须紧邻且先于它服务的 `NEW_SEGMENT`，但不同 token 展开出的 op 数量不同
  （1 或 2 条），Phase 1 是向量化路径不能逐 token 决定行号。补了
  `extra[t]=new_seg[t] and count[t]>0`、
  `main_idx[t]=base+t+inclusive_prefix_sum(extra)[t]`、
  `pad_idx[t]=main_idx[t]-1` 的完整公式，并用一个三 token 的例子验证了
  "inclusive 前缀和"是必须的，用 exclusive 会导致两个 token 的 op 写进
  同一行、互相覆盖。
  ④ **P1：`K` 未满/冷启动的新簇初始化写得过于简略，容易和 Ward 分支的详细
  步骤脱节**——Ward 分支的 `alive`/元数据清零、`NEW_CLUSTER` 写入、
  `local_p_hi`/`local_segment` 初始化写得很细，但"K 未满"在 §5.6 的表里
  只有"占用第一个 false 槽位"一句，容易让实现者以为这条路径不需要同样的
  初始化。**修法**：把 Ward 分支步骤 4 开始的全部内容形式化为共享原语
  `allocate_new_cluster(slot_idx, group)`，三条路径（冷启动/K 未满/K 已满）
  只在"如何拿到一个空 `slot_idx`"这一步分叉，初始化逻辑完全共用；同时点出
  一条隐含前提——"K 未满"分支依赖"一个从未 `alive` 过的槽本来就是全零"，
  这必须由 `reset_parameters()` 的显式 `torch.zeros(...)` 保证，不能依赖
  `torch.empty` 之类不保证清零的分配，否则 `n_eff_pre==0` 这条判据会在
  垃圾值上失效。
  ⑤ **P2：`current_segment=max(a,b)` 的语义边界需要精确重新表述**——原表述
  "不会和调试/分析工具已经见过的旧 id 撞车"不准确：`max` 只保证合并后
  `keep_slot` 未来新开的 segment id 大于 `a`/`b` 双方的历史最大值，不保证
  `a`/`b` 各自独立计数（都从 0 开始）的历史 segment id 互不相同——但这天然
  没关系，因为 `op_log` 里两段历史各自带着原本的 `cluster` 字段（没有被
  重定向），本来就可以区分；真正需要小心的是"同一个物理槽先后被两个不同
  逻辑簇使用"这种槽位复用场景，这正是 `derive_final_cluster` 的
  `(slot, epoch)` 版本化身份已经解决过的问题，不是 segment 机制独有的新
  坑。结论：需要跨这类边界分组的分析工具必须按 `(slot, epoch, segment)`
  做 key，且跨 lineage 的时间先后顺序只能从 `op_log` 线性位置读，不能比较
  两个独立 lineage 的 segment 数值大小。
  ⑥ **P2：`glossary.md` 的"新增 buffer"列表漏了这两轮新加的
  `current_segment`/`op_log_len`**——已同步补上。

- **2026-08-14｜第八轮核实：补上 Phase 1 缺失的持久 segment 状态与批内向量化
  segment-id/PAD_INSERT 计算、`op_log` 的有效长度规格与 backward ctx 生命周期
  定案、`derive_final_cluster` 的切片安全性，并删掉一处从未有过定义、和已证明
  不变量矛盾的 Ward 合并 fallback。** 动机：用户对照最新远端逐条核实上一轮的
  四处修复都已生效，同时指出五处更底层的实现缺口——本质上都是"看起来定了，
  细究会发现某个边界情形没人接住"。逐条结论：
  ① **P0 级缺口：Phase 1 没有持久 `current_segment` 状态，且同批同簇多个
  token 独立开新段时会互相踩踏**——`JOIN`/`NEW_SEGMENT` 都要写具体的
  `segment` 整数，但 §5.13 只有 `p_hi_c`，从没有一个持久 buffer 记"这个簇
  现在 segment id 是多少"；上一轮为 Phase 2 orphan 组加的 `local_p_hi`/
  `local_segment` 只覆盖新建簇，覆盖不了 Phase 1 批量路由到既有簇的 token
  （恰恰是每批处理量最大的路径）。同时，若批内同簇多个 token 都判定"该开
  新段"，各自读批前持久值计算 `segment`/`PAD_INSERT count` 会给出重复或
  过期的结果——这和 Phase 2 那个 bug 同构，只是发生在向量化路径上，不能退回
  逐 token 串行去修。**修法**：新增持久 buffer `current_segment: (B,G,K_max)`
  （Phase 3 唯一写入点，`ward_merge_only` 合并时取 `max`，和 `p_hi_new` 同一
  模式）；批内 segment id 分配用一次 **segmented inclusive cumsum**
  （`op.segment[t] = current_segment[c*[t]] + 组内到 t 为止 new_seg 的包含性
  前缀和`，一个公式同时覆盖 JOIN 和 NEW_SEGMENT）；`PAD_INSERT` 的 count 用
  一次 **segmented reset-scan**（关键化简：每次新段事件后，level-0 的 mod
  计数器必然精确回到 `1 mod 2^ℓ_block`，与之前 pad 了多少无关，于是只需要
  "距离本簇上一次开新段过了几步"这一个无状态量，不需要像 §5.21-3 的 carry
  那样模拟计数器怎么折返）。两者都是标准向量化原语，不需要数据相关的有界
  循环轮数，要求补 CPU 参考实现对拍单测。
  ② **P1：`PAD_INSERT` 读 `level_count[cluster,0]` 的公式本身是对的（上一轮
  已修），但没说清楚"批内多次触发时读的是哪个 level_count"**——这个疑问随①
  一并解决：`level_count[c*[t],0] mod 2^ℓ_block` 只在"本批内这个簇还没开过
  新段"时作为 `base_mod` 使用，一旦本批内发生过一次新段事件，后续同簇的
  `count` 计算改用"距离那次事件几步"，不再依赖是否读到"批前"还是"批内更新
  过"的 `level_count`——化简后这个问题不需要维护一份额外的"本地 level_count"
  也能正确处理。
  ③ **P1：`op_log` 缺有效长度规格，replay 伪代码 `for op in op_log` 会扫到
  未写入的尾部行**——新增 `op_log_len: (B,G)`（§5.13）记录当前写到第几行，
  一切遍历 `op_log` 的代码（重放、`derive_final_cluster`、S0.8 对拍）必须先
  按它截断；本地缓冲同样需要 `local_op_len`，提交进持久 `op_log` 就是一次
  定长拷贝 + 两个指针相加，不需要逐条判断。**顺带把 `op_log` 和
  `reset_parameters()`/backward `ctx` 的生命周期从"两个选项都列出但没拍板"
  改成定案**：`op_log`/`op_log_len` 在 `forward()` 处理完整条序列后，和
  `q`/`k_raw`/`k_roped`/`v` 一起存进 `ctx`，`backward()` 只读这份快照，cache
  对象自己的 `op_log`/`op_log_len` 该被下一次 `forward()` 的
  `reset_parameters()` 清空就清空，不需要任何豁免逻辑——和已有 `v`/`k_raw`
  的处理方式完全对称，不给 `op_log` 发明第二套生命周期规则。
  ④ **P1：`derive_final_cluster` 只对"从冷启动开始的完整日志"安全，拿切片
  调用会让上一轮刚修的槽位复用 bug 以另一种方式复活**——`epoch.get(slot,0)`
  的默认值会把"切片开始前已经复用过的槽"重新当成第一次出现。**修法**：把
  `epoch`/`parent` 做成可选种子参数（`initial_epoch`/`initial_parent`），
  函数变成可以对同一条 `op_log` 分段串联调用的形式，返回值里带上本段结束时
  的 `epoch`/`parent` 状态供下一段调用直接传入；`op_log` 参数本身也要求是
  按 `op_log_len` 截断过的有效前缀，不是整个静态 buffer，和③统一。
  ⑤ **P2：`K` 已满时"合并失败必须有 fallback：强制并入最近簇"这句话和文档
  别处已经证明的不变量矛盾，必须删掉**——`K_max≥2` 时 Ward 候选池必然非空
  （§5.6 已证明），这个 fallback 对应的前提根本不成立，硬留着只会制造一个
  从未定义过语义（该写 JOIN 还是 NEW_SEGMENT？消费哪个 token？）的死代码分支。
  **改法**：从"新簇形成条件"表里删掉这句话，替换成"这一步不会失败，不需要
  fallback"的说明；实现层面该有的是一条 `assert alive[keep_slot] and
  alive[free_slot] and keep_slot != free_slot`，断言失败说明前面状态维护
  出了 bug，需要去修那个 bug，不是在这里兜底掩盖。连带清掉 `K_max=1` 分支
  里一处指向这条 fallback、且用了脆弱行号引用（"第 192 行"）的交叉引用。

- **2026-08-14｜第七轮核实：修掉离线派生工具的槽位复用 bug、`docs/position.md`
  里 PAD_INSERT 公式的回归、Phase 3 元数据写入权限的过强表述，并补上 orphan
  组内后续成员的局部时序状态。** 动机：用户对照最新远端 `semanticLogKV`
  （HEAD `01ce49e`，已含 PR #17 合并结果和一个独立提交新增的 `docs/position.md`）
  逐条核实上一轮的修复，指出一处 P0、三处需要立即澄清/修正的问题。逐条结论：
  ① **`derive_final_cluster` 仍有严重漏洞：槽位复用会被 union-find 错误合并**
  ——上一版直接对裸槽号做并查集，`WARD_MERGE(C,B)` 之后 `free_slot=B` 被后续
  `NEW_CLUSTER(B)` 复用建立一个全新、无关的簇时，裸槽号并查集会把这个新簇的
  token 也错误地路由到 C（具体反例：`token0` 进 B，`WARD_MERGE(C,B)`，`token1`
  又 `NEW_CLUSTER(B)`，离线 derive 会把 `token1` 也推到 C）。**这是 §5.4 反例
  在离线派生工具这一侧的镜像问题**——真正的执行路径（forward/replay）不会犯
  这个错，因为它们按时间顺序执行，槽的"当前身份"由最后一次写入决定；但这个
  离线函数跳过时间顺序、只扫 `WARD_MERGE` 边，裸槽号不足以区分"同一物理槽在
  不同时期代表的不同簇"。**修法**：引入版本化身份 `(slot, epoch)`——每次
  `NEW_CLUSTER(slot,...)`（首次建立或复用）都让该槽 `epoch` 前进一代，
  `WARD_MERGE` 只 union 当前这一代的身份，不触碰 `epoch` 本身；`epoch` 只是这个
  只读函数内部的临时簿记，不影响 forward/backward/replay 的任何状态。
  ② **`docs/position.md` 的 `PAD_INSERT` 公式写回了已经被证伪的版本**——位置
  手册的 P7.1 写着 `count = (-n_total_c) mod 2^ℓ_block`，这正是
  `algorithm-spec.md` §5.11 更正框已经用反例否定过的公式（`n_total_c` 不含 pad，
  第一次填充后就会和 level 0 真实插入流分叉）。已改成
  `count = (-level_count[cluster,0]) mod 2^ℓ_block`，并加一段简短说明加指向
  `algorithm-spec.md` 的权威引用，防止两份文档再次漂移。
  ③ **"Phase 3 是簇级元数据唯一写入点"是过强表述，和 `ward_merge_only` 冲突**
  ——`ward_merge_only` 步骤 1（合并簇级元数据）本来就会写
  `μ_new`/`n_eff_new`/`n_total_new`/`p_hi_new`，这是必须存在的第二个写入来源，
  不是需要消灭的冗余。**改成精确表述**：Phase 3 是本批"主操作 token 内容贡献"
  唯一的写入点；Ward 合并是"结构性合并事件"的写入点，处理的是两个既有簇已
  积累的历史值组合，不处理本批任何 token 的原始内容。两者写入范围不重叠
  （`keep_slot` 的历史值 vs. 本批新内容增量），顺序上 Ward 合并总在 Phase 3
  之前（发生在 Phase 2 内部），所以 Phase 3 读到的永远是"本批结构变动都已落地
  之后"的起点，不存在竞争。
  ④ **orphan 组内后续成员的 `JOIN`/`NEW_SEGMENT` 判据读的是刚被清零的
  `p_hi_c`，时序判据会错**——Phase 2 第 4 步把新簇的 `p_hi_c` 清零备用，但
  紧接着又说组内后续成员按"`slot_idx` 当前 `p_hi`"判断要不要开新段；若这个
  "当前 p_hi"指全局 buffer，读到的就是刚清零的占位值 0，`p_t − 0` 对任何有
  意义长度的文档都远超 `g_max`，组内除最早成员外全部被错误判成"时序打断"，
  各自被迫开新 segment——而它们本来就是同一次 mini DP-means 判定为彼此接近、
  到达时间上也大概率紧挨着的一组 token。**修法**：引入 Phase 2 私有的局部
  状态 `local_p_hi[slot_idx]`/`local_segment[slot_idx]`（不是新持久 buffer，
  不进 §4/§5.13 账目，Phase 3 也不需要读它——Phase 3 判断新段直接看 op 类型
  本身），由第 4 步用最早成员的位置初始化，组内后续成员读写这个局部状态而
  非全局 `p_hi_c`。同时说明这个 bug 为什么只出现在 Phase 2 的 orphan 组、
  不出现在 Phase 1：Phase 1 的"批前冻结 `p_hi_c`"是一个已知、已在 S0.8 测的
  近似（读到的始终是真实历史值，只是不够新），而 Phase 2 这里读到的是一个不
  代表任何历史的占位哨兵，是正确性 bug 不是近似误差，两者不能混为一谈。
  ⑤ **新簇的 segment 初始化一并定案**——`local_segment[slot_idx]` 从 0 开始
  （与 `NEW_CLUSTER` 的 `arg1=0`"新簇第一段"约定一致），后续成员按
  `p_t − local_p_hi[slot_idx] > g_max` 决定 `JOIN` 还是 `NEW_SEGMENT`（后者
  照常触发 §5.11 的 `PAD_INSERT`），处理完毕后 `local_p_hi` 无条件推进到
  `p_t`（不论是否开了新段）——这就是 `p_hi` 的既有定义"最近一次收到成员的
  位置"，随④一并解决，不再是未定义行为。

- **2026-08-14｜第六轮核实：推翻上一轮（第五轮）刚钉死的"批内重定向"机制本身
  ——它会把可执行的 `op_log` 改坏，改用"op_log 只追加、永不改写 + 需要时按需
  派生最终归属"。** 动机：用户指出上一轮把"可执行日志"和"最终归属索引"这两个
  不同的东西合成了同一个可变 `op_log`，这才是重定向反复出问题的根源，一拆开，
  重定向的风险自然消失。给出六条意见，两处 P0、三处 P1、一处 P2。逐条结论：
  ① **P0：重定向会把"可执行 op_log"改坏，用具体反例证实**——本批内 orphan
  组 1 先 `NEW_CLUSTER(B)`，orphan 组 2 因 `K_max` 满触发 Ward 合并，代价矩阵
  在"全部 alive、非对角线槽"里选中 `(keep=C, free=B)` 完全可能发生、甚至是
  大概率事件（B 刚建立、`n_total` 极小，Ward 尺寸加权代价对它天然最低，和
  §5.6"multi-needle 被漏斗到同一簇"是同一种偏好，只是这次发生在同一批内）。
  按上一轮的重定向规则，这会把 orphan 组 1 更早写下的 `NEW_CLUSTER(B,...)`
  该写成 `arg0=C`，留在原时序位置，产生两个独立的致命错误：（a）重定向只碰
  `arg0` 不碰 `arg1`，改写后的 `NEW_CLUSTER(C, 0, -1)` 仍带着"新簇第一段"的
  `arg1=0`，把一个刚到达的 token 记成运行了很久的既有簇 C 的"segment 0"，
  违反 segment 单调性；（b）`WARD_MERGE` 类型本身被排除在重定向目标之外，
  所以 `WARD_MERGE(C, B, -1)` 原样留在序列里，但 B 的建立事件已经被重定向
  改写走了——重放重建出的状态里槽 B 从未被 `alive` 过，`ward_merge_only(C, B)`
  要么撞断言崩溃，要么悄悄用垃圾内容"合并"进 C。**这不是能靠加字段、加特判
  堵住的漏洞**，只要 Ward 候选池不排除本批内新建的簇（而它必须不排除，否则
  §5.6"`K_max≥2` 时永远有候选"这条不变量就不成立），这类结构性矛盾会反复
  出现。**推翻重做**：完全不做重定向，`op_log`（本地缓冲和持久存储一样）
  任何已追加条目永不改写——这不需要放弃任何已证明成立的性质，本节此前就已
  论证过"结构操作和主操作共享同一条未经改写的时间线时，严格按 op_log 顺序
  重放本身就足以正确重建 forward 的 cache 结构"，上一轮误把这条已成立的性质
  当成"重放正确性还需要重定向来加强"，答案反了。
  ② **P0：`ward_merge_only` 是否写日志自相矛盾，修法是把候选选择和日志写入
  都移出这个 primitive**——§5.6 原定义"1-3+5 步"里的"5"是写 `WARD_MERGE`
  日志，但 §5.4 的 Phase 2 在调用完 `ward_merge_only` 后又显式 append 一次
  （双写），backward 重放调用它时注释却写着"只合并"（暗示不该写）。核实后
  两处用法一致指向同一个正确定义：**`ward_merge_only` 只负责纯状态 mutation
  （合并簇级元数据、合并两条 ladder、释放槽位，重编号为 1-2 步），Ward 代价
  矩阵候选选择在它之前由调用方完成，`WARD_MERGE` 日志写入在它之后由调用方
  完成**——重放时这个 primitive 只读日志决定的 `(keep_slot, free_slot)`，
  从不重新计算 Δ，也从不写日志，天然不会有双写或"重放时产生副作用写"的问题。
  ③ **P1：新簇元数据被 Phase 2 初始化和 Phase 3 更新双计数**——上一轮的 Phase 2
  "新簇写进去"那一步同时给 centroid/`n_eff`/`n_total`/`p_hi_c` 按 orphan 内容
  设初值，Phase 3 后又对整个本地缓冲统一更新一遍同一批 token，计入两次。**选
  第一种修法**：Phase 2 腾出槽位后只置 `alive=true`、数值元数据全部清零（白纸
  状态），Phase 3 是这些字段唯一的写入点，对 Phase 1 分配到的既有簇和 Phase 2
  新建的簇一视同仁，从本批完整的 op 序列统一构造。新建簇的第一个成员不需要
  特判——§5.5 公式里 `n_eff_pre==0 时 μ_c←k` 这条分支本就是为"冷启动或 γ=0
  归零"设计的，天然接得住"刚建立的簇"这个情形。
  ④ **P1：replay 正确性单测不该要求 centroid 逐位一致**——§5.21-2 明确"重放
  完全不需要 centroid"，但上一轮的单测又要求 replay 产出的"ladder/centroid"
  状态与批量前向逐位一致，自相矛盾。**拆成三条独立断言**：重放正确性（S0.1，
  只覆盖 attention 需要的 ladder 张量、`level_count`/`alive`/`pad_mask`，
  明确排除 centroid 类元数据）、批量近似质量（S0.8，允许分歧率）、Phase 3
  元数据更新正确性（S0.1，与重放正确性完全独立的一条断言，专门抓"双计数"
  这类 bug，断言新建簇的最终 `n_total` 精确等于它实际收到的 token 数）。
  ⑤ **P1：`NEW_CLUSTER` 被列为重定向目标尤其危险**——它比 `JOIN`/`NEW_SEGMENT`
  更严重，因为它的语义依赖目标槽在那一刻是 free 的，重定向后这个前提不再成立
  （见①的反例 b）。随①的推翻一并解决：不存在"重定向目标"这个概念了，`op_log`
  里没有任何字段会在写入后被后续操作改写，六类 op 一视同仁。
  ⑥ **P2：本地缓冲容量 `4×flush_granularity` 对 debug `CARRY` 最坏 burst 缺
  断言**——生产默认不写 `CARRY` 时这条余量足够，但 debug/对拍 build 打开
  `CARRY` 后，§5.21-2 的 `OP_max` 推导是对整条序列的摊还论证（O(T/B′)），不
  等于对单个 flush 批内的最坏 burst 也成立。补上：追加前必须显式检查是否会
  超过 `local_op_cap`，超出则硬失败，不做动态扩容、不做静默截断——沿用项目
  一贯的"矩形预分配 + 硬失败"原则，避免对拍场景里本地缓冲静默溢出/截断产出
  一份看似完整、实则丢了尾部结构操作的日志。

  **顺带新增的机制**：重定向想买的"`op_log` 对每个 token 的最终归属自
  解释"这个真实需求，改用一个只读、离线的 `derive_final_cluster` 函数按需
  从 `op_log` 派生（对 `WARD_MERGE` 序列做一次并查集），不新增任何持久
  buffer、不改变 §4/§5.13 的内存账目，forward/backward 的正确性路径永远
  不依赖它、也不会被它影响——S0.8 对拍、调试工具需要"token 最终去了哪"时
  调用它就够了。

  **总体教训**：上一轮把"可执行日志"（forward/backward 正确性依赖的东西）
  和"最终归属索引"（调试/分析工具想要的东西）合成了同一个可变 `op_log`，
  这个表示层混淆是本轮几乎所有问题的共同根源——一旦拆开成"不可变执行日志 +
  按需只读派生"，重定向本身连同它带来的一整类边界 bug 就不再需要存在。

- **2026-08-14｜第五轮核实：把批内重定向从"原则"钉成"可实现的规格"——区分
  重定向该改哪些 op 字段、`WARD_MERGE` 何时落地、本地缓冲的真实容量、同一新簇
  多个 orphan 的 op 序列、合并 primitive 拆分、Phase 3 该读哪份数据、单测该测
  哪个性质，外加训练梯度文档里一处容易踩的表述。** 动机：用户对上一轮"批内
  重定向"设计再做一遍逐条核实，指出两处 P0（会让实现者写出结构错误或语义模糊
  的版本）、五处 P1、一处 P2。逐条结论：
  ① **P0：重定向笼统写成"改 `op.cluster`"，但六类 op 字段异构**——`WARD_MERGE`
  的 `arg0/arg1` 是 `keep_slot/free_slot`，是历史记录而非"簇身份"，重定向如果
  也去改它就是在篡改已发生的合并事实。补一张类型表，明确重定向只碰
  `NEW_CLUSTER`/`NEW_SEGMENT`/`JOIN`/`PAD_INSERT`/`CARRY` 这五类的 `arg0`，
  `WARD_MERGE` 排除在外；改写用 `is_target_type & (arg0==free_slot)` 这样的
  类型感知掩码，不是无条件的 `where`。
  ② **P0：`WARD_MERGE` 该在本地缓冲的哪个位置落地，新伪代码没写**——补上：物理
  合并（`ward_merge_only`）执行后立即 `append WARD_MERGE`，再做重定向，最后才
  `append NEW_CLUSTER`——`WARD_MERGE` 落在"紧邻 `NEW_CLUSTER` 之前"这个既有
  顺序要求里，且因为①的类型排除，它自己不会被同一次重定向误伤。
  ③ **P1：本地缓冲容量按"batch token 数"分配是错的**——缓冲最终要装下会提交进
  持久 `op_log` 的完整内容，必须像全局 `OP_max` 一样按"每 token 最坏 4 条 op"
  分配：`local_op_cap = 4 × flush_granularity`，不是 `≤128`。
  ④ **P1："多个 orphan 共享一个 `NEW_CLUSTER`"和"每 token 恰好一个主操作"直接
  冲突**——mini DP-means 分进同一临时簇的 orphan 里，只有到达顺序最早的那个
  产生 `NEW_CLUSTER`，其余对同一 `slot_idx` 产生 `JOIN`/`NEW_SEGMENT`，和"一个
  已存在的簇收到新成员"是同一套逻辑。
  ⑤ **P1：`merge_cluster_ladders` 到底是"只合并"还是"合并并写新簇"，两处伪
  代码用法不一致**——把 §5.6 步骤 4（"释放槽位，新簇写进去"）拆成两句：
  `ward_merge_only`（步骤 1-3+5，只合并、只释放）是一个 primitive，"新簇写
  进去"是调用方（无论是单 orphan 场景还是批量 Phase 2）自己的职责，不属于这个
  primitive；连带把 replay 伪代码里 `WARD_MERGE` 分支调用的函数名同步改过来。
  ⑥ **P1：重定向会改变本批部分 token 的最终归属，但 Phase 3 该用哪份数据更新
  centroid/`n_eff`/`n_total`/`p_hi_c` 没写清楚**——补硬规则：只能读"重定向执行
  完毕、提交进持久 `op_log` 之前"的最终本地缓冲，不能读 Phase 1 原始输出，否则
  会把 token 计入它最终并不属于的簇。
  ⑦ **P1：把"批量近似 vs 严格串行路由"的分歧率测试和"批量前向 vs 重放"的逐位
  一致性测试混成一条**——两者拆开：前者是 S0.8 已经在测的近似质量问题（Phase 1
  冻结 centroid/`p_hi_c` 本来就承认是近似），不要求逐位一致；后者才是需要逐位
  精确的正确性契约（"`op_log` 忠实记录了批量前向做了什么"），Ward 合并撞见
  Phase 1 已分配槽位只是作为 S0.8 分歧率统计必须覆盖的一类特殊输入，不该单独
  拔高成正确性要求。
  ⑧ **P2：训练梯度那节说"`k_raw` 传进来 detach 与否都无所谓"容易被误读成
  "可以在算 `k_roped` 之前把上游 qk-norm 输出整体 detach"**——补充区分：这句
  话只针对"这个 Function 的 `k_raw` 输入参数本身"，`k_roped` 必须从未经任何
  detach 的 qk-norm 输出算出，否则 `k_roped` 的梯度链会被一并切断——"`k_raw`
  不产生梯度"完全由这个 Function 的 `backward()` 恒返回 `None` 保证，不需要、
  也不应该在调用方源头做任何 detach。

- **2026-08-14｜第四轮核实：推翻上一轮的"屏蔽+顺延"设计（会让 token 在
  attention 里凭空消失），改用批内向量化重定向；纠正训练梯度目标与现有
  stop-gradient 前提的直接冲突；补 Ward 合并后 `level_count` 的硬性不变量；
  扣上虚拟槽展开与 `causal_tail` 的 API；三处 P2 级残留说法同步更新。** 动机：
  用户逐条指出两处 P0（会让实现出现"看似能跑但语义已经变了"的版本）、三处 P1、
  三处 P2。逐条结论：
  ① **P0：上一轮"屏蔽 touched 槽 + 无候选时顺延到下一批"的修法本身有更严重的
  问题**：语义簇路径的 flush 复用现有 recent window"溢出即驱逐"的语义
  （`log_kv_cache.py:1169-1188` `add_recent`），token 一旦被挤出 recent
  window 就没有"留着等下一批"这个中间态可用。"顺延"发明了一个现有架构里不
  存在的第三态（既不在 recent window、也不在 ladder cache），对 attention
  可不可见、算不算 recent、进不进 `op_log`、连续顺延时因果顺序怎么保证，一个
  都答不上。**推翻重做**：让 Ward 合并保持完全不受限（不再有 `touched_mask`），
  在合并发生的那一刻，对"本批目前为止已经写入的本地 op 缓冲"做一次有界、
  向量化的重定向改写（`torch.where(ops.cluster==free_slot, keep_slot,
  ops.cluster)`，只碰本批本地缓冲，不碰跨批持久 `op_log`），新建的
  `NEW_CLUSTER` 写在重定向**之后**，天然不会被误伤。副作用是"顺延"这个概念
  整个消失——`K_max≥2` 时 Ward 永远有候选（§5.6 已证明的性质），每个 orphan
  都能在自己所在的批次内拿到真实归宿，不再需要面对 recent window 的语义空洞。
  上一轮拒绝"回填修补"是因为把它想象成动态回溯，这次意识到它可以是一次有界
  向量化操作，判断错了，此处更正。
  ② **P0：训练 autograd 的 `grad_k_raw`（通过 `op_log` 重放对 `compact()` 求
  解析梯度）和现有训练目标直接矛盾**：`LogKVStreamTrainingAttention` 的
  docstring 明确写着"gradient reaches q/k/v only through each token's own
  block (the cache commits detached copies)"——这是整个 Function 能把训练
  内存从 `O(T·S)` 压到 `O(T+train_block·S)` 的数学基础，不是可以顺手改掉的
  实现细节。上一轮把"接口怎么传参"和"训练目标要不要变"混成一个问题，答案
  错了。**推翻重做**：v1 不改训练目标，`k_raw` 和 cache 写入这条路完全不参与
  反向传播，和现状代码里 `v` 的处理方式完全对称（`backward()` 对 `k_raw` 的
  梯度槽位恒返回 `None`）；`k_roped` 继续扮演现状代码里 `k` 的角色，梯度计算
  一字不改。`op_log` 存在的理由回到最初就讲清楚的那条——只是为了让重放能正确
  重建"某一时刻 cache 的（依然完全 detached 的）内容"，不支持任何新梯度路径。
  "cache 写入也可微"列为明确不在 v1 范围内的独立方向，需要先证明计算图有界、
  配 naive reference 和 backward 数值对拍。
  ③ **P1：`touched_mask` 只解决 Ward 候选冲突，没说同批多个 orphan 能不能
  互相 join**：随①的重做一并解决——§5.4 Phase 2"在这批 orphan 内部跑一个小
  DP-means"这个既有设计本就在分配槽位**之前**把互相接近的 orphan 分进同一个
  临时簇，不存在"orphan 2 处理完才发现该并入 orphan 1 刚建的簇"这种事后修正
  的情形。
  ④ **P1：Ward 合并后"`level_count` 天然自修复"只是解释里的一句话，没有硬性
  不变量和测试**：补上——`merge_cluster_ladders` 第 3 步结束时，每一层的
  `level_count` 必须由构造过程直接写出（尤其是"总数 ≤ B′ 直接放入"这个不触发
  compact 的分支，容易被误认为不需要更新计数），不能沿用任一侧合并前的旧值；
  新增单测要求既断言数值对，也要断言"用这个数值算出的下一次 `PAD_INSERT`
  确实让 level 0 流对齐"，不只测计数器本身。
  ⑤ **P1：虚拟槽展开的 `slot_valid` 掩码和现有 `causal_tail` API 没扣上**：
  `mask` 和 `causal_tail` 现有互斥，且 `causal_tail` 假设"最后 `causal_tail`
  个 slot 是逐 token 对齐的 in-flight chunk、之前的 slot 无条件可见"，完全没有
  "pooled 区域某些槽无效"这个概念。新增第三个、与另外两者正交的
  `slot_valid: (B,G,S_pooled)` 参数，只覆盖 pooled 前缀、可以和 `causal_tail`
  同时使用；钉死 flatten 顺序（pooled 在前、exact in-flight 在后，沿用
  `append_exact_tokens` 现有约定），in-flight chunk 从不参与锚点展开。
  ⑥ **P2：Stage 2 Config B（纯语义参考点）仍写 `K=16, g_max=∞`，没设
  `η=0`**：按 §5.3 已经钉死的"纯语义 = `η=0` 且 `(g_max=∞` 或 `ℓ_block=0)`"
  精确定义补上 `η=0`，否则这一档实际跑出来的不是真正的纯语义端点。
  ⑦ **P2：glossary.md 仍把 Config A 说成"`K_max=1` 正确性闸门"**，与
  experiments.md 早就把 A/B/C 三档降级为 eval-time 消融参考点（不设通过/
  失败容差）矛盾，同步改口。
  ⑧ **P2：`K_eff→c` 的决策规则说明有重复且过时的措辞**（"拟合斜率怎么换算成
  `c`"），合并成一条，统一成"观测区间上界 + `K_max` 绑定率校正"。

- **2026-08-14｜第三轮代码核实复查：修掉 `PAD_INSERT` 对齐公式在第一次填充后
  失准的 bug，闭合批量路由撞见 Ward 合并的槽位冲突，修好 `NEW_CLUSTER` 与重放
  伪代码的字段不一致，收紧"纯语义聚类"端点的定义，钉死训练 autograd 的
  `k_raw`/`k_roped` 双输入契约，堵上 mass bias 的 0/0 隐患，加固 `c` 的取值
  规则，改口"退化到今天的 LogKV"为"退化到同一类结构"。** 动机：用户对上一轮
  修完的文档再做一遍逐条核实，指出八处主要纰漏。逐条结论：
  ① **`PAD_INSERT` 的 `count = (-n_total_c) mod 2^ℓ_block` 在第一次填充后就
  不对**：`n_total_c` 只数真实 token、不含已插入的 pad，而对齐要保护的是"真实
  + pad 混合而成的 level 0 逻辑插入流"，两个量从第一次填充起就分叉，用具体反例
  （`ℓ_block=1`，段 A 长 3 插 1 个 pad，段 B 长 1 后公式误判"不需要再填充"）
  验证。**这个 bug 被两个人独立发现**——远端 `44826b0` 提交（"Clarify segment
  padding entry invariants"）同时到达，指出不能复用 `n_total_c`、建议改用
  `pad_mask`/`level_count`；核实后确认 `level_count[cluster,0]`（现有 buffer，
  簇 level 0 的当前占用数）单独就够，不需要再引入新计数器——因为只要 `B′` 是
  `2^ℓ_block` 的倍数（默认组合天然满足），"当前占用数 mod 2^ℓ_block"和"累积
  逻辑插入数 mod 2^ℓ_block"全程相等，且这个量在 Ward 合并后自动保持正确（合并
  过程本就会重写 `level_count`），不需要额外的合并时手工同步。补一条新校验：
  `B′` 必须是 `2^ℓ_block` 的倍数，和 `ℓ_block∈{0,1,2}` 那条放一起。`n_total_c`
  继续只服务 Ward 代价和 §5.8 空间界不变——"一个计数器身兼数职导致下游算错"这类
  问题第二次出现（第一次是 `n_eff`/`n_total` 拆分），但这次的修法不是拆出第三
  个计数器，而是发现已有的 `level_count` 就能兼任，模式相同（先问计数器服务
  几个下游，语义不一致就拆或复用），结论比最初判断的更省。
  ② **批量路由（§5.4）里 Phase 2 的 Ward 合并会让 Phase 1 已经写进 op_log 的
  槽位引用失效**：Phase 1 用批前快照把大多数 token 批量分配到某些槽，Phase 2
  处理 orphan 时若触发 Ward 合并，可能释放/复用一个 Phase 1 本批已经写过的槽，
  导致同一个槽号在同一批 op_log 里前后指向不同的簇，重放时产生歧义。决定用
  "屏蔽+顺延"而非"合并后回填"或"推迟到批末"：给 Ward 代价矩阵新增一条
  "本批已被 Phase 1/Phase 2 触碰过的槽不可选"的掩码（复用已有的对角线/dead
  槽位屏蔽机制），若因此没有可用候选，该 orphan 顺延到下一个 flush 批的最前面
  （不丢失、不错误合并），代价有界（顺延数 ≤ `K_max`），不引入新的无界风险。
  ③ **`NEW_CLUSTER(slot_idx, -1, -1)` 和重放伪代码的通用调用
  `append_to_ladder(..., op.cluster, op.segment)` 对不上**：重放对三类主操作
  一视同仁地读 `op.segment`，`NEW_CLUSTER` 的 arg1=-1 会让新簇第一个 token 带
  上不存在的段号。改成 `NEW_CLUSTER(slot_idx, 0, -1)`——新簇第一段天然是段 0，
  选择改数据格式而不是给重放循环加特判分支：字段自洽比处理逻辑自洽更简单。
  ④ **"`g_max→∞` 或 `ℓ_block=0` 就是纯语义聚类"这个说法不精确**：只要
  `η>0`，统一代价 `d_c` 的 argmin 选出的 `c*` 仍可能不是纯语义最近簇，novelty
  判据是在 `c*` 上判的，所以即使不分段，`η` 仍能通过 tie-break 影响新簇/并入
  的判定。把"纯语义聚类"端点精确定义为 `η=0 且 (g_max=∞ 或 ℓ_block=0)`，S0.0
  扫 `(g_max, ℓ_block)` 时固定 `η=0`，理由是 S0.0 存在的意义就是干净地分离
  "语义分组的贡献"与"分段边界的贡献"，端点混入未受控的时序因素会让问题本身
  不适定。
  ⑤ **训练 autograd 的 `k_raw`/`k_roped` 双输入契约此前只有"要同时持有"这句话，
  没有 `Function` 签名、`ctx.save_for_backward` 存什么、要不要 detach、梯度怎么
  汇合这四个问题的答案**：核对现有 `LogKVStreamTrainingAttention`
  （`log_kv_cache.py:1838`）后给出具体改法——`forward`/`backward` 都从三个
  张量（`q,k,v`）改成四个（`q,k_raw,k_roped,v`）；**两者都不能 detach**，因为
  它们是同一个 qk-norm 输出的两个下游消费者而非独立叶子，只要都保持
  `requires_grad`，PyTorch 常规 autograd 会在这个 Function 外部把
  `grad_k_raw`（直接）和"`grad_k_roped` 经 `apply_rope.backward` 拉回来的
  梯度"在共享祖先处自动相加，不需要这个 Function 自己合并；`grad_k_roped` 是
  现状代码里 `grad_k` 的平移（in-flight exact attention 数学不变），
  `grad_k_raw` 是新增的一路（通过 `op_log` 重放链路对 `compact()`
  的加权统计求解析梯度）。
  ⑥ **虚拟槽 `M=0`（无效 entry）会让 `log(w/M)` 算出 `0/0`**：现有代码靠
  "`slot_w >= 1 by construction`"这条不变量加 `if lam != 0.0` 门控防 NaN，
  语义簇路径的无效 entry 会打破这条不变量。修法是让 `dedup_anchors` 在返回
  `M` 之前就 `clamp_min(1)`（和 `mid_anchor` 已经在用的 `ww=w.clamp_min(1)`
  同一个模式），使无效 entry 的 `w/M=0/1=0`，`log(0)=-inf`（IEEE754 良定义，
  不是 NaN），`λ·(-inf)` 在 `λ≠0` 分支里也良定义，现有"先加 bias、后 mask"的
  顺序完全不用改。
  ⑦ **`K_max_default` 的 `c` 只看拟合斜率会在截距不可忽略时系统性低估
  32k/128k 的需求**（渐近正确不等于已测区间正确）：改用观测区间上界法
  `c = max_n ⌈(K_eff(n)+margin)/log₂n⌉`，直接对已验证区间的每个点取"至少
  需要多大的 `c`"再取最大值，不依赖"外推到 1M 曲线形状不变"这个额外假设。
  同时钉死 S0.2 要测两个不同口径的 `K_eff`（纯 DP-means 喂曲线形状判定，生产
  三路路由喂 `c` 的取值），此前只说了前者。
  ⑧ **"K 撞满后退化到今天的 LogKV 行为"是过强表述**：`K_max=1` 也不会复现
  旧数字（level 0 单 token、真实锚点位置表示都是不可选的设计改动，§5.6 的
  `K_max=1` 讨论已经验证过），CLAUDE.md §4 和 §5.6 里两处这个说法统一改成
  "退化到同一类结构（少数大簇 + 位置序 ladder），不是数值上复现现有 LogKV"。

- **2026-08-14｜第二轮代码核实复查：修掉 `n_eff` 公式的除零/off-by-one，闭合
  `K_max=1` 退化路径，补齐批量前向与串行重放的等价契约，钉死 `PAD_INSERT`/
  `CARRY` 字段语义与锚点去重的矩形/掩码契约，统一 `s_h` 的 KV-group 口径，补上
  `K_max` 的 `c` 决策规则和第三笔内存账。** 动机：用户对上一轮修完的文档再做一遍
  逐条核实，指出十处主要纰漏和三处较小的建议。逐条结论：
  ① **`n_eff` 更新公式字面实现会 off-by-one 且在 `γ=0` 时除零**：原写法"先用旧
  `n_eff` 算 μ、再 `n_eff += 1`"除数错了一位（该用新计数），且 `γ=0` 时新段第一个
  成员会除以 0。改写成显式的 `n_eff_pre`/`n_eff_new` 两步，`n_eff_pre==0` 时直接
  `μ ← k`，同时精确对应"`γ=0` 直接替换"这个此前只在注释里提过、从未在公式里兑现
  的语义。
  ② **`K_max=1` 时 Ward 合并结构性无定义**（`(K,K)` 屏蔽对角线后是空矩阵），但
  实验又把它当消融基线用。明确 `K_max=1` 时彻底跳过语义新簇判定，novelty 距离
  只记录不触发新簇，token 只能是 JOIN 或 NEW_SEGMENT——退化成"只保留时序分段"，
  和 Stage 2 Config A 的意图对齐，CPU 参考实现与生产路径共享这条分支，防止两者
  在这个边界上分叉。
  ③ **批量前向写入和 `op_log` 串行重放的等价性从未被要求验证**：forward 是
  §5.4 批量路由 + §5.21-3 向量化 carry，不是逐 token 循环；`op_log` 记的是"如果
  串行执行会产生的逻辑顺序"。补上这个等价契约成立的理由（`compact` 的正确性只
  依赖同簇成员的相对到达顺序，不依赖处理粒度）、对实现的具体要求（批量 scatter
  必须保序，批量 carry 必须等价于逐一单项 carry `K` 次），以及一条新增的必须
  单测（批量路径 vs 串行重放，逐位断言整条统计元组相等）。
  ④ **`CARRY` 日志字段不足以支撑断言，且从未说清楚它是不是权威日志**：明确
  `CARRY` 非权威、不参与 backward 正确性，生产 `op_log` 默认不写；调试/对拍 build
  才需要它，此时第三个字段填 `resulting_count`（这次进位后该层存活 entry 数），
  重放侧据此断言，而不是像原来那样是个不携带信息的占位符。
  ⑤ **`PAD_INSERT(cluster, level, count)` 的字段语义没有闭式定义**：钉死
  `level` 恒为 0（填充和普通 entry 一样只在 level 0 插入，靠自然级联对齐，不支持
  直接插高层）、`count = (-n_total_c) mod 2^ℓ_block`（标准的对齐到 2 的幂边界
  公式，且与已有的"每边界浪费 ≤ 2^ℓ_block−1"这行代价表用的是同一个量，不是巧合），
  `count=0` 时不产生这条 op。
  ⑥ **`dedup_anchors` 只有函数签名和一句 docstring，没有 GPU 需要的矩形/掩码
  契约，也没定义 dead/pad entry 的锚点该怎么处理**：改成返回固定 `(...,S,3)` 的
  矩形 `anchors` + 同形状的 `slot_valid` 掩码 + `M`，并规定顺序——先按 `w>0` 把
  整个无效 entry 的三个槽全部标记 invalid 且锚点覆写成安全哨兵 `0`，再在剩下的
  有效 entry 内部去重；顺序不能反，否则会拿 `p_lo=+INT_MAX,p_hi=-1` 这类合并期
  哨兵去参与去重比较或喂给 `_rotate_at_anchors` 索引 `cos_cache`——后者在
  `mid_anchor` 的 `clamp(mid, INT_MAX, -1)` 上会静默算出 `-1`，而负索引会静默
  环绕到最后一个位置而不是报错，是比越界崩溃更危险的一类 bug。
  ⑦ **`_rotate_at_anchors` 只实现了 split-half 布局，没覆盖 `rope_interleave=True`
  用的偶奇分组布局**：两者数据排布不同，不是同一函数换参数。Qwen3-1.7B 不需要
  interleave，但仓库其它配置会用到；构造时对 `log_kv_semantic_clusters=True` 加
  `rope_interleave` 的硬校验，泛化留作后续工作。
  ⑧ **`s_h` 的粒度写成"per (layer, head)"，和 §5.13 buffer 表的 `(n_layer, G)`
  形状对不上**：聚类只在 k 空间做，一个 KV group 只有一份 k，`s_h` 的物理粒度
  本就该是 KV group 而不是 query head；统一改口径，并补上 `k̄` 的精确定义——整个
  标定集上的全局均值，不是逐 prompt 或运行均值。
  ⑨ **`K_max_default` 公式里的 `c` 从测出 `K_eff` 到怎么定数值之间没有规则**：
  新增三步决策规则——对数形状成立时用拟合斜率换算 `c`；新增"`K_max` 绑定率"这个
  Stage 1/2 运行时指标作为调 `c` 的直接依据；加大 `c` 也压不下绑定率时应转向重新
  评估曲线形状假设本身（而不是无限调参），逼近 vanilla 预算时应如实报告方案对该
  工作负载退化，而不是继续加大内存把退化伪装成改进。
  ⑩ **CLAUDE.md §4 的"两笔账"漏了训练期 `op_log` 的 448MB**：补第三笔账，明确
  它只属于训练侧（backward 重放用），serving 不分配、不进 memory-matched 对比，
  不能和另外两笔账混进同一张表比较。

  三处较小的点顺带修掉：Stage 0"几乎不花 GPU"的说法改成"不碰训练、不落地 full
  attention"，并说明机制 B 本身的分块手算和一次真实 full attention 是同一数量级
  的浮点运算，不是零成本；新增 `log_kv_semantic_clusters=True` 必须拒绝
  `MultiheadLatentAttention`（构造时 `isinstance` 校验），因为本节通篇假设的
  `CausalSelfAttention` 式显式 per-KV-group `k_raw` 布局在 MLA 的低秩 latent 表示
  下根本不存在对应输入；entry 存储表现在直接标出 `w` 必须是 fp32/int32、不能沿用
  `k̄_raw`/`v̄` 的 activation dtype，不用去翻改动对照表才知道。

- **2026-08-14｜代码核实驱动的复查：修掉五处会静默产生错误结构/分数的实现级 bug，
  补三处规格空白。** 动机：用户对照 HEAD `8b1ec8a` 的实际代码逐条给出 file:line
  级证据，指出文档里"看起来定了"的几处实际经不起代码核实。逐条给出结论：
  ① **`PAD_INSERT` 记录顺序原来是反的**：§5.11 的例子要求填充插在新 segment 第一个
  token **之前**（`a3, ∅, b1`），但 §5.21-2 原文把它记在服务的 `NEW_SEGMENT` 之后，
  replay 会先 append 新 token 再补填充，实际产出 `a3, b1, ∅`，对齐填充完全失效。
  改成 `PAD_INSERT` 先于它服务的 `NEW_SEGMENT`，并把 `pos[token_ptr]` 显式加进
  replay 的 `append_to_ladder` 调用——锚点是绝对位置的函数，重放没有 pos 无法定义。
  ② **`n_c` 被同时用作 centroid 混合权重和 Ward 簇大小，语义冲突**：`γ` 衰减过的
  计数拿去算 Ward 代价，会让历史长、内容多但衰减过的簇被误判成小簇、廉价合并。
  拆成 `n_eff`（浮点，`γ` 衰减，只喂 §5.5 centroid 更新）和 `n_total`（整数，单调
  不减，Ward 代价 §5.6 与 §5.8 的 O(log n) 空间界都用它）。
  ③ **Σ/Γ 挪到 pre-RoPE 空间后，读出侧从没跟着改**：`log_kv_slot_attention` 现有
  公式用 post-RoPE `q` 点乘 `sigma_u`/`gamma_a`，若这两个方向仍存 pre-RoPE 空间，
  点积就是在错坐标系里算。把 `_rotate_at_anchors` 从只服务 `k_raw` 泛化成
  `materialize_anchor_directions`，`sigma_u`/`gamma_a` 和 `k_raw` 走同一套按锚点
  RoPE 物化；`gamma_b`（value 方向）和 `gamma`（标量）不需要，因为 value 从不
  被 RoPE。新增两条单测（锚点物化嵌套精确性、坐标系回归）。
  ④ **Phase 1 的批量快速路径和 §5.3 的串行判定不等价**：原文用全局
  `min_c ‖k−μ_c‖² ≤ λ_new` 做批量直接分配的判据，但 §5.3 的真实规则是先用统一
  代价（含 η 时序项）选出 `c*`，再检查 `c*` 自己的语义距离——当语义最近簇和统一
  代价赢家不是同一个簇时两者会给出不同答案。改成同一遍里把 `D`（统一代价）和
  `S`（纯语义距离）都算出来，`c* = argmin D`，用 `gather(S, c*)` 判定，额外开销
  可忽略。
  ⑤ **decode pending 的推广公式写反了**：文档说要留
  `flush_granularity − (T mod flush_granularity)` 个，但这是"还差多少补满一批"，
  和真正该留的残留数（`T mod flush_granularity`）互补而非相等；`T mod g == 0` 时
  错误公式给出 `g`（整批都标记成待补），正确公式给出 `0`。已改并在
  `risks-and-open-questions.md` 加更正框。

  另外三处规格空白：
  ⑥ **`OP_max = 4·T_max` 隐含了未声明的前提**：这个上界只有在 `ℓ_block ≤ 2`、
  `PAD_INSERT` 按 `count` 字段聚合（不是每个填充槽一条独立 op）、`PAD_INSERT`
  顺序正确（见①）三条同时成立时才安全；`ℓ_block` 若能取到 Stage 0 探索范围
  `L_alloc`（32k 下 11~15），`count` 会超过单层容量 `B′`，聚合假设失效，指数项会
  压垮上界。补上生产路径构造时的硬校验 `ℓ_block ∈ {0,1,2}`（不满足直接
  `raise ValueError`），并注明 Stage 0 的 S0.0 扫描是离线 CPU 模拟、不经过
  `op_log`/真实 cache，不受这条约束。
  ⑦ **`K_eff` 的两种口径混在一起会自相矛盾**：S0.2 说 32k 下 `K_eff` 应在 10² 量级，
  但默认 `K_max` 在 32k 只有 15，若不说清楚测的是哪个量，S0.2 的结果和 §4 的
  2344-entry 内存故事看起来互相打脸。明确 S0.2 测的是 **unclipped DP-means**
  `K_eff`（离线分析口径，回答"内容本身有多少语义多样性"），和生产路径里被
  `K_max` 截断后的实际簇数是两个不同的量，前者大不直接推翻后者的内存预算——
  它是路由质量信号，不是内存溢出信号。
  ⑧ **Stage 0 机制 B 的手算分数没有复现真实 attention 的三个步骤**：`model.py`
  确认真实计算用的是拼接完整通道后的 post-RoPE `q`/`k`（`816-817`，
  `q_roped`/`k_roped` 只是被旋转的位置子通道切片，Qwen3-1.7B 因为
  `rotary_percentage=1.0` 恰好两者相等，这是这个模型特例而非通用结论）、GQA 在
  算分数前把 `k` 按 `q_per_kv=2` `repeat_interleave` 展开到查询头数（`857-860`）、
  以及可选的 softcapping（`1454-1456`，Qwen3-1.7B 不启用）。原文的伪代码
  `q_roped @ k_roped.mT` 三者都没做。已在 `experiments.md` 把这三步显式写成
  机制 B 的必经步骤。

  三处次要点顺带修掉：`log_kv_semantic_clusters=True` 现在构造时硬性拒绝
  `importance_pooling=True`/`pin_size>0`（两者与 Ward/DP-means 依赖的"`w` 就是
  真实计数权重"这个前提冲突，共存数学未推导）；`glossary.md` 的 `n_c`/`K_eff`
  词条同步拆分/澄清；`risks-and-open-questions.md` 里引用 Ward 尺寸加权的地方
  统一改成 `n_total`。

- **2026-08-14｜补齐六处会导致训练/统计静默出错的实现难点。** 动机：用户逐条指出
  `op_log`、Stage 0 dump、`w=0` 填充、decode 阶段都还停在"有原则没规格"的状态。
  逐条给出结论：
  ① **`op_log` 补成完整可执行规格**：六类操作拆成"主操作"（`NEW_CLUSTER`/
  `JOIN`/`NEW_SEGMENT`，每 token 恰好一条，直接对应 §5.3 的三路判定）和"结构操作"
  （`WARD_MERGE`/`PAD_INSERT`/`CARRY`，穿插在主操作旁边，顺序规则写死：合并在被
  服务的建簇之前，填充在被服务的开段之后）。**最重要的认识**：重放完全不需要
  centroid、不需要重跑 Phase 1/2/3——centroid 只是路由决策的依据而决策本身已经被
  `op_log` 记录了，`compact`/`_binary_carry` 那套决定数值的算术只依赖"谁进了哪个
  (簇,段)"，跟 centroid 数值无关。顺带发现批量路由（§5.4）冻结 `p_hi_c` 一整个
  flush 批可能让同批内连续同簇 token 误判成"隔了很久"、系统性推高段数，记为 S0.8
  待测项，不是新写死的结论。
  ② **`OP_max` 从经验估计换成可证明的硬上界**：主操作恰好 T 条，`WARD_MERGE`/
  `PAD_INSERT` 最坏各 O(T)（病态输入下 K_max 绑定后几乎每 token 都要腾位——这本该是
  §4 最坏情况分析要覆盖的场景，不是可以假设罕见的），`CARRY` 是标准二进制计数器的
  O(T/B′) 摊还量，低一个量级。**`OP_max = 4·T_max`**，32k 下约 448MB（原来的 224MB
  是经验值不是上界，两者不矛盾，只是后者更可信）。溢出**硬失败**，不动态扩容不静默
  截断，和现有 `_count_tokens` 对 `max_seq_length` 的处理同一套逻辑。
  ③ **`w=0` 填充只验证了均值和锚点，没验证 Σ/Γ**。核对 `compact()` 现有实现
  （`log_kv_cache.py:723-816`）确认：`frac_b`/`cross_frac` 在 `wb=0` 时精确为 0，
  但 `0×有限数=0` 不等于"乘 0 天然安全"——`0×NaN=NaN`。填充槽必须**全部字段**显式
  清零（不只是 `w`/锚点），这不是新规矩，`_append_level0` 在 `second_order=False`
  时已经这么做过。单测相应扩展成断言完整统计元组逐位相等，不只测均值和锚点。
  ④ **Stage 0 dump 的 hook 点拿不到 `attn_mass_by_dist`**。核对 `model.py:1447-1467`
  确认默认路径（无 softcapping）走 `F.scaled_dot_product_attention` 融合 kernel，
  中间 score 张量根本不会被实例化；而原 hook 点在 `apply_rope` 之前，就算能拿到
  score 也是错的（要 post-RoPE）。拆成两套机制：机制 A（现有 hook，拿 `k_raw`/`v`，
  服务聚类统计）+ 机制 B（dump 脚本自己按 query 分块手算 `q_roped @ k_roped.mT`、
  softmax、按距离分桶累加，用完即弃，不实例化 `(T,T)`）。
  ⑤ **`q` 的歧义**：删掉，只 dump `q_roped`——聚类只在 k 空间做，`q_raw` 用不上，
  引入它只会制造第二个歧义源。
  ⑥ **decode 阶段给了 v1 默认决定，不再是完全空白**：生成 token 与 prompt 共享同一
  套簇/`K_max`，不特殊处理（路由本来就不知道 token 从哪来，这是训推一致性的自然
  推论）。但明确记下这不是没有代价：长 CoT 场景下 Ward 合并可能挤掉后续还要用的
  prompt 簇，尺寸加权只是部分缓解（needle 式小簇不论来自 prompt 还是 decode 都最
  容易被挤）。操作后果是 Stage 0/2 的 dump 和 eval 必须覆盖真实生成过程，不能只测
  prefill。顺带记了一条机械但容易漏的推论：`_log_kv_pending` 的"留 1 个 token 挂起"
  要随 flush 粒度提到 128 一起推广。
- **2026-08-14｜修掉四处文档级不一致，其中一处需要重新设计而非改措辞。** 动机：
  用户逐条指出。
  ① **`_binary_carry()` 在 §5.20 被列为"完全复用一行不改"，与 §5.21-3 的"host 镜像
  必须删除、carry 必须全向量化"矛盾**。拆开处理：合并算子本身（level 空放下、非空
  compact 后翻倍上浮）确实复用，移进"完全复用"表；但**驱动它的控制流是核心重写**，
  移进"需要改动"表并显式引用 §5.21-3，不再暗示这是小改动。
  ② **mass bias 公式在 §5.15 已经是 `λ·log(w/M)`，但 §5.20 表格和 glossary 的
  `log_kv_slot_attention()` 词条还写着 `λ·log w`**。后两处标注为"这行描述的是现有
  代码"，并显式指向新公式，避免写单测时抄错。
  ③ **`op_log` 换成新 buffer 后，紧跟着的内存估算还是旧 `route_log` 的 14MB**，与
  上面 `(B,G,OP_max,4)` 的 shape 对不上。换成 §5.21-2 算出的真实数字：224MB/32k。
  ④ **Config A（`K_max=1`）容差待定，且指向一个不存在的 §13-G**。深入之后发现这不是
  "该定多大容差"的问题——**没有任何容差是有原则的**：level 0 从 2-token 合并改成
  单 token 是设计里不可选的改动（§2.1），跟 `K_max` 正交，所以就算实现完全正确，
  `K_max=1` 也不该复现旧数字。**重新设计正确性检验**：挪到 Stage 0/1，用一条纯 CPU
  单测（`anchor_mode=z` + honest-decay 应与 Dirichlet 闭式解逐位一致，绕开 level 0
  宽度差异）；Stage 2 的 A/B/C 三档全部降级为消融参考点，不再是通过/失败判据。
- **2026-08-14｜修掉三处文档级自相矛盾。** 动机：用户逐条指出。
  ① **`p_mid` 的舍入规则冲突**（正文写 `round`、伪代码写 `floor`）。定为
  **round-half-up 的整数实现** `(2·sum_wp + w) // (2·w)`。三条理由：`floor` 对所有
  非整数结果都向下取、系统性左偏约 0.5 个位置，而 `round` 只在平局时才偏（`w=2` 的
  entry 约一半会遇到平局，所以 tie-breaker 必须写明）；浮点 `round` 在 1M 下不安全，
  `sum_wp` 量程 `10¹²` **超出 fp32 的精确整数范围**；**最关键的是 §11-A 要求重放逐位
  可复现**，浮点除法在不同 kernel 下末位可能不同，而 `p_mid` 差一位就会改变锚点、
  改变去重后的 `M`、进而改变 mass bias——所以整数算术不是性能选择，是正确性要求。
  ② **`s_h` 已在 §5.21-4 定案为离线标定，但 glossary 和未决清单还写着"尚未定案"**。
  两处都改掉；§12-B 保留为已解决条目并指向 §5.21-4，只留"标定需要多少留出样本"这个
  小尾巴跟着 S0.2 一起测。
  ③ **§5.16 仍说"唯一的接口性改动"**，与 §5.21-1 列出的七项影响面冲突。改成"核心的
  接口性改动"并加警告框，把 `q`/`k_raw`/`k_roped` 双持有、两个调用点、in-flight
  chunk 逐位一致、`_log_kv_pending` 带 `k_raw`、注释同步改这几项直接列出来，避免
  实现者按"多传两个参数"估工作量。
- **2026-08-14｜新增 §5.21「开工前必须定死的五个决定」与 Stage 0 的 dump 规格。**
  动机：用户指出五处"写得太轻、会在实现到一半卡死"的地方。逐条给了结论：
  ① **pre-RoPE 接口不是"多传两个参数"**。核实 `model.py:790-816` 后确认，pre-RoPE 精确
  指**qk-norm 之后、apply_rope 之前**（Qwen3 `norm_qk=True`），取错位置会让度量落在
  未归一化空间、`s_h` 标定失效。影响面列了七项：`q` 仍需 post-RoPE 所以要**同时持有
  `k_roped` 和 `k_raw`**；training 与 inference 是**两个不同调用点**；`append_exact_tokens`
  的 in-flight chunk 必须用 post-RoPE，且与 cache 里 `w=1` entry 物化出的键**逐位一致**
  （硬性单测）；`_log_kv_pending` 挂起的元组要带 `k_raw`；partial rotary 下只有前段需
  区分；GQA 折叠只在读出侧、写入侧无影响；`model.py:805-808` 那段 expected-RoPE 注释
  **正是被替换的东西，必须同步改**。
  ② **`route_log` 规格不足，改为 `op_log`**。`K_max` 满时 Ward 合并会**释放并复用
  cluster id**，只记 token→cluster 无法重建结构。改成六类操作的日志，其中
  **`WARD_MERGE` 必须记明保留/释放哪个槽**、**`PAD_INSERT` 必须记**（否则重放时对齐
  位置错位、后续配对全错）。代价 224MB/32k（比 route_log 贵一个量级），是正确性的价格。
  ③ **carry 必须全 GPU 向量化，`_counts` 的 host 镜像要删掉而非扩展**。现实现靠它避免
  GPU sync 且所有控制流读它；按 `(B,G,K,L)` 展开后 Python 标量分支语义上不成立。决定：
  **循环轴换成 level（静态界 `L_alloc`），对全部 (B,G,K) 用掩码并行 carry**——级联深度
  虽数据相关但被 `L_alloc` 静态界住，于是数据相关性从控制流挪进掩码。代价是不能提前
  退出，被 §5.4 把 flush 粒度提到 128 吸收。
  ④ **`s_h` 定案为离线标定**，在线估计降级为消融。理由：在线会引入 §11-E 的新变体、
  序列开头未收敛（而早期路由错误不可恢复）、跨层曲线不可比。**标定值必须写进 eval
  metadata**。
  ⑤ **Stage 0 dump 规格补全并提为第 0 步**（hook 点、张量清单、维度约定、层子集、
  needle span 对齐、落盘格式）。其中两条关键：**不要 dump 完整 attention**（32k 下
  172 亿元素），§13.2 需要的距离分桶质量应在 hook 内就地累加；**needle span 必须是
  分词后绝对 token 下标且与 eval 用同一份转换代码**，否则 S0.3 全是噪声。
- **2026-08-14｜给"小簇合并零损失"加条件；补 CompressKV / ChunkKV 进相关工作并新增
  §9-A 正视其数字。** 动机：用户指出两处。
  ① **"零信息损失"说得太绝对**。它只覆盖**单次**合并且合并后该层占用 `≤ B′` 的情形，
  **不是不变量**。`K_max` 持续绑定时占用会累积：而 Ward 的尺寸加权偏好小簇，needle
  簇正是最小的那些，于是反复合并会把 needle **系统性漏斗到同一个簇**，其 level 0 在
  `B′=8`、needle 数为 8 时正好填满，此后 needle 之间开始互相平均，且该簇 centroid
  已是 8 个无关 needle 的混合体。**multi-needle + sink 就是这条失效路径的最坏情况**。
  口径改为"`K_max` 不绑定时无池化损失"，并要求 **multi-needle 实验必须同时报告
  `K_max` 绑定频率**，否则分数差异无法归因。
  ② **相关工作缺了 semantic-retrieval 这一支**。补入 [CompressKV](https://arxiv.org/abs/2606.24467)
  与 [ChunkKV](https://arxiv.org/pdf/2502.00299)。**CompressKV 的数字对本方案的动机
  是直接冲击**——NIAH 用 0.7% 存储达基线 90%，而我们的预算是 7%。新增 §9-A 正面处理：
  三条区分（eviction vs 压缩、`O(n)` vs 次线性、不可逆承诺）都成立，但**第一条正是
  pin 系列的死因之三**，是我们自己付过学费的，所以不能当托词用。诚实结论是
  **单针 NIAH 已不足以证明本方案的价值**，有区分度的是 multi-needle、多轮、以及
  **query 不在 prompt 尾部**的设定（第三个要补进消融）。
  连带：§13.2 的新颖性主张让一步——CompressKV 的**层间** Frobenius 误差预算分配在
  方法论上就是测量驱动的率失真分配，所以本方案能主张的是**分配轴不同**（ladder 层级
  vs 模型层），不是"没人做过"。ChunkKV 的"连续块优于离散选择"则是 **S0.0 的旁证**。
- **2026-08-14｜收紧 mass bias 单测的表述；澄清 `K_max` 的"推导 vs 可覆盖"语义。**
  动机：用户指出两处措辞比事实更强。
  ① **"mass bias 守恒"写成了一般恒等式,实际只在等 logit 下成立**。M 槽总质量是
  `(w/M)^λ·Σ_a exp(s_a)`，单槽是 `w^λ·exp(s)`，两者相等需要**所有 `s_a` 相等且
  `λ=1`**（`λ≠1` 时还差 `M^(1-λ)`）。而不同锚点 score 不同**正是锚点展开的目的**——
  位置敏感检索靠它实现，不该被当成偏差去测。单测改成两条不同强度的命题：**可以精确
  断言的是"计数守恒"**（`Σ_a` 计数因子 = `w` 而非 `M·w`，不依赖 score，且正好抓住
  原 bug）；总质量守恒则必须先把锚点强制取同一位置使 logit 相等再测。
  ② **`K_max` 一处写"不是自由参数"，另一处消融又在扫 16/64**，自相矛盾。改成
  "**默认** `max(4, ⌈log₂N⌉)` 由 `N` 推导，**但允许实验覆盖**"，并写明取上取整的理由
  （`K_max` 绑定会改变算法性质，宁松勿紧）。连带补上一条之前漏掉的约束：
  **覆盖 `K_max` 时必须重算 `L_alloc`**——否则 `K_max` 从 15 调到 64 而 `L_alloc`
  不动，总 entry 数会从 1320 变成 5632，把内存对齐的公平性论证打穿。
- **2026-08-14｜修正 `p_mid` 的合并规则（原规则不满足结合律，且在实践中恒等退化）。**
  动机：用户指出 `p_mid = A.p_mid if w_A ≥ w_B else B.p_mid` 破坏结合律——三个等权
  token 下 `merge(merge(A,B),C)` 得 `p_A` 而 `merge(A,merge(B,C))` 得 `p_B`，于是
  §2.2 声称的"合并顺序不影响最终值"和 S0.1 的"任意二叉合并顺序结果一致"单测都会挂。
  核实后发现**退化比这更彻底**：`_binary_carry` 的进位路径上每次 `compact` 都是两个
  **等宽**块相merge，`w_A == w_B` 恒成立、`>=` 恒选左侧，归纳可得**平衡路径上
  `p_mid ≡ p_lo` 恒等**——第三个锚点信息量严格为零，`M` 恒为 2。
  **修法**：`p_mid` 改成位置的加权均值，存**未归一化的 int64 累加器** `sum_wp = Σ w_j p_j`
  （合并就是整数加法，结合律逐位成立；只在读出时除一次并 clamp 回 `[p_lo, p_hi]`）。
  关键认识是**抗相消性根本不依赖"锚点是真实成员位置"**——它只依赖"每个虚拟槽在单一
  确定位置上做一次 RoPE"，`ρ` 照样恒为 1。所以 `p_mid` 可以是质心。
  **顺带补上 v3 的一个已知短板**：§2.3 对比表里"命中典型/中心位置：弱"正是这个退化
  造成的，改成真质心后 v3 拿到了 v2 圆均值的中心估计能力而没有相消代价。
  **但代价要诚实记**：bug 的副作用恰好压低了 `M`，修好后 `M` 从 2 趋近 3，直接顶到
  §4 那行"E[M]=3 ⇒ 4984 槽 ⇒ 超 vanilla"，**所以 S0.6 从验证性测量升格为真正的决策门**，
  退路是砍掉第三锚点（相对修正前零损失）。`sum_wp` **必须 int64**（量程 `n²`，1M 下
  达 `10¹²`）。
- **2026-08-14｜新增 `docs/glossary.md` 术语表。** 动机：用户问 ladder 是什么、
  `(B, G, ...)` 里的 B 和 G 是什么，暴露出这些词一直在用但从没定义过。**最值得记的
  是 `B` 在代码库里被重载了**：张量 shape 注释里是 batch size，而构造函数参数
  `B`、`self.B`、CLI `--log_kv_B`、`compact` docstring 的 "B-slot block" 全都是
  **每层 entry 数**（默认 512）——这就是设计文档里改叫 `B′` 的原因。术语表另收了
  结构层次的嵌套关系（cache → cluster → segment → ladder → level → entry → slot）、
  维度记号、当前参数、**已废弃记号**（`z`/`ρ`/`β`/`κ`/`τ`/`SEG_max`，它们只出现在
  变更记录里，读旧条目时需要）、代码符号和外部概念。
- **2026-08-14｜文档拆分成四个文件（内容逐字不变）。** 动机：单文件到 1392 行，每次
  接续都要读全文，开销持续上升。拆法见文件开头的**文档地图**：CLAUDE.md 留 §0–§4
  （动机与核心设计）、§9、§10、§14（变更记录）；§5 进 `docs/algorithm-spec.md`；
  §6–§7 进 `docs/experiments.md`；§8、§11–§13 进 `docs/risks-and-open-questions.md`。
  **全局章节编号刻意保持连续**，所以既有的 `§5.19`、`§11-A`、`§2.3` 这类交叉引用
  在任何一个文件里都仍然指向同一处，不需要逐条改写。**变更记录一律加到 CLAUDE.md
  §14，正文细节改到对应章节所在的文件。**
- **2026-08-14｜新增 §5.20（与现有实现的复用边界）与 §13（压缩机制本身的剩余空间）。**
  动机：用户问"之前做的 Fenwick 和二阶修正还能不能延续"以及"压缩方向还有多少提升
  空间"。要点：
  ① **复用边界**：被替换的只有**槽成员划分规则**和**位置表示**两个设计决定；`compact`
  的加权均值、`_binary_carry`、整套 Chan + rank-1 机制、打分/读出结构、GQA 折叠全部
  一行不改。**唯一一处"看起来能复用实际不能"的是训练重放的依据**——docstring 把确定性
  论证为"count-based binary carries"（只依赖计数不依赖数据），而语义路由打破了这个
  前提，必须改成重放 `op_log`（§11-A、§5.21-2），不处理不报错但梯度属于另一个函数。
  ② 顺带发现**二阶修正的精度应该变好**：现在 Σ 统计在 post-RoPE 空间，槽内差异里混着
  方向弥散的**位置相位方差**，白白吃掉 rank-1 唯一的那个秩；改到 pre-RoPE 后 Σ 只量
  内容方差，而簇内同质正是聚类在优化的目标。所以 D1 自检 width≥8 的失真可能有一部分
  是被位置污染拖累的，不是方法上限。
  ③ **压缩轴上还没被动过的三处**（按回报÷成本排序）：**窗口占了 44% 预算且零压缩**，
  而 `recent_size=1024` 是位置分桶时代继承的——那时不存在"语义精确"这个选项，现在
  level 0 存单 token 提供了同等精确性且与位置无关，所以 `recent_size : K_max : B′`
  应当整体重扫；**每层固定 `B′` 个 entry 这个分辨率分配曲线从来没被论证过**，它只是
  二进制计数器的副产品，正确提法是率失真分配，而命中概率和失真两个因子都可离线测；
  **Σ 和 Γ 分了同样的秩但重要性不对等**（分数误差被 softmax 部分自我纠正，读出误差
  直接进输出），且"提高秩"与"改善成员划分"是攻击同一误差项的**替代品而非互补品**。
- **2026-08-14｜更正 Ward 合并的理由（原来写错了），补全 K_max 满时的完整合并过程。**
  动机：用户追问"`K_max = f(N)` 具体是什么"和"满了怎么合并"，核对时发现 §5.6 里
  "被牺牲的是最相似的两个簇，多半是两个 haystack 簇"这句话是错的——Ward 代价带
  尺寸加权 `(n_a n_b)/(n_a+n_b)`，两个 haystack 大簇的代价（500×0.01=5）远高于两个
  needle 单点簇（0.5×1.0=0.5），**Ward 优先合并的是小簇而不是大簇**。原句是按
  "合并最近的两个 centroid"的直觉写的，漏了尺寸加权会翻转排序。
  **结论仍然成立但机制不同**：合并两个簇**不必然合并它们的 entry**——只有当某层
  entry 总数超过 `B′` 才触发 compact，所以**在 `K_max` 不持续绑定的前提下**两个小簇
  合并时 entry 原样共存、无池化损失，needle 的分数/读出/位置全不受影响（§3 只依赖它自己的 entry，不依赖 centroid）。
  真正的代价是路由质量而非已存内容的分辨率。而 Ward 的尺寸加权恰好与"entry 会不会
  被迫合并"同向，所以 **Ward 依然是正确准则，只是正确的理由是这个**。
  另：补全五步合并过程（含 `if 总数 ≤ B′` 这个关键分支）；修正 §5.6 表格里 128k 行
  的算术错误（`L_alloc` 13→12，`E` 1768→1632，相对值 1.19×→1.13×）。

- **2026-08-14｜补齐算法规格的三处空白，修掉两处自相矛盾，新增 §12 未决问题清单。**
  动机：用户连续追问"簇数要不要限制/怎么随长度变化/段边界怎么阻断"，暴露出规格里
  几处只有原则没有方案的地方。要点：
  ① **段机制曾经完全空转**——§2.1 说 segment 用于"约束合并"，但 §5.11 建议"允许
  跨段合并"，净效果是这个字段从不被读。修法是**对齐填充**：不拒绝某一对（`compact`
  按固定下标 `(2i,2i+1)` 全配对，没有逐对否决的接口），而是插入 `w=0` 空位把边界
  推到两对之间；`w=0` 在现有合并数学下天然是恒等元，`compact` 零改动。代价是
  `2^ℓ_block − 1` 槽/边界，**指数增长直接把 `ℓ_block` 限死在 1~2**。
  ② **`η` 不能退化成纯连续分段**——它只出现在排序代价里，`η→∞` 的实际效果是退化
  行为而非纯分段。真正插值的是 `(g_max, ℓ_block)`，**S0.0 相应改扫这一对**。
  ③ **`K_max` 改为 `max_seq_length` 的函数**（`L_alloc` 早就是了，这是不一致）。
  `K_max ∝ log₂N` 时总预算是 `Θ(log²N)`：上下文 32k→1M 涨 32 倍，内存只涨 1.46 倍。
  并列出 `K_eff(n)` 的三种可能形状（**填充数饱和** / `α log n` / `n^d`），指出 32k
  单点分辨不出、S0.2 必须测整条曲线才能外推。
  ④ **不设按长度的运行时配额**：簇数跟踪语义多样性而非长度，短序列用满配额是正常的；
  救命的性质是"簇多⟹每簇浅、某簇巨大⟹其余很小"这两个最坏情况互斥。
  ⑤ 由 ④ 推出**矩形预分配超配 1.3~1.4 倍**，`L_alloc` 改按均衡界定尺，极端不均衡时
  **顶层做饱和累加**（`compact` 本就不要求两侧等宽）。
  ⑥ 新增 §5.13 buffer 清单、§5.18 实现顺序、**§5.19 实现 tips 与易错点**（其中
  `op_log` 与 `reset_parameters()` 的时序冲突、锚点哨兵值越界、`w=0` 槽必须显式
  掩码三条是会静默出错的）。
  ⑦ 新增 **§12 未决问题清单**：decode 阶段、`s_h` 估计方式、sink 占预算、`K_max` 与
  multi-needle 的冲突、Config A 闸门无容差、缺 eval-time 探针。
- **2026-08-14｜写入完整算法规格（§5.1–§5.12），新增 §11 技术难点清单。** 动机：
  用户要求把"到底怎么分簇、簇内怎么压缩保证 O(log n)"落成可实现的规格。要点：
  ①**度量统一为原始 pre-RoPE key 上的平方欧氏距离**，因为同一个量同时是 DP-means
  分配准则、Ward 合并代价、和 rank-1 残差；阈值必须相对化 `λ_new = λ_rel·s_h`。
  ②**三路判定**：统一代价只用于挑候选，两个阈值分别管"新簇"和"新段"。
  ③**三阶段批量化路由**，摊还论证是"串行的 Phase 2 总次数被 E[K] 界住而非 O(T)"。
  ④**O(log n) 的准确陈述**：由 log 的凹性可知**最坏情况是簇均衡**，界是安全的。
  ⑤**"小簇无损"**被识别为 §3 needle 论证的真正承重点。
  另修掉记号冲突（`λ` 曾同时表示 mass bias 和 segment 阈值）。
- **2026-08-14｜v3 → v3.1：时序打断不再 fork 新簇，改为 cluster/segment 两层分离。**
  动机：用户要求为 v3 的规则找反例。反例一致命——高频复现实体在 fork 规则下产生 50
  个语义相同的簇，K 正比于话题切换次数而非 log n。根因是 fork 把**簇身份**与**存储
  局部性**绑死。副产品：最早被降级的"连续性约束 Ward 合并"作为存储层回归。
  连带产出：反例二（相邻更正仍被均值池化，§2.4）、反例三（key 空间未必是语义空间，
  §2.5，要求 Stage 0 逐层逐头报告）。
- **2026-08-14｜v2 → v3：位置表示从"求平均的 z 统计量"改为"不做平均的真实锚点集"。**
  动机：用户指出 v2 的 z 仍是加权求和，相位相消依然存在，只是被诚实报告而已。解法：
  只存真实成员的 `(p_lo, p_hi, p_mid)`，读出时在每个真实位置上做标准 `apply_rope`
  （ρ≡1）；合并用 min/max 幂等半格运算。用户同时提出的时序约束，被 DeltaKV 实测的
  "60% 相似 token 相距 >16 位置"证明是**前提而非补丁**。
  连带产出：mass bias 必须按去重锚点数 `M_s` 摊薄；v2 的 `κ`/`log ρ` 整条被消解。
- **2026-08-14｜新增 §2.4 开放问题：均值池化无法表达 supersession。** 动机：用户提出
  "考 100 分→更正为 90 分"场景并提议每簇维护 gated DeltaNet 线性 state。结论：
  **问题真实，但该方案暂不采纳**——线性 state 的 `qᵀS` 读出没有选择性，且簇内 key
  近平行是线性容量最坏情况，会让阈值失手的代价从"稀释"恶化成"不可恢复的抹除"。
  可行替代：现有 rank-1 Γ **结构上已是秩 1 线性 state**，改构造方式即可。
- **2026-08-14｜v1 → v2：位置重物化从 β 插值旋钮改为满模长旋转 + log ρ 独立置信项。**
  （已被 v3 取代）动机：`gain=ρ^{(β-1)}` 在 β=0 时对最不可信的频率通道施加最大重
  物化力度，方向性错误。
- **2026-08-14｜方案定型：语义簇 + 簇级抽象位置。** Ward 合并降级为簇内合并顺序
  策略。**注：v3.1 把它作为 segment 存储层重新纳入，v3.2 进一步登记为 S0.2 决定的
  存储层分叉选项。**
- **2026-08-14｜建档。** 从 `claude/semantic-cluster-log-compression-8ai32a` 分支切出
  `semanticLogKV`，本文件取代该分支的 CLAUDE.md 作为独立存档。
