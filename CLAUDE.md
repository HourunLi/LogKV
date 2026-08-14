# SemanticLogKV 开发指导文档（存档，随讨论推进持续更新）

> 本文件是 `semanticLogKV` 分支的唯一存档文档，记录"语义簇 + 相位因子化簇级位置向量"
> 这条压缩方案的设计、实现进度、实验结果和踩过的坑。**每次讨论出现突破或进展，都要
> 写回本文件**（新增到 §11 变更记录，并同步更新受影响的正文章节），不要另开文档。
>
> 本文件只保留 SemanticLogKV 直接相关的内容。原 `logKV`/`claude/semantic-cluster-
> log-compression-8ai32a` 分支上的 CLAUDE.md 是另一条独立技术路线（位置分桶 +
> rank-1 二阶修正 + pin 机制）的进展存档，与本分支无关，不在这里维护——那条线的
> 结论只在本文档 §1、§9 里作为背景锚点和教训引用，不展开叙述。

## 0. 现状速览（2026-08-14 建档）

**这是什么**：LogKV 现有压缩机制（`litgpt/log_kv_cache.py`）按**位置**做 Fenwick 树式
分层压缩——固定 2:1 合并，越老的 token 槽越宽、压得越狠。已验证的结论（另一分支的
存档）：这套方案的最大瓶颈是**槽内异质性**——语义无关的 token 被硬塞进同一个位置槽
做均值池化，needle 稀释成 1/width，niah@32768 从稠密上限 0.9353 掉到 0.0827。

**本分支的方案**：把"按位置分桶"换成"按语义分簇"——语义相近的 token 才共享槽，
needle 因为和周围内容语义不相关而倾向于自成一簇，槽内不再异质，稀释问题从结构上
消失。配套需要解决两个新问题：①位置信息不能再靠 post-RoPE 均值池化表示（成员位置
散布会导致 RoPE 相位相消,连累内容一起被抹掉）,需要一种"簇级抽象位置"表示；
②簇数不能固定，需要一个由内容驱动、预算仍是次线性的自适应机制。

**当前阶段：设计已完成两轮迭代，代码尚未开始写。**

- v1（初版）：位置用逐频率复数统计量 `z`（特征函数），读出时用单一标量 `β` 在
  "诚实衰减"（β=1，等价现有方案）和"满模长旋转"（β=0，内容不衰减）之间插值。
- **v2（当前版本，已取代 v1）**：v1 的 β 旋钮被指出有结构性缺陷——它把"往哪个方向
  转"和"这个方向有多可信"这两件不同性质的事情耦合进同一个标量，且 β=0 时对着
  最不可信的频率通道（ρ 最小）反而施加最大的重物化力度，等于把不确定性当噪声灌回
  内容。v2 改为：**旋转恒定用满模长（永远相信相位点估计,不衰减内容)，可信度改走
  一条独立的加法置信项进分数**（`κ·log ρ̄`，与现有 `λ·log w` mass bias 同源同权）。
  完整推导见 §2.3，这是目前唯一的"突破性进展"，之后的讨论应在此基础上继续。

**下一步（按优先级）**：
1. Stage 0 离线证伪实验（§6）——不需要 GPU 训练，只需要一次 dump + CPU 分析，
   决定整个方案值不值得往下做。**这是唯一应该先做的事**，不要跳过直接写生产代码。
2. `litgpt/log_kv_position.py` 四个纯函数 + 单测（§5.1），CPU 可测，不依赖 Stage 0
   的结论。
3. 视 Stage 0 结果决定是否继续 Stage 1（生产代码）。

## 1. 背景：为什么从位置分桶转向语义分簇

（完整证据链在另一分支的 CLAUDE.md 里，这里只摘录作为本方案设计依据的结论性数字，
不重复过程。）

- 稠密 attention 上限（CPT 后模型）：niah@32768 = **0.9353**。
- 现有 LogKV（位置分桶 + rank-1 二阶修正，无 pin）：niah = **0.0827**，LongBench =
  **0.1716**，LongBench_e = **0.1918**，ACC = **0.6113**。压缩本身吃掉 85+ 个百分点，
  是全链路最大的单一瓶颈，比 pin 全系列实验的影响大一个数量级。
- 机制诊断（D1 自检）：width≥8 时 rank-1 近似失真明显；根因是均值池化把 needle
  稀释成 1/width，二阶修正是 rank-1、只能加不能减，救不回一个已经被稀释到"不参与
  softmax 竞争"的槽,尤其救不回**读出侧**(`v_slot` 同样被稀释,Γ 修正同样是
  rank-1)。
- 已经试过的邻近方案（重要性加权池化,不改变槽成员划分,只改池化权重）在 niah 上
  定向变差（0.0827→0.0787）——说明问题出在"哪些 token 同槽"，不是"槽内怎么加权"。
  这是本方案（改变槽成员划分）相对该方案（改变槽内权重）的直接论据。

这几个数字是本分支所有实验的**对照锚点**，任何新结果都应该对着这张表汇报：

| 配置 | ACC | LongBench | LongBench_e | niah@32768 |
|---|---:|---:|---:|---:|
| 稠密（CPT 后）| — | — | — | 0.9353 |
| 现有 LogKV vanilla（位置分桶）| 0.6113 | 0.1716 | 0.1918 | 0.0827 |
| SemanticLogKV（本分支目标）| ? | ? | ? | 目标 ≫ 0.0827 |

## 2. 核心设计

### 2.1 语义簇路由 + 簇内 ladder + 非参预算

三层结构，从新到旧：

| 层 | 内容 | 位置表示 | 规模 |
|---|---|---|---|
| recent window | 精确 token，不压缩 | 标准 RoPE（`ρ=1`）| `recent_size`（1024）|
| 簇池 | K 个语义簇，DP-means 路由 | 每簇每 entry 一个 z | `K ≤ K_max` |
| 簇内 ladder | 该簇成员的 Fenwick 层级 | 合并时 z 精确加权平均 | `B′ × L_max` per cluster |

**路由规则（DP-means，即 CRP 混合模型的小方差极限）**：token 离开 recent window
（flush 时）才参与路由——与各簇 centroid 的余弦相似度取最大，`≥ τ` 则加入，否则
开新簇；`K` 满且仍需开新簇时，按 Ward 代价（`p_a·p_b·‖k_a-k_b‖²`，与现有
`_pair_rank1_stats` 里已经在算的量同构）合并最近的两个簇腾位。聚类在 **pre-RoPE**
key 空间进行——簇身份必须是纯语义的，这也是"必须先做位置-内容解耦"的原因之一（否则
位置差异会污染语义相似度）。

**簇内 ladder 是承重墙，不是装饰**：如果只有一层平面簇，needle 一旦没被 `τ` 隔离、
而是并入一个几千成员的大簇，就被稀释成 1/几千——比现在的 1/32 更糟。簇内 ladder
保证新近成员保持细粒度。与现状的一处语义变化：ladder 的 level 0 存**单 token**
（`w=1`），不是像现有方案那样存"2 token 合并"——这样每个簇的 ladder 是它自己成员
序列上的标准 Fenwick，单成员簇的 entry 永不合并（§3 的核心论据）。

**读出端不感知簇结构**：`log_kv_slot_attention` 只看到一个扁平槽池
`(k_eff, v̄, w, stats)` 加一个有效位掩码，接口和数学不用改。簇只是写路径的路由
元数据。这是保持张量矩形、避免 ragged 布局的关键设计（见 §5.4）。

**预算的理论形状**：DP-means 是 CRP 小方差极限，簇数天然满足

```
K_n = 1 + Σ_{i=1}^{n-1} Bern( α / (α+i) )      E[K_n] ≈ α·ln n
```

因为是独立 Bernoulli 之和，Chernoff 直接适用，超出常数倍期望的概率是 n 的多项式
小量——**高概率界**，不只是期望好看。最坏 `K=n`，且只在"每个 token 都远离所有
已有簇"（数据真的没有可压缩结构）时发生，退化是数据挣来的，不是运气差。

**必须提前接受的经验事实（未验证，Stage 0 S0.2 待测）**：自然语言的类型数按
**Heaps 定律** `~ n^d`（d≈0.4–0.6）增长，不是 log n，对应 Pitman-Yor 而非 CRP。
`K_eff` 大概率是幂律而非对数——`O(√n)` 依然是合格的次线性目标（32k 下约 181 簇，
仍远少于现在的约 2560 槽），论文里写 `O(n^d)` 带可调 d 比硬说 log n 更站得住，但
这必须实测，不能假设。

**生产实现用硬顶,不是无限期望**：预分配 `K_max × B′ × L_max` 个槽，占用随内容
浮动但上界固定。"不设上限、期望 O(log n)、最坏 O(N)"的版本保留为论文里的曲线和
ablation，不进生产路径（serving 不能接受期望有界但尾部无界）。

### 2.2 位置表示：z 统计量（特征函数）

**为什么必须做**：现有方案存 post-RoPE 键均值 `E[R(p_j)·k_j]`。Qwen3 是 full
rotary，频率 `θ_f` 波长从约 2π token（高频）到百万 token（低频）。槽成员位置散布
`s` 时，`θ_f·s ≫ 2π` 的频率上 `E[e^{iθ_f p}] → 0`——而 full rotary 下**这些频率
搭载的内容维度会被一起清零**。语义簇成员散布上千 token，直接套用现有池化会让
键的大部分维度归零，内容检索信号和位置信息同归于尽。这不是精度损失，是表示的
湮灭，位置必须单独表示。

**定义**：每个槽在原有 `(k̄, v̄, w, Σ, Γ)` 之外维护逐频率复数统计量：

```
z_f = ( Σ_j w_j · e^{i·θ_f·p_j} ) / ( Σ_j w_j )      f = 0 … F-1,  F = rope_n_elem/2
```

实数形式存 `z_re[f] = E[cos θ_f p]`、`z_im[f] = E[sin θ_f p]`。相位 `arg z_f`
是圆均值位置，模长 `ρ_f = |z_f|` 是该频率上的位置集中度。键改存 **pre-RoPE 内容
均值** `k̄_raw = E[k^raw]`。存储代价：每槽 `rope_n_elem` 个 fp32,槽数仍是 O(log n)
量级,不改变渐近。

**三条性质（v1、v2 通用，是整个方案的数学基础）**：

1. **嵌套精确性**：单 token 槽 `ρ=1`、相位 `=θ_f p`，重物化后严格等于标准
   RoPE——精确 token attention 是本表示的特例。
2. **合并精确封闭**：`z_merged = (w₁z₁+w₂z₂)/(w₁+w₂)`，加权复数平均，精确、可
   结合、**零累积误差**（对比内容侧 rank-1 截断误差会随层级累积，位置这条轴现在
   完全无损）。
3. **连续 span 闭式解**：内容恒定的连续 w-token span，`ρ_f` 精确等于
   `|sin(wθ_f/2)/(w·sin(θ_f/2))|`，正是现有 LogKV docstring 里的 Dirichlet
   因子——新表示连续地包含旧方案，这给单测提供了逐位可核对的锚点。

**因子化的诚实代价**：现有 `E[R(p)k]` 是精确的一阶矩（线性性）；换成
`E[R(p)]·E[k]` 会丢掉位置-内容耦合项：

```
E[R(p)k] = E[R(p)]·E[k] + Cov( R(p), k )
                          └─ 被丢弃，量级 ≈ (位置散布) × (内容方差)
```

该误差由簇内内容方差控制——**聚类越纯，因子化越准**，这是"语义分簇"和"位置因子化"
互为前提的数学表述，也是 Stage 0 S0.4 要实测的量。

**实现红利（litgpt 特有）**：`apply_rope`（`litgpt/model.py:2074`）用"前后半重复"
布局——`x1=x[...,:d/2]`、`x2=x[...,d/2:]`，`out=x·cos+cat(-x2,x1)·sin`,且
`cos[f]==cos[f+d/2]`。于是：
- 单 token 的 z **不需要任何三角函数**——`z_re=cos_cache[...,:F]`、
  `z_im=sin_cache[...,:F]`,直接切模型已算好的 RoPE cache。
- 物化不需要 atan2——增益直接乘在 `z_re/z_im` 上就是 `cos_eff/sin_eff`，复制成
  前后半后喂给现成的 `apply_rope` 数学。

整个位置机制约 40 行纯张量代码，没有新的数值原语。

### 2.3 v2 修正：满模长旋转 + log ρ 置信项（取代 v1 的 β 插值旋钮）

> **本节记录了本分支迄今唯一一次实质性设计突破，完整过程见 §11 2026-08-14 条目。**

**v1 的做法（已废弃）**：读出时 `gain_f = ρ_f^{(β-1)}`，`β=1` 是"诚实衰减"
（`gain≡1`，直接用未归一化的 z，等价现有方案）,`β=0` 是"满模长旋转"
（`gain=1/ρ_f`，内容不衰减）,中间值插值。

**v1 被指出的缺陷**：`arg(z_f)` 在 `ρ_f` 很小时几乎是任意的——一个接近相消为零的
向量,其辐角对微小扰动极其敏感,携带信息量趋近于零。而 `gain=1/ρ_f` 在 `ρ_f` 越小
时越大——**对着信息量最少的通道,反而施加最大的重物化力度**,等于把不确定性当
噪声灌回内容,而不是老实地报告"这里不确定"。这是方向性错误,不是精度问题。

**关键的统计学观察**：圆均值相位 `arg(z_f)` 本身是有原则的点估计——它是最小化
`E[1-cos(Φ-φ̂)]` 的最优解（`E[cos(Φ-φ̂)] = ρ·cos(φ̂-arg z)`，在 `φ̂=arg z` 时
取最大值,**与 ρ 大小无关**）。也就是说"往哪个方向转"这件事,用相位点估计本身
没有错;错的是"这个估计有多可信"不应该体现在旋转的模长上（模长是正交变换,天然
应该保范数,不该被用来编码不确定性),而应该是一个独立的量。

**v2（当前版本）**：

```
k_eff   = apply_rope( k̄_raw, cos = z_re/ρ, sin = z_im/ρ )     ← 恒定满模长,不设 β
score_s = scale·(q · k_eff_s)  +  κ · pool_f( log ρ_f,s )  +  λ·log(w_s)  +  [二阶修正项]
```

- 旋转**恒定用满模长**(去掉 β,只保留作为消融对照,见 §7)——内容能量永远不因
  簇内位置散布而衰减,这一点不需要调参就成立。
- ρ 的置信信息改走一条**独立的加法项**进分数,`pool_f` 是对逐频率 `log ρ_f`
  的某种汇总（候选：只取内容真正占权重的低/中频段均值,或按该频率对应通道在
  `k̄_raw` 中的能量加权）,`κ` 是可学习/CPT warmup 的标量,与 `second_order_scale`
  同一套爬坡机制。这与现有 `+λ·log(w_s)` mass bias **同源同权**——都是"槽的
  元信息（代表了多少 token / 位置有多确定）不该弄脏内容,应该直接进 softmax
  排序逻辑",风格上是这套代码库一以贯之的设计语言,而不是新发明的机制。

**needle 论证不受影响，而且更干净**：单点簇 `ρ_f≡1`（精确位置,无散布）→
`log ρ̄=0` → 置信项恒为零 → §3 的"needle 逐位恢复稠密行为"完全保留，且不再需要
指定任何 β 值——needle 的行为不再依赖任何超参,论证反而比 v1 更干净。

**这次修法解决的和没解决的,要分清楚**：
- **解决了**：内容不再因为位置散布被系统性衰减/污染；不确定性从"暗中损坏内容"
  变成"模型可见、可学习权重的显式信号"。
- **没有、也不可能解决**：z 本身依然是加权和，簇成员散布依然会导致 `ρ_f→0`——
  这是数学事实,不是实现细节,任何试图把多个位置压缩成一个数的表示都躲不开。
  同簇里两个时间上相距很远但语义相近的成员,z 的圆均值相位会给出一个"虚构的
  中间位置",这是聚类丢弃单成员位置精度的本质代价,只能靠簇内 ladder（§2.1）
  降低发生频率,无法从表示层面消除。这一点必须在论文/文档里如实写清楚,不要
  暗示"相位问题被解决了"。

## 3. 为什么能解决捞针

逐项对比三种情形下 needle 的命运。

**稠密注意力**：`score = scale·(q·k_needle)`，`value = v_needle`。needle 靠
`q·k_needle` 显著高于 haystack 而胜出，读出的是它自己的 value。

**现有池化（needle 落在宽度 w 的槽里）**：

```
k_slot   = ( k_needle + Σ_{j≠n} k_j ) / w
q·k_slot = (q·k_needle)/w + (w-1)/w · (q·k̄_hay)
v_slot   = ( v_needle + Σ_{j≠n} v_j ) / w
```

分数侧和读出侧**同时**被稀释 w 倍。`+λ log w` 补的是计数质量而非内容,补不回
稀释；即使这个槽赢得注意力,返回的 `v_slot` 也已经是 haystack 均值,而 Γ 修正是
rank-1,救不回一个被 w 倍稀释的方向。这解释了为什么二阶修正把 niah 从 0.032 提到
0.0827（2.6×）却依然离 0.9353 极远：它修的是分数,修不了读出。

**SemanticLogKV**：needle 与所有 haystack centroid 余弦 `< τ` ⇒ 自成一簇 ⇒
该簇成员数=1 ⇒ ladder 永不填满 ⇒ entry 永不合并：

```
score = scale·(q · k_needle) + κ·log 1 + λ·log 1 = scale·(q · k_needle)   ← 与稠密逐位相同
value = v_needle                                                           ← 与稠密逐位相同
z: ρ=1,相位 = θ_f·p_needle                                                ← 位置也精确
```

**分数、读出、位置三者同时恢复到稠密**,这是"needle 自成簇"这个结构带来的,不是
靠调权重或加修正项凑出来的。代价只有一个槽——haystack 语义高度同质、`K_eff`
小、压得很狠,省下的预算转移给异常点。**这就是为什么自适应预算不是锦上添花而是
唯一机制**：固定均匀预算在信息论上无法同时做到"压缩 haystack"和"保留
needle",因为它不允许预算转移。

**失败模式（必须在 Stage 0 证伪）**：τ 太松 → needle 并入 haystack 簇 → 退回
稀释。τ 是全方案最敏感的超参,S0.3 就是为它设计的。

## 4. 最坏情况的双重兜底

**表现的地板**：内容不可压缩时 `K` 增长 → 槽变多 → 质量向稠密平滑逼近。`K` 撞上
`K_max` 时强制 Ward 合并,把结构推回"少数大簇 + 位置序 ladder",等价 `K_max=1`
锚点,即**今天的 LogKV 行为**。退化的终点是现状,不是比现状更差。

**性能的天花板**：v1/v2 都用硬性 `S_max`（§2.1），占用随内容浮动但上界固定。

| 配置 | 槽数 | 说明 |
|---|---:|---|
| 现有 LogKV recent + B=512 × ~5 级 | ≈ 3584 | 对照基线 |
| SemanticLogKV recent 1024 + K16×B′8×L13 | ≈ 2688 | 预分配上界,典型占用更低 |

> **论文层面最大的攻击面**：自适应预算意味着 niah 变好可能仅仅因为多用了内存——
> 而 niah 恰恰最容易把 K 推高。**从第一个实验开始就必须报告每个 benchmark 上的
> 实际占用槽数**,并准备 memory-matched 对照（把 vanilla 的 B 调大到相同槽数再比）。
> 这一条如果留到最后补,整套实验要重跑。

## 5. 实现方案

### 5.1 新模块 `litgpt/log_kv_position.py`

四个纯函数,无状态、无可学参数、CPU 可测。先写这个模块和它的单测,其余改动全部
依赖它。**v2 更新**：`materialize_slot_key` 不再需要 `beta` 参数（恒定满模长）；
新增 `position_confidence(z_re, z_im)` 返回 `log ρ_f`（供分数侧置信项使用）。

```python
# 单 token 的 z：直接切现成 RoPE cache，零三角函数
def position_stat_from_rope(cos, sin, rope_n_elem):
    """cos/sin: (..., T, rope_n_elem)，前后两半重复
    returns z_re, z_im: (..., T, F) fp32, F = rope_n_elem // 2"""
    F = rope_n_elem // 2
    return cos[..., :F].float(), sin[..., :F].float()


# 加权复数平均：精确、可结合、零累积误差
def merge_position_stat(z_re1, z_im1, w1, z_re2, z_im2, w2):
    tot = (w1 + w2).clamp_min(EPS)
    a, b = (w1 / tot).unsqueeze(-1), (w2 / tot).unsqueeze(-1)
    return a * z_re1 + b * z_re2, a * z_im1 + b * z_im2


# z -> (cos_eff, sin_eff)：v2 恒定满模长（不设 beta），仅用于旋转，不编码置信度
def materialize_rope_factors(z_re, z_im, rho_min=1e-6):
    rho = torch.sqrt(z_re * z_re + z_im * z_im).clamp_min(rho_min)
    ch, sh = z_re / rho, z_im / rho
    return torch.cat([ch, ch], -1), torch.cat([sh, sh], -1)


# 独立的置信度量：喂进分数侧的加法项，不喂进旋转
def position_confidence(z_re, z_im, rho_min=1e-6):
    """returns log rho_f: (..., F)，逐频率，<=0；调用方按需 pool 成标量。"""
    rho = torch.sqrt(z_re * z_re + z_im * z_im).clamp(rho_min, 1.0)
    return torch.log(rho)


# 槽键物化：镜像 model.apply_rope 的数学，去掉 dim==3 约束
def materialize_slot_key(k_raw, z_re, z_im, rope_n_elem):
    cos_eff, sin_eff = materialize_rope_factors(z_re, z_im)
    x   = k_raw[..., :rope_n_elem]      # partial rotary: 尾部内容通道不旋转
    h   = rope_n_elem // 2
    rot = torch.cat((-x[..., h:], x[..., :h]), -1)
    return torch.cat([x * cos_eff + rot * sin_eff, k_raw[..., rope_n_elem:]], -1)
```

**该模块的单测（Stage 0.1，纯 CPU，不依赖 Stage 0 的 dump 结果）**：
- 嵌套精确性：单 token → 与 `model.apply_rope` 输出逐位一致（`ρ=1` 时满模长
  旋转就是恒等变换到标准 RoPE）。
- Dirichlet 闭式解：连续 w-token span 的 `ρ_f` 应等于
  `|sin(wθ_f/2)/(w·sin(θ_f/2))|`，相位等于 `θ_f × span 中心`。
- 合并精确性与结合律：任意二叉合并顺序结果一致；与直接对全体成员求平均一致。
- 数值边界：全相消（`ρ→0`）时不产生 NaN/Inf，`rho_min` 钳住分母。
- **v2 新增**：`position_confidence` 在 `ρ=1` 时返回 0；`ρ→0` 时返回大负数但
  有限；`materialize_rope_factors` 输出的 `(cos_eff, sin_eff)` 在任意 `ρ` 下都
  满足 `cos_eff²+sin_eff²=1`（恒定满模长,与 ρ 无关，这是 v2 相对 v1 的关键
  回归测试）。

### 5.2 `litgpt/log_kv_cache.py`

- 新增 buffer：每槽 `z_re/z_im`（`(B,G,S,F)` fp32）、每槽 `cluster_id`、每簇
  `centroid`（`(B,G,K_max,d)`，L2 归一）与 `cluster_size`。
- 键改存 **pre-RoPE**。recent window 同样存 pre-RoPE + 其 z，读出时统一走
  `materialize_slot_key`——**一条代码路径**，保证嵌套性质在生产里被真实执行而
  不只是单测里。代价是每次读出对窗口做一次逐元素旋转，与已有的 `O(S·d)` matmul
  同量级，可忽略。
- `compact()`/`_binary_carry()` 增加 z 的合并（一行 `merge_position_stat`），
  Σ/Γ 全部改在 **pre-RoPE 内容空间**统计，物化时对 `σu`、`γa` 施加与 `k̄` 相同的
  旋转（rank-1 在线性映射下仍是 rank-1）。**顺带的好处**：Σ 不再混入位置相位
  方差，簇内同质时 rank-1 近似精度应系统性恢复——这是另一分支 D1 自检里
  width≥8 失真的根因之一。
- 新增路由函数（`@torch.no_grad()`，与 cache 更新同路径，训推自动一致——这是
  从另一分支 pin 系列教训里吸取的设计原则，见 §9）：

```python
@torch.no_grad()
def _route_dpmeans(self, k_raw, z_re, z_im):
    """k_raw: (B, G, m, d) 刚离开窗口的 token。m 通常为 2，顺序处理即可。
    聚类在 pre-RoPE 空间进行 —— 簇身份是纯语义的，这只有解耦之后才可能。"""
    for t in range(k_raw.size(2)):
        kt  = F.normalize(k_raw[:, :, t], dim=-1)            # (B,G,d)
        sim = torch.einsum("bgd,bgkd->bgk", kt, self.centroid)
        best, idx = sim.max(-1)
        need_new = (best < self.tau) & (self.cluster_size.gt(0).sum(-1) < self.K_max)
        idx = torch.where(need_new, self._first_free_slot(), idx)
        self._append_to_cluster(idx, k_raw[:, :, t], z_re[:, :, t], z_im[:, :, t])
        self._update_centroid(idx, kt)                       # 在线均值
        # K 满且 best < tau：先 Ward 合并最近两簇腾位，再新建
```

- `get_attention_state()` 额外返回 `(B,G,S)` 有效位掩码；`log_kv_slot_attention`
  增加一个可选的槽有效性掩码参数（fp32 分数上填 `-inf`），与现有 `causal_tail`
  正交；**v2 新增**：分数公式加 `κ·pool_f(log ρ_f,s)` 项，`κ` 走跟
  `second_order_scale` 相同的运行时标量 + CPT warmup 机制。

### 5.3 `litgpt/model.py`

唯一的接口性改动：cache 现在需要 **pre-RoPE 的 k** 和该 token 的 **cos/sin 切片**，
而不是 post-RoPE 的 k。模型本来就同时持有这两样（RoPE 就在这里施加），所以是
传参改动而非新计算。`build_log_kv_cache`/`set_log_kv_cache`/
`enable_log_kv_training` 三处透传新参数。

### 5.4 矩形张量与 batching

不同 (batch, kv-head) 的簇分配不同。v1 的处理方式：
- **按 (B, G) 独立聚类**——不同头关注不同语义，共享簇是错的。metadata 代价
  `K_max × d` per head，可接受。
- **预分配 + 掩码**：槽张量固定为 `K_max × L_max × B′`，占用不满时掩码屏蔽。
  张量始终矩形，不引入 ragged 布局。
- ragged/paged 布局留到工程化阶段，不在研究验证期做。

### 5.5 参数与开关

沿用另一分支已验证的项目惯例：`demo.py`/`eval.py` 走 `run_cli()` 签名自省与
`_o()`，**YAML 新字段默认必须写 `null`**（否则会被同名 CLI 参数静默覆盖，见
§9 的坑清单），不进 `eval.sh`/`majob.sh` 核心列表，走 `DIAG_ARGS` opt-in。

| 参数 | 默认 | 作用 |
|---|---|---|
| `log_kv_semantic_clusters` | `false` | 总开关，关闭时逐字节复现改动前行为 |
| `log_kv_cluster_k_max` | `16` | 簇数硬顶 = 性能天花板 |
| `log_kv_cluster_tau` | `0.6` | DP-means 阈值，最敏感的超参 |
| `log_kv_position_kappa` | `0.0` | `log ρ` 置信项系数，v2 新增，取代 v1 的 `log_kv_position_beta` |
| `log_kv_cluster_entries` | `8` | 簇内每级 entry 数 B′ |

## 6. 实验协议

四个阶段，每阶段带一个可证伪的决策门。顺序不能打乱：Stage 0 几乎不花 GPU，却能
在写任何生产代码之前否掉整个方案。

### Stage 0 — 离线可证伪（1 次 GPU dump + 全部 CPU 分析）

只需要一次前向：对几条 32k 的 NIAH prompt dump 每层每头的 **pre-RoPE k** 与
needle 的 token span（复用另一分支已有的 `log_kv_pin_diag.py` 定位逻辑）。之后
所有分析在 CPU 上做。

| 编号 | 测什么 | 为什么 |
|---|---|---|
| S0.1 | §5.1 单测（含 v2 的满模长回归测试）| 数学正确性，纯 CPU，不需要 dump |
| S0.2 | `K_eff(n)` 随 τ 的曲线，拟合 log / 幂律 / 线性 | 预算故事成不成立 |
| S0.3 | needle 隔离率：所在簇的 `w` 分布 | §3 的核心机制成不成立 |
| S0.4 | 因子化误差 `‖E[Rk]−E[R]E[k]‖ / ‖E[Rk]‖` | §2.2 的代价有多大 |
| S0.5 | `log ρ` 分布：needle 簇 vs haystack 大簇 | 验证置信项确实能区分两者，是 v2 特有的检验 |

**决策门**：
- **S0.2**：32k 下 τ∈[0.5,0.8] 时 `K_eff` 应在 10² 量级。若达到 10³ 以上，预算
  故事垮掉——回去把聚类粒度调粗再谈。
- **S0.3**：needle 落在 `w≤4` 的簇里的比例应显著高于随机基线。若 needle 大多
  并入大簇，§3 的机制不成立，方案应就地停止，不要硬着头皮训。
- **S0.4**：真实簇上的相对误差中位数应 < 0.2。若普遍很大，说明簇不够纯，因子化
  前提不成立。
- **S0.5**：needle 单点簇的 `log ρ ≈ 0`，大 haystack 簇的 `log ρ` 应明显偏负。
  若区分度不够，κ 这个置信项在实践中可能学不出有意义的权重，需要重新设计
  `pool_f` 的汇总方式（比如只用低频段而非全频段）。

### Stage 1 — 实现

按 §5 落地，默认关闭时逐字节复现现状。额外必测：`K_max=1` 且路由永远选中同一个
簇（即退化成单一位置序 ladder）时，在数值上应逼近现有位置分桶实现。

### Stage 2 — eval-time 探测

诚实的预期：新表示对现有 CPT 权重是分布外的，绝对分数可能不好看。价值在于**形状
而非绝对值**，Config A 是硬性正确性闸门。

| 配置 | 参数 | 期望 |
|---|---|---|
| A 正确性闸门 | `K=1`（强制单簇）| ≈ 0.1716 / 0.0827。**对不上就是实现有 bug，不要往下走** |
| B κ 关闭 | `K=16, κ=0` | 只看聚类结构本身的贡献，隔离置信项 |
| C 主实验 | `K=16`，扫 `κ` 和 `τ` | niah 应显著高于 A |

```bash
DIAG_ARGS="--log_kv_semantic_clusters true --log_kv_cluster_k_max 16 \
  --log_kv_position_kappa 0.1 --log_kv_cluster_tau 0.6" \
    bash majob.sh exp/qwen1.7b-32k/arc_warmup.yaml
```

沿用另一分支 §8.5 记录的坑：主 benchmark 参数不要传 `none`（LongBench 来自主
列表）；多配置串行跑（输出目录不按超参分开，靠 JSON 内的 metadata 区分）；新
参数需要一并写进 eval 的 metadata 字段。

### Stage 3 — CPT

Stage 2 有信号后再投入。`κ` 走跟 `second_order_scale` 相同的爬坡机制（目标值别
设太激进、warmup 别只给 10 步——这是另一分支 §4 真实踩过的训练崩溃教训）。

## 7. 消融表

每一行回答一个独立问题，不是穷举网格。所有行都必须附带实际占用槽数（§4）。

| 旋钮 | 取值 | 回答的问题 |
|---|---|---|
| `K_max` | 1 / 4 / 16 / 64 | 语义分组本身值多少分？1 是现状锚点 |
| `κ` | 0 / 0.05 / 0.1 / 0.2 | 置信项的边际价值，0 是"只做聚类不做置信"的对照 |
| `τ` | 0.4 – 0.85 | needle 隔离与簇纯度的平衡点 |
| `λ`（mass bias）| 0 / 1 | 大簇的计数质量补偿是否仍然正确 |
| rank-1 Σ/Γ | 开 / 关 | 解耦后二阶修正是否还有边际价值 |
| v1 β 插值（对照）| 0 / 0.5 / 1 | **验证 v2 确实优于 v1**，应作为论文里的正式消融 |
| vanilla memory-matched | B 调大到同槽数 | **排除"只是多用了内存"** |

指标沿用现有四项（ACC/LongBench/LongBench_e/niah@32768），**另加 multi-needle**——
单 needle 一旦从 0.08 提上去就会迅速失去区分度，multi-needle 才是这类分层压缩
方法真正拉开差距、也最被审稿人看重的实验。

## 8. 风险与对策

| 风险 | 后果 | 对策 |
|---|---|---|
| **τ 敏感** | needle 并入大簇，机制失效 | S0.3 提前证伪；τ 进主扫描；v2 可做 τ 反馈控制器（带迟滞防振荡）|
| **同簇远距离成员相位混叠**（v2 依然存在，§2.3 已如实标注）| 虚构的"平均位置"，位置敏感任务可能受影响 | 簇内 ladder 降低发生频率；multi-needle 实验里专门观察 |
| **centroid drift** | 早期成员的归属基于后来会漂移的 centroid | 在线聚类固有近似，接受并记录；周期性重聚类是 v2 之后的选项（摊还 O(S)）|
| **内存不对齐比较** | 论文被质疑"只是多用了内存" | 见 §4，从第一个实验起报告槽数 + memory-matched 对照 |
| **存储增加约 40%** | z 是每槽 `rope_n_elem` 个 fp32 | 不改变渐近（槽数仍 O(log n)）；先 fp32 保精度，实测后再评估 bf16 |
| **训练期显存随内容波动** | batch 内簇占用不同 | `K_max` 预分配上界固定，训练期必须开硬顶 |
| **κ 需要 CPT 才能校准** | eval-time 直接开可能表现不稳 | 与 `second_order_scale` 同一套已验证的 warmup 机制，见 §6 Stage 3 |

## 9. 复用另一分支的基础设施与教训（精简版）

只保留对本分支**操作上仍然生效**的结论，不重复完整叙事（完整版在
`claude/semantic-cluster-log-compression-8ai32a` 分支的 CLAUDE.md）。

**RoPE 布局约定**：`apply_rope`（`litgpt/model.py:2074`）是"前后半重复"布局，
`cos[f]==cos[f+d/2]`，partial rotary 只旋转前 `rope_n_elem` 维、尾部内容通道
不旋转。§2.2/§5.1 的 z 统计量实现直接依赖这个约定。

**pin 系列四条死因（设计规避清单，本方案逐条对照）**：
1. 训推不一致——pin 选择只在推理期跑（`@torch.no_grad()`，训练从未见过）。
   本方案：路由同样 `@torch.no_grad()`，但训练和推理走**同一条** cache 更新
   路径，从设计上不存在"训练没见过"的问题。
2. 分数尺度失配——pin 槽是精确 key，混进池化槽的 softmax 行时点积尺度不同，
   引发异常塌缩。本方案：槽种群完全同质（都是池化槽,包括单点簇),不存在
   两种尺度混合。
3. 观察窗口的架构性局限——pin 依赖一次性 dense prefill 和"prompt 尾部是问题"
   两个假设，长上下文/多轮场景下不成立。本方案：路由在 flush 时逐 token 发生，
   不需要看到整段 prompt，也不依赖任何位置性假设。
4. 剂量效应——pin 插得越多伤害越大。本方案：不新增槽种类，只改变成员划分，
   没有"插入精确槽"这个操作。

**工程习惯**：
- `eval.sh` 把 YAML 展平成纯 CLI flag（不传 `--config`），`_o()` 的"YAML 非
  null 覆盖 CLI"逻辑不会触发；直接用 `--config <yaml>` 会触发。新增会被扫参
  覆盖的字段，YAML 里必须写 `null`。
- `majob.sh` 如果 `save_path` 下已有 checkpoint 会跳过训练直接 eval——新实验
  必须用独立 `save_path`，不要复用另一分支的目录。
- 本地跑 pytest 用 conda env `mineru`（`/Users/hourunli/anaconda3/envs/mineru`，
  Python 3.12）；系统自带 Python 无法解析仓库里到处用的 `X | None` 类型注解。

## 10. 相关工作定位（2026-08 快照，务必用 search 复核最新情况）

| 工作 | 做了什么 | 与本方案的区别 |
|---|---|---|
| SemantiCache (arXiv 2603.14303) | 按分隔符切语义块 + 贪心种子聚类合并 | 固定 budget，无簇内分辨率层级，frozen model |
| Multipole Attention (NeurIPS 2025, arXiv 2506.13059) | key 做 k-means，远处用 centroid 近似打分，越远越粗 | **保留全量 KV（O(n) 显存）**，centroid 只是索引，不是压缩 |
| SeKV (arXiv 2606.31145) | entropy 引导语义 span，GPU summary + CPU SVD 按需重建 | offload + 检索系统，什么都不丢，非次线性 |
| ClusterAttn (ACL 2025) | 密度聚类自适应簇数 | 稀疏注意力，固定 1024 token 预算 |
| GVote (ICLR 2026, arXiv 2509.03136) | 蒙特卡洛采样未来 query 免手工设预算 | eviction/selection 框架，预算仍是 O(1) 固定池 |

空着的生态位：**非参先验决定簇数 + 严格次线性空间 + 簇内分辨率层级 + 特征函数式
抽象位置**。CRP 的集中不等式给了这条线一个别人写不出的理论节。搜索这几个撞车
候选时没有找到任何工作用非参贝叶斯给簇数一个先验、同时保持严格次线性显存。

## 11. 变更记录

> 每次讨论产生突破或进展,在这里加一条,新的在最上面。只记"改变了什么结论/设计",
> 不重复已经写进正文的细节——细节改到对应章节,这里留指针和一句话动机。

- **2026-08-14｜v1 → v2：位置重物化从 β 插值旋钮改为满模长旋转 + log ρ 独立
  置信项。** 动机：用户指出 v1 的 z 统计量本身仍是加权和，天然带相位相消，
  质疑"是否真的解决了相位问题"。分析发现 v1 的 `gain=ρ^{(β-1)}` 在 β=0 时对
  最不可信的频率通道（ρ 最小）施加最大重物化力度，是方向性错误。修法：利用
  "圆均值相位是原则性点估计,与 ρ 无关"这一统计学事实,把旋转固定为满模长
  （不再衰减内容），把 ρ 携带的不确定性改走一条独立加法项进分数（`κ·logρ`，
  与现有 `log(w)` mass bias 同源同权）。详见 §2.3。需要更新的下游章节
  （§5.1/§5.2/§5.5/§6/§7）已在本次一并同步。
- **2026-08-14｜方案定型：语义簇 + z 统计量,取代最初讨论的"连续性约束 Ward
  合并"方案。** 动机：用户明确偏好语义簇路线的"优美性"，并提出用簇级抽象
  position embedding 解决 RoPE 相位相消问题。确定 Ward 合并降级为簇内 ladder
  的可选合并顺序策略（正交、可叠加，非互斥）。
- **2026-08-14｜建档。** 从 `claude/semantic-cluster-log-compression-8ai32a`
  分支切出 `semanticLogKV`，本文件取代该分支的 CLAUDE.md 作为独立存档，只保留
  与语义簇方案直接相关的内容。
