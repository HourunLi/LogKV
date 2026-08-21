# SemanticLogKV Position Embedding 手册

> 本文件是 position/RoPE 专题手册，记录本分支关于"压缩 entry 如何携带位置信息"的
> 设计动机、当前方案、实现接口、验证计划和外部研究脉络。算法落地细节仍以
> [`algorithm-spec.md`](algorithm-spec.md) 的 §5.10、§5.11、§5.14、§5.21 为准；
> 本文负责解释为什么这样做，以及后续如何继续调研和证伪。
>
> 当前结论：v1 主线采用 **pre-RoPE 内容均值 + `(p_lo, p_mid, p_hi)` 锚点展开**。
> 它是一个 training-free、RoPE-compatible、可解释的 baseline。它不是最终答案；
> 后续可以比较 learned anchor / learned bias / decoupled position branch 等更强方案。

## P0. 一句话概览

SemanticLogKV 的 entry 是**存储单位**，attention 看到的是**读出时物化的虚拟 slot**。

```
entry:
  k_raw_mean, v_mean, w, p_lo, p_hi, sum_wp

readout:
  p_mid = round_half_up(sum_wp / w)
  anchors = dedup([p_lo, p_mid, p_hi])
  entry -> M 个 virtual slots, M ∈ {1,2,3}

slot a:
  key   = apply_rope(k_raw_mean, anchor_a)
  value = v_mean
  bias  = λ · log(w / M)
```

这等价于说：**一个 compressed entry 在 attention 计算时最多变成 3 个位置不同、内容相同、
value 相同、mass 被摊薄的 virtual slots**。读出端仍然是扁平 slot attention，不感知
cluster / segment / ladder。

## P1. 要解决的问题

### P1.1 RoPE 和均值池化天然冲突

RoPE 把位置写成对 key/query 子空间的旋转。对单个 token，attention score 里出现的是：

```
q(p_q) · k(p_k) = R(p_q)q_raw · R(p_k)k_raw
```

它的好处是相对位置以旋转相位差的形式进入点积；坏处是，**不同位置的 post-RoPE key
不能随便求均值**。

旧 LogKV 的位置分桶会把一段 token 的 post-RoPE key 直接平均：

```
k_slot = mean_i R(p_i) k_i
```

如果 `p_i` 分散，高频 RoPE 通道的相位会在圆上互相抵消。这个平均向量通常不再等价于
任何一个合法位置上的 `R(p*) k*`。结果是：

- 位置语义被抹平，尤其宽 slot 上高频位置分辨率下降；
- content 和 position 被混在同一个均值里，后续无法单独修正；
- needle 一旦被合进宽 slot，分数和读出都会被稀释；
- 二阶修正的秩被"位置相位方差"消耗，留给真实内容方差的表达能力变少。

### P1.2 语义簇会让位置问题更尖锐

按语义聚类后，同一 cluster 的成员可能在全文中多次复现：

```
Alice ... 4k tokens ... Alice ... 8k tokens ... Alice
```

如果只看语义，这些 token 很可能属于同一个 cluster；如果不加时序约束，它们在同一个
entry 里合并后，`p_lo` 到 `p_hi` 的跨度可能覆盖大半篇文档。于是"语义同质"解决了内容
异质性，但会放大位置跨度。

所以 v3.1 把结构拆成两层：

- **cluster**：语义身份，一个 centroid，只用于路由；
- **segment**：同一个 cluster 内不被时序打断的一段连续访问，只约束低层合并；
- **ladder**：每个 cluster 一条，entry 真正存储在这里。

segment 的作用不是给每段单独建 ladder，而是通过对齐填充保护低层不跨段合并：

```
不填充:  [a1][a2][a3] | [b1][b2]  -> (a3,b1) 跨段
填充后:  [a1][a2][a3][pad] | [b1][b2] -> (a3,pad), (b1,b2)
```

`ℓ_block` 决定保护到第几层；更高层仍允许跨 segment 合并，这是预算压力下的有意降级。

## P2. 设计目标

当前 position 方案必须同时满足五个目标：

1. **RoPE-compatible**：物化出来的 key 必须像普通 token key 一样，由标准 `apply_rope`
   在某个确定位置上得到。不要发明 frozen model 没见过的新位置编码。
2. **training-free first**：v1 不依赖上游 CPT/finetune，先证明结构本身能不能赢。
3. **不平均相位**：禁止用 post-RoPE key 的均值来代表一组位置分散的 token。
4. **扁平读出**：`log_kv_slot_attention` 只看 slot 池和 mask，不读 cluster/segment。
5. **可测可证伪**：每个设计选择都必须落到 Stage 0/Stage 1 可测指标，而不是只靠直觉。

这几个目标会约束方案形态：我们可以在 entry 里维护位置统计，但真正喂给 attention 时，
最好还是转成标准 RoPE key。

## P3. 设计迭代记录

### P3.1 v1：复数统计量 `z` + 标量 `β`

早期想法是为每个 RoPE 频率维护：

```
z_f = Σ_j w_j · exp(i θ_f p_j) / Σ_j w_j
```

`|z_f|` 表示该频率下位置分布的相干性，`arg(z_f)` 表示平均相位。然后用一个标量 `β`
控制"重物化"力度。

问题：

- `z` 本质仍是相位平均，成员位置一散就会相消；
- `β` 把"往哪转"和"多可信"耦合在一个标量里；
- 某些设定下会对最不可信的频率通道施加最大修正，方向错误。

### P3.2 v2：`z` + 满模长旋转 + `log ρ`

v2 拆出置信度：

```
ρ_f = |z_f|
```

读出时用满模长旋转表示方向，再用 `κ · log ρ` 报告该频率是否可信。

进步：不再把不确定性直接塞进内容向量。

问题：`z` 仍然来自相位求和。它只是诚实报告"相消发生了"，不能结构性避免相消。

### P3.3 v3：真实锚点集 `(p_lo, p_hi, p_mid)`

v3 的关键转变是：**不把多个位置压成一个复数或一个 embedding**。entry 只维护整数锚点：

```
p_lo   = 覆盖范围内最早真实位置
p_hi   = 覆盖范围内最晚真实位置
sum_wp = Σ_j w_j · p_j
p_mid  = round_half_up(sum_wp / w)      # 读出时才算
```

读出时，在这些确定位置上分别调用标准 `apply_rope`。

### P3.4 v3.1：cluster/segment 分离

v3 曾经把"时序被打断"实现成 fork 新 cluster。反例是高频复现实体：同一个 Alice 出现
50 次就可能 fork 50 个语义相同的 cluster，K 跟话题切换次数走，预算会炸。

v3.1 改成：

- 时序被打断：同 cluster 开新 segment；
- segment 不分配独立 ladder；
- 对齐填充只保护低层合并；
- 每个 cluster 仍然只有一条 ladder。

位置方案本身仍是 `(p_lo, p_mid, p_hi)`，但靠 segment 降低 entry 的早期跨度。

## P4. 当前 entry 里怎么维护位置

每个 entry 至少有这些字段：

```
k_raw_mean   # pre-RoPE key 内容均值
v_mean
w            # 真实 token 质量，不含 pad
p_lo
p_hi
sum_wp       # int64，未归一化位置加权和
```

合并两个相邻 entry `A, B` 时：

```
w       = w_A + w_B
k_raw   = (w_A · k_A + w_B · k_B) / w
v       = (w_A · v_A + w_B · v_B) / w
p_lo    = min(p_lo_A, p_lo_B)
p_hi    = max(p_hi_A, p_hi_B)
sum_wp  = sum_wp_A + sum_wp_B
```

`p_mid` 不在 compact 时存成最终值，而是在读出时由 `sum_wp` 计算：

```
p_mid = clamp((2 · sum_wp + w) // (2 · w), p_lo, p_hi)
```

这个 round-half-up 的整数实现有两个理由：

- `sum_wp` 可结合、可重放，不受合并树形顺序影响；
- 只在读出时除一次，避免层层合并积累舍入误差。

### P4.1 为什么不用 `p_mid = 权重大的一侧`

早期曾经考虑过：

```
p_mid = A.p_mid if w_A >= w_B else B.p_mid
```

这不满足结合律，而且在 Fenwick 平衡合并里几乎恒等退化成 `p_mid == p_lo`：因为每次
compact 往往合并等宽块，`w_A == w_B` 时总选左边。这样第三锚点没有信息量。

所以当前定案是 `sum_wp` 累加，读出时一次性算质心锚点。

## P5. attention 里怎么用这些位置

### P5.1 entry -> virtual slots

一个 entry 读出时会先得到候选锚点：

```
anchors_raw = [p_lo, p_mid, p_hi]
anchors = dedup(anchors_raw)
M = len(anchors)
```

然后展开成 `M` 个 virtual slots：

```
for p in anchors:
    slot_k = apply_rope(k_raw_mean, p)
    slot_v = v_mean
```

实际打分：

```
score_{s,a} = scale · (q · apply_rope(k_raw_s, p_a)) + λ · log(w_s) − log(M_s)
value_{s,a} = v_s
```

其中：

- `s` 是 entry；
- `a` 是该 entry 的第几个 anchor；
- `M_s` 是去重后的 anchor 数；
- `q` 已经是正常 post-RoPE query；
- `k_raw_s` 是 pre-RoPE 内容均值；
- `value` 不随 anchor 改变。

### P5.2 为什么必须减去 `log(M)`，而且不能挂在 `λ` 门控之内

如果一个 entry 展开成 3 个 slot，而每个 slot 都用 `λ·log(w)`、不做任何抵消，
softmax 里该 entry 实际拿到的质量会被放大 `M^{1-λ}` 倍（`λ=1` 时这个因子恰好是 1，
不容易被发现；`λ=0` 是最坏情形，退化成整整 `M` 倍——大跨度 entry 因为锚点多，白拿
额外注意力质量，且这个膨胀量和内容/权重完全无关，纯粹是"被拆成几个虚拟槽"这个存储
细节的副产品）。

早期版本把修正写成 `λ·log(w/M)`（等价于 `λ·log(w) − λ·log(M)`），这只在 `λ=1`
时才精确抵消，`λ` 一旦不是 1（尤其 `λ=0` 消融档，`experiments.md` §7 计划扫）
`M` 就会重新泄漏进总质量。正确写法是把抵消项挪到 `λ` 的门控**之外**：

```
λ · log(w) − log(M)
```

`λ` 只控制原始 `log(w)` 那部分（继承自 vanilla LogKV 的 mass bias，可以整体关掉），
`−log(M)` 永远无条件生效，只负责抵消锚点展开本身造成的候选膨胀。这样
`Σ_a exp(λ·log(w) − log(M)) = w^λ`，对**任意** `λ` 精确成立，不只是 `λ=1`。
这不是调参项，而是正确性修正。

### P5.3 读出伪代码

```
def materialize_entry(entry):
    if entry.w == 0:
        return []                       # pad/dead entry 不进 attention

    p_mid = round_half_up(entry.sum_wp / entry.w)
    p_mid = clamp(p_mid, entry.p_lo, entry.p_hi)

    anchors = dedup([entry.p_lo, p_mid, entry.p_hi])
    M = len(anchors)

    slots = []
    for p in anchors:
        slots.append({
            "k": apply_rope(entry.k_raw_mean, p),
            "v": entry.v_mean,
            "bias": lambda_mass * log(entry.w) - log(M),
        })
    return slots
```

GPU 实现不能真的生成 ragged list；`dedup_anchors` 应该返回固定 3 槽的矩形张量和
`slot_valid` mask。无效 anchor 必须覆写成安全哨兵位置（如 0）后再进 gather，不能让
`INT_MAX` / `-1` 参与 `cos_cache` 索引。

### P5.4 二阶统计如何接入

如果保留 `Σ/Γ` rank-1 修正，方向向量也必须和 `k_raw_mean` 一样在 anchor 位置上旋转：

```
sigma_u_eff = apply_rope(sigma_u_raw, p)
gamma_a_eff = apply_rope(gamma_a_raw, p)
gamma_b     = 不旋转，value 空间没有 RoPE
sigma2/gamma = 标量，不旋转
```

原因：`sigma_u` / `gamma_a` 属于 key 空间方向，query 与它们点积时应该处在同一个
RoPE 坐标系。value 空间从不做 RoPE。

## P6. `p_lo / p_mid / p_hi` 各自的直觉作用

### P6.1 `p_lo`

`p_lo` 是 entry 覆盖范围的最早位置。它保护"查询需要命中 span 左边界"的情况，例如：

- 一个信息块开头的定义；
- segment 开始处的实体名；
- needle 恰好落在被合并块的左侧。

### P6.2 `p_hi`

`p_hi` 是 entry 覆盖范围的最晚位置。它保护：

- 后续更正、最近提及；
- span 右边界的引用；
- decoder 对近期上下文偏好的相对位置模式。

### P6.3 `p_mid`

`p_mid` 是按 token 质量加权的位置质心。它代表"这个 entry 的平均位置"，用于覆盖查询
命中 span 中部的情况。和 `p_lo/p_hi` 不同，`p_mid` 不要求是真实 token 位置；它只需要是
一个确定整数位置，因为 RoPE 只关心在某个坐标上旋转。

### P6.4 三锚点不是恢复原分布

`p_lo/p_mid/p_hi` 不是原始 token 位置分布的充分统计量。它只是一个低成本近似：

- 对窄 entry：通常退化成 1 个或 2 个 anchor，几乎无开销；
- 对中等跨度 entry：边界 + 质心比单点更稳；
- 对超大跨度 entry：3 个点可能远远不够，需要靠 segment、`ℓ_block`、或更强替代方案。

所以这套方案必须用 Stage 0/Stage 1 证伪，而不能只凭直觉定死。

## P7. segment 与 position 的关系

segment 不直接出现在 attention 中；它通过影响 ladder 的合并边界，间接改善 entry 的
位置跨度。

### P7.1 `ℓ_block`

`ℓ_block` 表示 segment 边界保护到第几层：

```
保护到第 ℓ 层 => 新 segment 开始前，把 level 0 逻辑插入流的相位对齐到 2^ℓ 的倍数
count = (-level0_phase[cluster]) mod 2^ℓ_block
```

`count` 个 `w=0` pad 只插入 level 0，然后靠普通 binary carry 自然上升。

> **不能用 `n_total_c`（该簇累计的真实 token 数）算这个余数**——`n_total_c` 只数
> 真实 token，不含已经插入过的 pad，而这里要对齐的是 level 0 上"真实 token + pad
> 混合而成的插入流"，两者从第一次填充起就会分叉（具体反例见
> [`algorithm-spec.md`](algorithm-spec.md) §5.11 的更正框）。
>
> **也不能用 `level_count[cluster, 0]`（该簇 level 0 当前的真实占用数）**——
> 早期版本这样做过，在"level 满 `B′` 个后一次性清空/重置"这个进位模型下是对
> 的；但 `algorithm-spec.md` §5.12 把逐 token 进位精确定义成"每次溢出只合并
> 最老的两个、留下 `B′-1` 个 + 新到的 1 个"之后，`level_count` 只在 `B′-1`/
> `B′` 两个值之间永久振荡，不再遍历 `mod 2^ℓ_block` 的全部剩余类，这条等价
> 关系随之失效（具体反例同样见 `algorithm-spec.md` §5.11 的更正框）。
>
> **正确的量是独立的持久相位计数器 `level0_phase[cluster]`**——不依赖
> `level_count` 如何振荡、不依赖 `B′` 与 `2^ℓ_block` 的整除关系，只是老实地
> 数"该簇 level 0 逻辑插入流（真实 token + pad 混合计数）目前的相位"：每插入
> 1 个 entry（不论真实还是 pad）就 `+1 mod 2^ℓ_block`；新段事件（`NEW_
> CLUSTER`/`NEW_SEGMENT`）落地后重置为 `1 % 2^ℓ_block`，不是"先 pad 归零、
> 再套用逐 token 的 `+1` 公式"——这两条规则共享同一个 `prev_mod`/`count`
> 输入但不是同一条公式，混用会在 pad 之后把相位算大（`algorithm-spec.md`
> §5.11 有具体反例）。这里只复述结论，`algorithm-spec.md` 才是权威定义，两处
> 一旦不一致，以那边为准。

### P7.2 保护不是永久隔离

低层不跨段合并，高层仍可能跨段：

```
ℓ_block = 1: 保护 w=2 以内
ℓ_block = 2: 保护 w=4 以内
更高层: 预算需要时允许跨 segment
```

这符合 v3.1 的定位：segment 是低层局部性保护，不是永久分区。

### P7.3 position 方案依赖 segment 降低跨度

`p_lo/p_mid/p_hi` 的可行性高度依赖 entry 的跨度分布。如果纯语义 cluster 导致某些 entry
经常跨几千 token，那么三个 anchor 很可能不够。segment 的价值就是把"最容易出错的早期
合并"限制在局部连续段内，让三锚点方案的误差进入可控范围。

## P8. 现在必须调研/验证的问题

这是本文件建档时用户提出的核心问题集合。

### P8.1 `p_lo / p_mid / p_hi` 是否真的有效

要回答的不是"这三个点听起来合理吗"，而是：

- 相比 `p_mid only`，`p_lo+p_hi` 或三锚点是否显著降低 attention 误差；
- `p_lo/p_hi` 是否主要帮助 needle / 边界信息；
- `p_mid` 是否主要帮助普通连续内容；
- 三锚点带来的 `E[M]` slot 膨胀是否值得；
- 大跨度 entry 上三锚点是否仍然过粗。

### P8.2 是否有更好的 position 表示

候选方向：

| 方案 | 说明 | 优点 | 风险 |
|---|---|---|---|
| `p_mid only` | 每 entry 一个 anchor | 最省 slot | 边界位置损失大 |
| `p_lo+p_hi` | 只保留边界 | 比三锚点省，保护 span 端点 | 中部命中弱 |
| `p_lo+p_mid+p_hi` | 当前主线 | training-free，兼顾边界和中心 | 最坏 `M=3` 开销 |
| quantile anchors | 维护多个位置分位数 | 更接近位置分布 | 合并时维护复杂 |
| medoid anchors | 选真实成员位置代表 | 锚点都是真实 token | 合并后需保存候选 |
| frequency-aware anchors | 高频少展开/低频多展开或反过来 | 针对相位抵消来源 | 实现和解释复杂 |
| learned anchor weights | 固定 anchors，学习每个 anchor 的 bias/权重 | 小改动，可 CPT | frozen 下不可用 |
| learned scalar position | 从 entry stats 预测 `p_hat`，仍用 RoPE | 保持 RoPE-compatible | 需要训练，可能越界 |
| decoupled position branch | 内容压缩和位置通道分开 | 表达力强 | 架构改动大，需训练 |
| learned compactor | Perceiver/adapter 直接生成 compact KV | 最灵活 | 训练成本高，解释性低 |

### P8.3 语义 cluster 论文如何使用位置信息

调研口径：

- 有些工作把 cluster 只作为**召回索引**，cluster centroid 用来找相关 token，最终读出的
  仍是原始 token KV。这类方法不需要给 cluster 本身发明一个位置。
- 本项目不同：cluster 内 entry 要直接作为 compressed slot 参加 attention，因此必须给
  compressed entry 一个 RoPE-compatible 的位置表示。

这意味着不能简单照搬"semantic KV clustering"论文里的 centroid 逻辑。那些 centroid
通常解决的是检索问题，不解决"一个均值 entry 如何像 token 一样被 RoPE query 读懂"。

### P8.4 能否学习一个 cluster-level/token-like position

可以，但要区分三种强度。

**弱学习：learned anchor bias**

保留合法 RoPE 坐标，只学习每个 anchor 的额外 logit bias：

```
score_{s,a} = q · R(p_a)k_s + λ log(w_s) − log(M_s) + b_a(entry_stats)
```

**`− log(M_s)` 不能丢，且不能挂在 `λ` 门控之内**——P5.2 已经证明这不是调参项，
是正确性修正：一个 entry 展开成 `M_s` 个 anchor 时若每个都用 `λ·log(w_s)`、不做
任何抵消，softmax 里的总质量会被放大 `M_s^{1-λ}` 倍（`λ=0` 时退化成整整 `M_s`
倍）。`b_a(entry_stats)` 是在这个已经修正过的 mass bias 之上**额外**学到的一项
logit 偏置，不是用来替代 `− log(M_s)` 的——两者共存，`b_a` 学的是"这个 anchor
除了计数质量之外还应该多一点/少一点可信度"，不应该、也学不出"减去 `log(M_s)`"
这件事本身（`b_a` 只吃 `entry_stats`，不知道其它 anchor 的存在，没有信息量去
推导一个跨 anchor 的归一化项）。

这最安全，因为 key 仍在模型熟悉的 RoPE 坐标上。

**中学习：learned continuous/scalar anchors**

小模块输出一个或多个位置：

```
p_hat_a = f(entry_stats)
key_a = apply_rope(k_raw, p_hat_a)
```

如果要支持非整数位置，需要用 RoPE 频率公式直接算 `cos(θ p_hat)` / `sin(θ p_hat)`，而不是
索引离散 `cos_cache`。这仍然是 RoPE-compatible，但需要训练。

**强学习：learned positional key branch**

把内容 key 和位置 key 解耦：

```
key_eff = concat(content_key_compressed, position_key_learned)
```

这接近 MLA/latent attention 的方向，表达力强，但 frozen 模型不会天然理解新通道。
除非做 CPT/finetune，否则不适合作为 v1。

当前建议：v1 不上 learned position；先把三锚点作为可解释 baseline 跑穿，再决定是否
引入学习组件。

## P9. 外部研究脉络

### P9.1 RoPE 本身

[RoFormer](https://arxiv.org/abs/2104.09864) 提出 Rotary Position Embedding。它的关键
性质是把绝对位置编码为旋转，并让 self-attention 中的点积天然依赖相对位置。这正是
SemanticLogKV 必须保持 RoPE-compatible 的原因：frozen model 已经学会读这种几何。

### P9.2 semantic clustering KV cache

[ClusterKV](https://arxiv.org/abs/2412.03213) 代表了"在 semantic/key 空间组织 KV cache"
这一类方向。它的重点是用 cluster 提高 recallable compression：query 先和 cluster
centroid 交互，再取回相关 token/cache 内容。对本项目的启发是：cluster 可以作为路由和
召回单位；但它通常不需要把一个 cluster centroid 当作带位置的 attention slot。

本项目的不同点是：entry 会被压缩成少数 slots 直接喂 attention，因此 cluster/entry 的
position 表示必须单独设计。

### P9.3 decoupled position / latent attention

[DeepSeek-V2](https://arxiv.org/abs/2405.04434) 的 MLA 路线把 KV 压缩和 RoPE 位置通道
解耦，是一个重要信号：高效 KV 表示和 RoPE 位置几何之间确实存在结构性冲突。它说明
"内容压缩"和"位置表达"最好不要强行绑在同一个 post-RoPE 均值向量里。

### P9.4 learned compaction

[Still](https://arxiv.org/abs/2606.07878) 这类 neural KV compaction 方法说明，压缩 KV
可以交给小型学习模块完成。对本项目的启发不是立刻上 learned compactor，而是把它作为
v2/v3 替代路线：如果 training-free anchors 的误差上限过低，可以学习 anchor 选择、
anchor bias，甚至学习 compact KV。

### P9.5 RoPE-aware / frequency-aware KV compression

[KV-Latent](https://aclanthology.org/2025.acl-long.77/) 等工作关注 RoPE 频率与 KV 压缩的
关系。它们提醒我们：相位抵消主要发生在高频通道，不同频率对压缩误差的敏感性不同。

可转化成 SemanticLogKV 的消融：

- compressed slots 是否应该只保留低频 RoPE；
- `p_lo/p_mid/p_hi` 是否应该按频率选择不同 anchor；
- 高频是否更需要边界锚点，低频是否用 `p_mid` 就够；
- 锚点数是否应该由 entry 的频率相干性或跨度自适应决定。

## P10. 可行性验证计划

### P10.1 离线 position 诊断

在 Stage 0 dump 里额外记录/计算：

| 指标 | 目的 |
|---|---|
| entry span = `p_hi - p_lo` | 判断三锚点面对的跨度分布 |
| `E[M]` | 估算去重后平均每个 entry 需要多少不同 anchor；它衡量的是 fixed-3 展开里有多少可被未来 gather/packed 省掉的成本，不是"anchor 越少信息越好" |
| anchor 去重率 | 判断 `p_mid` 是否常常提供新信息；anchor 多通常表达力更强，但也意味着读出宽度更贵 |
| per-frequency phase coherence `ρ_f` | 量化相位抵消强度 |
| dense score vs anchor score 误差 | 直接测位置近似对 attention logits 的影响 |
| dense attention KL | 测 softmax 分布偏移 |
| output L2 / cosine | 测最终读出误差 |
| needle 所在 entry 的 anchor 覆盖情况 | 看 needle 是否靠边界锚点获益 |
| segment 边界污染率 | 看 `ℓ_block` 是否真的压住低层跨段 |

### P10.2 anchor 消融

至少比较：

```
A0: old post-RoPE mean
A1: p_mid only
A2: p_lo + p_hi
A3: p_lo + p_mid + p_hi
A4: p_lo + p_hi + learned/fixed bias
A5: quantile/5-anchor 上界
```

如果 `A3` 明显优于 `A1/A2` 且 `E[M]` 没顶到 3，三锚点主线成立。

如果 `A3` 只比 `A2` 小幅提升，考虑砍掉 `p_mid`，把 slot 预算还给更多 entry。

如果 `A3` 仍远低于 `A5` 或 dense，上 learned/adaptive anchors。

### P10.3 segment 消融

扫 `(g_max, ℓ_block)` 时，不只看 benchmark，还要看：

- 平均每 cluster 有多少 segment；
- padding 占总 entry 比例；
- 低层跨段合并率；
- entry span 分布是否显著收缩；
- 三锚点误差是否随 `ℓ_block` 改善。

如果 `ℓ_block=0` 和 `ℓ_block=1/2` 在 position 误差上差别很小，说明 segment 保护可能
不值得；如果差别很大但 padding 爆炸，需要考虑替代存储层（如贪心相邻合并）。

### P10.4 learned position 的进入条件

只有在下面条件满足时才考虑 learned position：

- training-free 三锚点在 position 误差上明确成为瓶颈；
- 误差主要来自位置而不是 value 均值/supersession；
- `E[M]` 或 entry budget 已经不允许继续加 anchors；
- 有 CPT/finetune 预算，且能保证推理路径不会偏离训练路径。

优先顺序：

```
learned bias over fixed anchors
-> learned mixture over candidate anchors
-> learned continuous anchors
-> decoupled positional key branch / learned compactor
```

## P11. 实现注意事项

1. `k_raw_mean` 必须是 qk-norm 后、apply_rope 前的 key。当前旧 LogKV 用 post-RoPE key，
   SemanticLogKV 必须拆出 raw/roped 两份路径。
2. `p_lo/p_hi/sum_wp` 必须随 entry 走 ladder 合并，不能只存在 cluster 级。
3. `p_mid` 只在读出时算，不要在 compact 时逐层舍入。
4. `dedup_anchors` 必须先过滤 `w=0` pad/dead entry，再处理 `[lo, mid, hi]` 内部去重。
5. `λ·log(w) − log(M)` 是必须项，不是可选优化——`−log(M)` 必须挂在 `λ` 的
   门控之外无条件生效，不能挂在 `λ` 门控之内（P5.2）。
6. 二阶统计里的 key-space 方向要跟 anchor 一起 RoPE，value-space 方向不 RoPE。
7. `segment` 不进入 attention，只影响低层 compact 边界。
8. learned position 如果使用连续位置，不能走离散 `cos_cache[p]`；需要直接按频率算
   `cos(θp)` / `sin(θp)`。

## P12. 当前决策与未决项

### 已定

- v1 采用 `p_lo/p_mid/p_hi` 三锚点读出。
- entry 持久存储不复制成 3 份；只在 attention 读出时虚拟展开。
- mass bias 用 `λ·log(w) − log(M)`，`−log(M)` 不受 `λ` 门控。
- segment 只通过 padding/`ℓ_block` 保护低层合并，不拥有独立 ladder。
- learned cluster-level position 不进 v1 主线。

### 未决

- `p_mid` 是否值得保留，还是 `p_lo+p_hi` 足够。
- `E[M]` 首轮实测约 2.0：三锚点没有退化成"几乎每个 entry 都需要 3 个不同锚点"，
  但 fixed-3 物理宽度仍明显超 vanilla，尤其 `lambda_rel=0.875` 的高召回配置更贵；
  现在先跑 `K_max` + Ward clipped S0.6，确认生产预算下的 entry/fixed-3 成本和绑定率，
  再用真实 anchor-score 消融判断这笔表达力是否值回成本。
- `lambda_rel=1.0` 与 `0.875` 的最终选择：首轮 unclipped 结果支持 `1.0` 作默认主线、
  `0.875` 作高召回候选；进入真实实现前必须看 S0.3 clipped gate 是否出现
  `K_max` 绑定或 Ward 吞并 needle 小簇。
- anchor 是否应由 entry span/phase coherence 自适应选择。
- compressed slots 是否应采用频率裁剪或频率自适应 RoPE。
- learned bias / learned anchor 是否能在少量 CPT 下显著优于固定锚点。
- 语义 cluster 的 value 均值误差是否会盖过 position 改进。

## P13. 推荐实验顺序

1. 在现有 dump 上跑 S0.3/S0.6 的 `K_max` + Ward clipped gate，先定 `lambda_rel`。
2. 补 S0.2 口径②和 S0.7，确认 `K_eff`/supersession 这两条风险没有反转结论。
3. 用真实 anchor-score 消融比较 `p_mid only`、`p_lo+p_hi`、三锚点和必要的自适应 anchor。
4. clipped gate 与 anchor-score 都过线后，再进入生产 cache 实现。
5. 如果三锚点不足，先试 learned bias over fixed anchors；不要直接跳到 learned compactor。

## P14. 术语速查

| 术语 | 含义 |
|---|---|
| entry | 存储单位，包含内容均值、value 均值、质量、锚点统计 |
| virtual slot | attention 单位，由 entry 按锚点展开得到 |
| anchor | 一个确定位置坐标，如 `p_lo/p_mid/p_hi` |
| `M` | 一个 entry 去重后的 anchor 数，1 到 3 |
| phase cancellation | 多个 post-RoPE 相位平均时互相抵消 |
| pre-RoPE key | qk-norm 后、apply_rope 前的 key |
| segment | 同 cluster 内的时序连续段，只保护低层合并 |
| `ℓ_block` | segment 边界保护到第几层 |
| learned anchor | 由模型/小模块预测的位置坐标或 anchor 权重 |
