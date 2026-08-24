# SemanticLogKV 术语表

> 本文件是 [`../CLAUDE.md`](../CLAUDE.md) 的配套速查表，不承载设计决策——**任何结论
> 和动机都在正文里**（设计看 CLAUDE.md §2，算法看 [`algorithm-spec.md`](algorithm-spec.md)，
> 风险看 [`risks-and-open-questions.md`](risks-and-open-questions.md)）。这里只回答
> "这个词/符号是什么意思"。
>
> **最容易踩的坑在 §T2 第一行：`B` 在代码库里有两个完全不同的含义。**

具体数值以 Qwen3-1.7B 为例（`litgpt/config.py:2740`）：`n_layer=28`、`n_head=16`、
`n_query_groups=8`、`n_embd=2048` ⇒ `head_dim=128`、`rotary_percentage=1.0`（full rotary）。

## T1. 结构层次（最容易混淆的一组）

从大到小的嵌套关系：

```
cache（每 layer 一个）
 └─ (batch 元素, KV 头) —— 每个组合一套独立的簇集合，共 B×G 套
     ├─ recent window：recent_size 个精确 token，不压缩
     └─ cluster × K          ← 语义身份，一个 centroid，只用于路由
         ├─ segment × 若干    ← 时序段，不各自分配存储，只约束合并
         └─ ladder（每 cluster 一条）
             └─ level 0..L_alloc-1
                 └─ entry × B′ per level   ← 真正存 k̄/v̄/w/锚点/Σ/Γ 的东西
                     └─ 读出时展开成 1~3 个 slot（虚拟槽）
```

| 术语 | 含义 |
|---|---|
| **cache** | `LogStructuredKVCache` 实例。**每个 transformer block 一个**，所以有 `n_layer` 个 |
| **recent window** | 滑动窗口里的精确 token，不做任何压缩。参数 `recent_size`，实验里恒为 **1024** |
| **flush** | token 滑出 recent window、进入压缩结构的动作。**路由就发生在这一刻** |
| **cluster（簇）** | 一个 centroid 代表的语义身份，**只用于路由**，本身不直接参与 attention 读出 |
| **segment（段）** | 一个 cluster 内一段不被时序打断的连续访问。**不各自分配 ladder**，只通过对齐填充约束合并（§5.11）|
| **ladder** | 本项目自造的词，代码里没有。指一个 cluster 内部那**一摞 level**，越往上每个 entry 覆盖的 token 越多，像梯子 |
| **level（层）** | ladder 的一级。level 0 是写入层（部分可填），level ≥1 是进位层 |
| **entry** | 存储的最小单位：`k̄_raw`、`v̄`、`w`、`p_lo`/`p_hi`/`sum_wp`、Σ/Γ。**这是算内存时该数的东西** |
| **slot（槽 / 虚拟槽）** | 读出时 entry 按锚点展开后喂给 attention 的单位。一个 entry → 1~3 个 slot。**只影响瞬时计算量，不影响持久内存**（§4）|
| **block** | `_binary_carry` 里一次进位搬运的整块 `B′` 个 entry。二进制计数器的"一位" |

## T2. 维度与规模记号

| 符号 | 含义 | Qwen3-1.7B 取值 |
|---|---|---|
| **`B`** | ⚠️ **重载**：张量 shape 注释 `(B,G,S,d)` 里是 **batch size**；构造函数参数 `B`、`self.B`、CLI `--log_kv_B`、`compact` docstring 里的 "B-slot block" 是**每层 entry 数** | batch 通常 1~8；entry 数默认 **512**（现有方案）|
| **`B′`** | 设计文档专用，就是上面那个"每层 entry 数"，**改名只为避免和 batch 撞车** | 当前 Stage-0 默认 **128**；旧 **8** 只作为 smoke-test 紧预算或历史反例 |
| **`G`** | KV 头数 = `n_query_groups`。**不是 query 头数** | **8** |
| `nh` | query 头数 = `n_head` | 16 |
| `rf` | `nh / G`，几个 query 头共享一个 KV 头。共享 KV 头的 query 头**必然共享同一套簇** | 2 |
| `d` / `k_dim` | head_dim | 128 |
| `rope_n_elem` | 参与旋转的维数。full rotary 时 = `head_dim` | 128 |
| `F` | 不同 RoPE 频率数 = `rope_n_elem / 2` | 64 |
| `n` / `T` | 当前序列长度 / token 数 | — |
| `N` | `max_seq_length`，预分配依据 | 32768 |
| `S` | 读出时扁平槽池的大小 | — |
| `w` | 一个 entry 覆盖多少个原始 token | — |
| `M` / `M_s` | 一个 entry 去重后的锚点数（1~3）。**mass bias 必须除以它**（§2.3）| — |

## T3. 簇与预算记号

| 符号 | 含义 |
|---|---|
| `K` | 当前实际簇数，内容驱动，每个 (batch, KV头) 各不相同 |
| `K_max` | 簇数硬上界。**默认** `max(4, ⌈log₂N⌉)` 由 `max_seq_length` 推导，**但可用 `--log_kv_cluster_k_max` 覆盖**（消融正是在扫它，§5.6）|
| `K_eff` | 实测的"内容真正需要多少簇"，**unclipped DP-means，不设 `K_max` 上限**——是离线 CPU 分析口径，不是生产路径里被 `K_max` 截断后的实际簇数（两者不是同一个量，见 `experiments.md` S0.2）。**S0.2 要测它随 n 的整条曲线** |
| `L_alloc` | 每簇 ladder 的层数，按均衡界推导（§5.12）。**纯推导量，无 CLI 开关**；覆盖 `K_max` 时必须连带重算，否则预算算术失效 |
| `L_max` | "单簇独吞整条序列"所需层数。只用于说明超配，不用于定尺 |
| `ℓ` | 层索引 |
| `ℓ_block` | 段对齐保护到第几层。`0` 是合法的"关闭段边界保护"消融档（零代价，`2^0-1=0`）；需要非退化保护时代价是 `2^ℓ_block − 1` 槽/边界，**指数增长，只能取 1~2**；生产路径构造时硬校验 `ℓ_block ∈ {0,1,2}`（`0` 在内），Stage 0 的离线扫描（不经过 `op_log`/真实 cache）不受此约束（§5.21-2，§5.11 更正框）|
| `n_eff` | 簇 `c` 的 centroid 混合权重，**`γ` 衰减，浮点**。只喂 §5.5 的在线均值更新，不进 Ward 代价 |
| `n_total` | 簇 `c` 的真实物理规模（token/entry 数，**不含 pad**），**单调不减，从不衰减，整数**。Ward 合并代价（§5.6）和 §5.8 的 O(log n) 空间界都用这个，不能用 `n_eff`——早期版本只有一个 `n_c` 两处混用，会让 Ward 把"历史长但被衰减过"的簇误判成小簇（§5.5/§5.6 更正框）。**`§5.11` 的 `PAD_INSERT` 对齐也不能用它**（会在第一次填充后算错），改用独立的持久相位计数器 `level0_phase`——**不是**复用 `level_count[cluster,0]`：两者一度被认为等价，但 `carry_into_level`（§5.12）精确定义后这个等价性不再成立，见 §5.11 更正框 |

## T4. 算法参数（当前有效）

| 符号 | 参数名 | 含义 |
|---|---|---|
| `λ` | `log_kv_lambda` | **mass bias 系数**，只控制 `+λ·log(w)` 这一半；`−log(M)` 不受 `λ` 门控，无条件生效（§2.3）。沿用现有语义 |
| `λ_new` | — | 开新簇的距离阈值 = `λ_rel · s_h` |
| `λ_rel` | `log_kv_cluster_lambda_rel` | 上面那个的相对系数。**全方案最敏感的超参** |
| `s_h` | — | 每 (layer, **KV group**，不是 query head——聚类只在 k 空间做，一个 KV group 只有一份 k) 的 key 尺度估计 `E‖k−k̄‖²`，`k̄` 是整个标定集上的全局均值。**v1 用离线标定**（§5.21-4），标定值须写进 eval metadata；在线估计降级为消融 |
| `η` | `log_kv_seg_eta` | join cost 的时序权重。**只影响候选排序，不做决策**（§5.3）|
| `g0` | `log_kv_seg_g0` | 时序项 `φ(g)=g/(g+g0)` 的饱和尺度 |
| `g_max` | `log_kv_seg_gap_max` | 开新 segment 的间隔阈值 |
| `γ` | `log_kv_seg_forget` | 跨段时 centroid 的计数遗忘因子。`1`=纯均值，`0`=直接替换 |

> **记号冲突史**：`λ` 曾同时被用作 mass bias 系数和 segment 阈值，`τ` 曾表示 cosine
> 阈值。已统一为上表。读旧的变更记录时注意这一点。

## T5. 已废弃的记号（只出现在变更记录里）

| 符号 | 属于 | 为什么废弃 |
|---|---|---|
| `z` / `z_f` | v1、v2 | 位置的特征函数统计量。**本身是加权和，相位相消无法避免**，被 v3 的真实锚点取代（保留为消融对照）|
| `ρ` / `ρ_f` | v1、v2 | `\|z_f\|`，位置集中度。v3 里每个虚拟槽 `ρ≡1`，这个量不再存在 |
| `β` | v1 | 重物化增益 `ρ^(β-1)`。把"往哪转"和"多可信"耦合进一个标量，方向性错误 |
| `κ` | v2 | `log ρ` 置信项的系数。v3 里没有槽在"撒谎"，整条被消解 |
| `τ` | v3 之前 | cosine 阈值。度量改为平方欧氏后由 `λ_new` 取代 |
| `SEG_max` | v3.1 早期 | 每簇 segment 数上限。曾导致预分配乘爆到 4 万槽（§11-C）|

## T6. 数学量

| 符号 | 含义 |
|---|---|
| `k̄_raw` | entry 的 **pre-RoPE** 内容均值。**改存 pre-RoPE 是 v3 的核心改动** |
| `v̄` | entry 的 value 均值 |
| `p_lo` / `p_hi` | 锚点：该 entry 内最早 / 最晚成员的**真实位置**。min/max 合并，幂等 |
| `sum_wp` | `Σ_j w_j·p_j` 的 **int64** 累加器。合并就是加法，精确可结合。**必须 int64**（量程 `n²`）|
| `p_mid` | `clamp((2·sum_wp + w) // (2·w), p_lo, p_hi)`，位置的加权均值，**round-half-up 的整数实现**（不是 floor，也不是浮点 round，理由见 §5.14）。**是质心，不是真实成员位置**——早期版本用"继承权重更大一侧"的规则，不满足结合律且在平衡 Fenwick 路径下恒等于 `p_lo`（CLAUDE.md §2.2 更正框）|
| `μ_c` | 簇 `c` 的 centroid |
| `Σ`（`sigma_u`,`sigma2`）| 槽内 key 协方差的 rank-1 近似，供**分数侧**二阶修正。存储在 pre-RoPE 空间，读出时 `sigma_u` 必须和 `k_raw` 一样按锚点物化成 post-RoPE 才能和 `q` 点乘，否则坐标系不匹配（§5.14）|
| `Γ`（`gamma_a`,`gamma_b`,`gamma`）| **读出侧**的 rank-1 修正。结构上就是 `qᵀ(γa γbᵀ)`，一个秩 1 线性 state（§2.4）。`gamma_a` 和 `sigma_u` 一样需要按锚点 RoPE 物化；`gamma_b`（value 方向）和 `gamma`（标量特征值）不需要——value 从不被 RoPE |
| Ward 代价 | `(n_a n_b)/(n_a+n_b)·‖μ_a−μ_b‖²`，簇内平方和的增量。**与聚类距离、rank-1 残差是同一个量**（§5.2）|
| Chan merge | 并行计算两组数据合并后二阶矩的标准公式，`compact` 里用它合并 Σ/Γ |

## T7. 代码符号（现有实现）

| 名字 | 是什么 |
|---|---|
| `compact()` | 把两个 `B′` 块合并成一个。**按时间序拼接后配对相邻槽 `(2i,2i+1)`**，所以合并出的槽覆盖连续区间 |
| `_binary_carry()` | 二进制进位：level 空就放下、非空就 `compact` 后带着翻倍宽度继续往上 |
| `_append_level0()` | 往 level 0 追加 entry，满 `B′` 个就触发进位 |
| `_flush_pairs()` | 批量版的窗口 flush。**它存在的唯一理由就是消除逐对串行**——语义路由会把这个串行请回来（§11-B）|
| `_pair_rank1_stats()` | 算两个槽合并时新增的协方差，rank-1 化 |
| `log_kv_slot_attention()` | 槽级 attention。**这一行描述的是现有代码**：`score = scale·(q·k) + ½scale²σ²(q·σu)² + λ·log w`，`read = v̄ + scale·γ(q·γa)·γb`。语义簇版本把 mass bias 换成 **`λ·log(w) − log(M)`**（`−log(M)` 不受 `λ` 门控），见 §2.3/§5.15，**写单测时不要抄这一行的公式** |
| `LogKVStreamTrainingAttention` | 训练用的自定义 autograd。**forward 不建图，backward 重置 cache 并重放整条流**——语义路由打破了它的确定性前提（§11-A）|
| `second_order` / `second_order_scale` | 是否构建 Σ/Γ / 它们的运行时缩放（CPT 期间 warmup 爬坡）|
| `causal_tail` | 在途 chunk 的因果掩码，省掉一个全尺寸 mask |
| `importance_pooling` | 已验证为负结果的邻近方案（niah 0.0827→0.0787）。**建议保持关闭**；语义簇开关打开时与 `pin` 一起被构造时硬性禁止组合（§5.1），共存数学尚未推导 |

**新增的 buffer**（§5.13）：`centroid`、`n_eff`、`n_total`、`p_hi_c`、
`current_segment`、`level0_phase`、`alive`、`level_count`、`pad_mask`、
`op_log`、`op_log_len`、`s_h`。`level0_phase` 是 `PAD_INSERT` 对齐唯一依据
的独立相位计数器，不能用 `level_count[cluster,0] mod 2^ℓ_block` 代替
（§5.11 的更正框：`carry_into_level` 精确定义后两者不再等价）。

## T8. 外部概念

| 术语 | 一句话 |
|---|---|
| **DP-means** | "够近就归入最近簇，否则开新簇"的确定性聚类，是 CRP 混合模型的小方差极限 |
| **CRP**（中餐馆过程）| 非参贝叶斯先验，`E[K_n] ≈ α·ln n`。因为是独立 Bernoulli 之和，Chernoff 给出**高概率界**而非仅期望 |
| **ddCRP** | 距离依赖的 CRP，落座概率含距离函数。`d` 取时间距离即得本方案的 join cost |
| **Pitman-Yor** | CRP 的推广，`E[K_n] ~ n^d`（幂律） |
| **Heaps 定律** | 自然语言的类型数按 `n^β`（β≈0.4~0.6）增长，**不是 log n**。这是 `K_eff` 可能是幂律的经验依据 |
| **填充数**（packing number）| 有界空间在给定半径下最多能放几个不相交球。DP-means 的簇数上界，**与 n 无关** ——若成立则 `K_eff` 会饱和 |
| **Fenwick tree** | 树状数组。本项目用的是它的"二进制计数器 + 进位"结构 |
| **RoPE / apply_rope** | 旋转位置编码。litgpt 用**前后半重复**布局，`cos[f]==cos[f+d/2]` |
| **pre-RoPE / post-RoPE** | 施加旋转之前 / 之后的 key。v3 改存 pre-RoPE |
| **GQA** | 分组查询注意力，多个 query 头共享一个 KV 头 |
| **needle / haystack / NIAH** | 长文本里插入的关键事实 / 无关背景 / 该任务。**multi-needle** 是多针版本，区分度更高 |
| **supersession** | 后面的信息作废前面的（"考了100分→更正为90分"）。均值池化表达不了（§2.4）|
| **delta rule / Gated DeltaNet** | `S_t = α_t(I−β_t k_tk_tᵀ)S_{t−1} + β_t k_tv_tᵀ`，先擦除当前 key 方向的旧关联再写入 |
| **attention sink** | 巨范数、吸引大量注意力的 token（通常在序列开头）。本方案里会因距离远而自成一簇 |
| **CPT** | 继续预训练（continued pre-training），让模型适应压缩表示 |

## T9. 实验记号

| 记号 | 含义 |
|---|---|
| **Stage 0** | 离线证伪：一次 GPU dump + 全部 CPU 分析。**在写任何生产代码之前**——但 S0.8 全部子项（不只 3b）都还需要 `cache_serial`/`cache_batch` 这两份朴素 CPU 参考实现（§5.3 严格串行、§5.4 批量近似算法各自的非向量化实现，**不是** §5.18 第 4 步的生产向量化实现，两者是不同的制品）先落地；3b 在此之上额外还需要 `log_kv_slot_attention`/`get_attention_state()` 的 `CacheAttentionState` 接口扩展，见 `algorithm-spec.md` §5.18 第 2 步与 §5.21-5 |
| **Stage 1 / 2 / 3** | 实现 / eval-time 探测 / CPT |
| **S0.0** | 扫 `(g_max, ℓ_block)`——**最高优先级**，回答"收益来自语义分组还是分段边界" |
| S0.1–S0.8 | 单测 / `K_eff` 曲线 / needle 隔离率 / 簇内 key+value 方差 / 跨度分布 / 锚点膨胀 `E[M]` / supersession 比例 / 批量化近似的分歧率 |
| **S0.1**（精确定义） | 只指 `log_kv_position.py`/§5.14 的纯 CPU 单测（mass bias 计数守恒、`w=0` 恒等元等）。**不包括**下面的"实现单测"，即使两者都不需要 dump |
| **实现单测** | `algorithm-spec.md` 里散落的一类正确性单测（重放正确性、Phase 3a/3b 元数据更新、Ward 合并 `level_count` 不变量、padding 槽字段对拍等），验证 §5.3/§5.4/§5.6/§5.11/§5.21-2 这些**实现本身**对不对。之前借用过 S0.1 标签（因为同样纯 CPU、不需要 dump），已统一改名——它们依赖各自测的那块路由/重放机制先写出来，不依赖 `log_kv_position.py`，前置条件和真正的 S0.1 不同，不能混用 |
| **Config A/B/C** | Stage 2 的三档，**全部是 eval-time 消融参考点，不设通过/失败容差**（正确性检验在 Stage 0/1 的纯 CPU 单测，不在这里）：`K_max=1` 参考基线（不预期复现旧数字）/ 纯语义（`η=0, g_max=∞`）/ 主实验 |
| **D1 自检** | 另一分支的诊断实验，测出 width≥8 时 rank-1 失真明显 |
| **memory-matched** | 把 vanilla 的 `B` 调大到同 entry 数再比，**排除"只是多用了内存"** |
