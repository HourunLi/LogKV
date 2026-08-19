# SemanticLogKV 实验协议与消融表

> 本文件是 [`../CLAUDE.md`](../CLAUDE.md) 的第 §6–§7 章，独立成文只为便于阅读和维护。
> **章节编号沿用全局编号**，交叉引用全部有效。算法规格看
> [`algorithm-spec.md`](algorithm-spec.md)，风险与未决问题看
> [`risks-and-open-questions.md`](risks-and-open-questions.md)。
>
> **Stage 0 不碰训练、不落地 full attention，却能在写任何生产代码之前否掉整个
> 方案。不要跳过。**

## 6. 实验协议

四个阶段，每阶段带可证伪的决策门。Stage 0 不需要训练、机制 A（k/v dump）的开销
可以忽略不计，但**机制 B 不是零成本**——它逐 query block 手算注意力分数并按距离
分桶（§"两套机制"一节），block size 和总运行时需要在实际跑之前估一下（一条
32k prompt、`block_size` 取几百到一千量级时，总浮点运算量和一次真实的
`(nh, T, T)` 全量 attention 是同一个数量级，只是不落盘、内存峰值低得多）；真正
省下来的是**训练**（不需要梯度、不需要反复迭代）和**落盘量**（分桶直方图而不是
完整 attention 矩阵），不是"几乎不花 GPU"这种程度的省。

> **贯穿性硬要求（§2.5）**：Stage 0 的所有统计**必须逐层 × 逐头分开报告**。

### Stage 0 — 离线可证伪（1 次 GPU dump + 全部 CPU 分析）

只需要一次前向：对几条 32k 的 NIAH prompt dump 每层每头的 **pre-RoPE k / v** 与
needle 的 token span（复用另一分支已有的 `log_kv_pin_diag.py` 定位逻辑）。

> **S0.0 是离线模拟，可以探索生产实现负担不起的配置**（例如 `ℓ_block = L_alloc`
> 的"完全连续分段"）。它的职责是回答科学问题：**可负担的区间里是否包含了大部分
> 价值**。

#### Stage 0 dump 规格（这个脚本是本项目该写的第一段代码）

Stage 0 的全部结论都建立在这份 dump 上，所以它排在**任何生产代码之前**（§5.21-5）。

**两套机制，不是一个 hook——`attn_mass_by_dist` 在原来那个 hook 点算不出来。**
原设计把它写成"在 hook 里就地累加"，但真实注意力质量需要**已经做完 RoPE 的 q、k
和真实的 attention score**，而 `norm_q`/`norm_k` 之后、`apply_rope` 之前这个点还
没旋转，且默认无 softcapping 路径直接调用 `F.scaled_dot_product_attention`
（`model.py:1464`，融合 kernel）——**中间的 score 张量根本不会被实例化，没有地方
可挂 hook 去拿它**。核对过 `model.py:1447-1467` 后确认，手动展开 score 的分支只在
`attention_logit_softcapping is not None` 时才走，默认路径拿不到。

**机制 A（cheap，现成的 hook 点）**：`CausalSelfAttention.forward` 里
**`norm_q`/`norm_k` 之后、`apply_rope` 之前**（`model.py:~813`），取
§5.21-1 定义的那个 `k_raw`。取错位置（比如取 `qkv.split` 的原始输出）会让度量落在
未归一化空间，`s_h` 标定和全部方差统计作废。

**机制 B（`attn_mass_by_dist` 专用，不是简单 hook，dump 脚本要显式接管这一步）**：
在 `apply_rope` **之后**捕获，脱离模型的融合 SDPA 调用，在 dump 脚本里自己按 query
分块手算——但必须**逐位复现** `model.py:816-817`/`857-860`/`1450-1466` 那三步，
否则算出来的 `attn_mass_by_dist` 和模型真实算的注意力不是同一个量，S0.0/§13.2 的
结论建立在错误的基准上：

1. **完整 post-RoPE q/k，不是旋转切片**。`model.py:816-817` 拼的是
   `q = cat(q_roped, q[..., rope_n_elem:])`——`q_roped`/`k_roped` 只是前
   `rope_n_elem` 维（位置通道），真实 attention 用的是拼上未旋转内容通道尾部之后
   的**完整向量**。partial rotary 模型下漏掉尾部会让分数系统性偏小。**Qwen3-1.7B
   的 `rotary_percentage=1.0`（`rope_n_elem == head_size`），尾部为空，
   `q_roped` 恰好等于完整 `q`**——这是这个模型上机制 B 能简化成只 dump
   `q_roped`/`k_roped` 的唯一原因，不是通用结论；换一个 partial-rotary 模型，
   dump 脚本必须同时落盘尾部通道并在这一步拼接。
2. **GQA 展开**：`model.py:857-860` 在算分数前把 `k`（`(B, n_query_groups, T, hs)`，
   这里 `n_query_groups=G=8`）按 `q_per_kv = n_head/n_query_groups = 2` 
   `repeat_interleave` 到 `(B, nh_q, T, hs)`（`nh_q=16`），即每个 KV group 被
   **2 个相邻 query head 共享**。dump 出来的 `k_roped` 是 `(G, T, hs)`，必须先做
   同样的 `repeat_interleave(q_per_kv, dim=0)` 展开到 `(nh, T, hs)` 再和
   `q_roped` 相乘——否则退化成 `nh_q` 和 `G` 直接错位相乘（`16` 头对 `8` 组，
   形状都对不上，或者对上了也是语义错的配对）。
3. **可选 softcapping**：`model.py:1454-1456` 若 `config.attention_logit_softcapping
   is not None`，在 softmax 前对 `scores` 做 `do_softcapping`（`tanh` 压缩）。
   **Qwen3-1.7B 不启用 softcapping**（那是 Gemma2 系列的机制），所以这一步在当前
   实验里是 no-op，但脚本要把这一步作为条件分支写出来，换模型配置时才不会静默漏掉。

```
q_per_kv = nh // G                                     # GQA 展开倍数，Qwen3-1.7B 是 2
k_expanded = k_roped.repeat_interleave(q_per_kv, dim=0)  # (G,T,hs) -> (nh,T,hs)
for query_block in chunks(q_roped, block_size):          # q_roped 在本模型上即完整 post-RoPE q
    scores = query_block @ k_expanded.mT * scale          # 只对这一块，不是整个 (T,T)
    if attention_logit_softcapping is not None:            # 当前模型是 no-op，保留分支
        scores = do_softcapping(scores, attention_logit_softcapping)
    scores = causal_mask(scores)                           # 该块相对完整 k 序列的因果掩码
    probs  = softmax(scores, dim=-1)
    attn_mass_by_dist += bincount_by_|i-j|(probs)           # 累加进分桶直方图，立刻丢掉 probs
```

这是 Stage 0 **专用**的离线计算，刻意绕开生产路径的融合 kernel——可以接受，因为它
不是训练/推理路径的一部分，只在这一次 dump 里跑。

**dump 什么**：

| 张量 | shape | dtype | 来源 | 用途 |
|---|---|---|---|---|
| `k_raw` | `(G, T, hs)` | fp16 | 机制 A | 全部聚类统计的输入 |
| `v` | `(G, T, hs)` | fp16 | 机制 A | **S0.4 的 value 方差、S0.7 的 supersession** |
| `q_roped` | `(nh, T, hs)` | fp16 | 机制 B（分块用完即弃，不必整块落盘）| §13.2 的注意力质量-距离曲线；S0.8 3b 的 attention 读出比较也复用这个循环、同样不落盘，见下方 S0.8 3b 的说明 |
| `attn_mass_by_dist` | `(nh, n_bins)` | fp32 | 机制 B | 落盘的是这个分桶直方图，不是 `q_roped` 本身 |

> **不要 dump 完整 attention，也不需要 dump `q_raw`。** `(nh, T, T)` 在 32k 下是
> 16×32768² ≈ 172 亿个元素，落盘不可行，机制 B 的分块计算天生避开了它。`q_raw`
> （旋转前的 query）在这份规格里没有任何用途——**聚类只在 k 空间做**，从不碰
> `q`，所以只需要 `q_roped`，不必再引入一个 `q_raw` 制造歧义。

**维度约定**：
- **`B` 固定为 1**。batch 维保留会引入 padding 对齐问题，而 Stage 0 不需要吞吐。
- `G` 维**必须保留且分开统计**——§2.5 的逐头要求就落在这里，聚合会掩盖"少数头有效"。
- 层维同理（§11-I）。

**层子集**：全 28 层 × k+v 的体量是 `28×8×32768×128×2B×2 ≈ 3.8 GB/条 prompt`，4 条
就 15 GB。**第一轮先 dump 层子集**（建议 `{0, 7, 14, 21, 27}`，覆盖浅/中/深），约
700 MB；只有当子集显示层间差异确实影响结论时，才做全量。

**needle span 对齐**（最容易出错的一处）：
- span 必须是**分词后的绝对 token 下标**，与 `k_raw` 的 `T` 轴同一坐标系；
- NIAH 构造器给的是字符区间，需要经 tokenizer 的 offset mapping 转换，**转换代码要和
  eval 用的是同一份**，否则 S0.3 的隔离率统计全是噪声；
- 复用另一分支 `log_kv_pin_diag.py` 已有的定位逻辑，不要重写。

**落盘格式**：每层一个 `.npy`（或单个 `.npz`），外加一份 JSON manifest 记录：
prompt id、tokenizer、序列长度、needle 的 token span、层/头索引、模型 config hash、
**以及 §5.21-4 的 `s_h` 标定值**。manifest 是后续所有分析的唯一真相来源。

| 编号 | 测什么 | 为什么 |
|---|---|---|
| **S0.0** | **扫 `(g_max, ℓ_block)`（固定 `η=0`，理由见 `algorithm-spec.md` §5.3 的更正框——`η>0` 会让"纯语义聚类"端点混入未受控的时序 tie-break）：一端纯语义聚类，一端完全连续分段。看槽内内容方差与锚点跨度的联合曲线** | **全课题最根本的问题：收益来自语义分组本身，还是仅仅来自更好的分段边界？** |
| S0.1 | §5.14 单测（含 mass bias **计数**守恒、`w=0` 恒等元）| 数学正确性，纯 CPU，不需要 dump |
| S0.2 | **`K_eff(n)` 整条曲线**（n 从 1k 到 32k）+ 每簇 segment 数，拟合饱和/log/幂律。**测两个口径,不是一个**：①纯 DP-means（不含 `η`/`g_max`/`γ`，不设 `K_max` 上限，离线 CPU 分析，衡量"内容本身有多少语义多样性"，喂曲线形状判定）；②生产三路路由（`η`/`g_max`/`γ` 按生产默认值打开，只是不设 `K_max` 上限，衡量"这套算法实际会尝试开多少簇"，喂 `algorithm-spec.md` §5.6 的 `c` 决策规则）。两者都与生产路径里被 `K_max` 截断后的实际簇数（§5.6）是不同的量，后者永远 `≤ K_max` | 预算故事成不成立；外推到 1M；**并决定 §5.11 走 Fenwick 还是扁平贪心** |
| S0.3 | needle 隔离率：所在簇的成员数分布（关键是 `≤ B′` 的比例）| §3 + §5.9 的核心机制成不成立 |
| S0.4 | 簇内 **key 方差**与 **value 方差**（两个都要）/ 现有位置槽内方差 | key 方差管分数侧，**value 方差管读出侧**，只测前者会高估收益（§11-D）|
| S0.5 | entry 的 `(p_hi − p_lo)` 跨度分布，随 `(g_max, ℓ_block)` 变化 | 验证段机制确实压住了跨度 |
| S0.6 | 锚点去重后的平均倍数 `E[M]` 与分布 | 读出槽池会不会失控（§4）|
| S0.7 | 簇内 value 的并存 vs 作废比例，**分相邻/远距离统计** | §2.4 开放问题的判定实验 |
| S0.8 | 批量化路由（§5.4）与严格串行版的分歧率——**拆成三项**（cluster assignment/Ward 事件、segment+PAD 开销、最终 cache/readout 误差），定义见下方决策门 | 那个近似能不能用 |

**决策门**：
- **S0.0**：如果"完全连续分段"已经拿到大部分收益，语义聚类这条线的边际价值有限，
  应当直接转向更简单的"语义分段"方案（工程量小一个量级、不需要 CPT）。
  **这个门开在最前面，就是为了避免在错误的复杂度上投入。**
- **S0.2（三重）**：①32k 下**unclipped** `K_eff` 应在 10² 量级；②曲线形状若是幂律，
  论文定位从 `O(log n)` 改为 `O(n^d log n)` 并重新评估内存公平性；③每簇 segment 数
  `S ≪ B′` 则 §5.11 走对齐填充，`S ~ B′` 则改走扁平贪心。
  > **这三条测的都是纯 DP-means 口径的 unclipped `K_eff`，不是生产路径里被
  > `K_max` 截断后的簇数**——后者被 §5.6 的默认公式 `K_max = max(4, ⌈log₂N⌉)`
  > 卡死在 32k 下 15，必然远小于 10²。若 unclipped `K_eff` 真的测到 ~100，说明
  > **内容的语义多样性**远超当前 `K_max` 表，但这不直接推翻 §4 的 2344-entry
  > 内存故事——那笔账算的是 `K_max` 截断之后、Ward 合并生效之后的持久 cache
  > 占用，`K_max` 本来就是设计成"绑定后退化到少簇的位置序 ladder"（§4，**不等同
  > 于现有 LogKV**——level 0 单 token、位置表示都已经变了，见 §5.6 的
  > `K_max=1` 讨论）的安全阀。unclipped `K_eff` 大只说明安全阀会经常触发，不
  > 说明内存会超预算；它是"路由质量有多少损失"的信号，不是"内存会不会超"的
  > 信号，两者不要混着读。
  >
  > **`K_max_default` 里的常数 `c` 不从这条纯 DP-means 曲线算**——它要用第二个
  > 口径（生产三路路由、`η`/`g_max`/`γ` 全开但不设 `K_max` 上限）的 `K_eff`，
  > 具体决策规则见 `algorithm-spec.md` §5.6"从 `K_eff` 到 `c` 的决策规则"一节：
  > **取观测区间上界**（不是只看渐近拟合斜率——截距不可忽略时渐近斜率会系统性
  > 低估已验证区间内的需求），再用新增的"`K_max` 绑定率"这个 Stage 1/2 运行时
  > 指标校正；加大 `c` 也压不下绑定率时应该转向重新评估曲线形状，而不是继续
  > 调参。
- **S0.3**：needle 落在成员数 `≤ B′` 的簇里的比例应显著高于随机基线。若 needle 大多
  并入大簇，§3 的机制不成立，方案应就地停止。
- **S0.4**：key 方差应显著低于现有位置槽；**若 value 方差没有同步下降**，说明读出侧
  仍是 smear，收益要打对折，需要考虑按 `[k;v]` 联合聚类或簇内二次分裂。
- **S0.6（自 `p_mid` 修正后升格为真正的决策门）**：`E[M]` 应 < 1.6。修正前的
  `p_mid` 继承规则在平衡 Fenwick 路径下恒等于 `p_lo`（CLAUDE.md §2.2 更正框），
  副作用是把 `M` 压在 2；改成真质心后 `p_mid` 通常严格落在 `(p_lo, p_hi)` 内，
  `M` 趋近 3，**直接顶到 §4 那张表里"E[M]=3 ⇒ 4984 槽 ⇒ 超过 vanilla 3584"那一行**。
  若实测确实逼近 3，除 §4 的三层缓解外还有一条干净退路：**砍掉第三锚点只留
  `(p_lo, p_hi)`**——相对修正前的实现零损失（它本来就等于 `p_lo`），只是放弃了
  修正带来的那部分中心估计能力。
- **S0.7**：若簇内 value 系统性作废的比例 > 30%，把 Γ 的 delta-rule 广义化提到
  Stage 1 范围内；否则记录结论并搁置 §2.4。
- **S0.8**：分歧率不是一个单一标量，必须拆成三项分别报告，理由是它们诊断的是
  不同层级的问题、且互相之间不是线性关系（cluster 分歧可能被 segment/pad 开销
  放大或抵消，两者都不直接等于最终读出误差）：
  1. **cluster assignment divergence（含 Ward 事件）**：用
     `algorithm-spec.md` §5.4 已经给出的 `scan_op_log`/`resolve_final_slots`
     分别解析批量路径和严格串行参考路径的 `op_log`，得到每个 token 最终的
     **物理槽号**（`resolve_final_slots` 返回 `dict[int, int]`，只保留
     `find(v)` 的物理槽号分量，不带 `epoch`——`algorithm-spec.md` 该函数
     定义处已证明这一步不丢信息：解析时刻同一槽号至多有一个身份仍是并查集
     的根）。**不能直接比较裸槽号**——这里说的是跳过 `resolve_final_slots`、
     直接比较 `op_log` 里原始 `op.cluster` 的做法：两条路径的簇建立顺序不
     保证一致，槽号可能整体错位但语义上完全等价。改用**成对共簇一致率**
     （pairwise co-assignment agreement，类似 Rand index 的构造）：概念上是
     对所有 token 对 `(i,j)`（`i≠j`，`i<j` 各算一次，分母是
     `C(T,2)=T(T-1)/2`，不含自身配对），检查"i、j 在批量路径下是否同簇"与
     "i、j 在严格串行参考下是否同簇"这两个布尔值是否一致，一致的比例就是这
     一项的分数。**但字面按"所有 token 对"实现是 `O(T²)`，32k 单层单头就是
     `C(32768,2)≈5.4×10⁸` 对，乘上层数和 KV group 数会直接把 Stage 0 卡
     死——必须写成标准的 contingency-table 算法，不是真的去枚举所有
     pair**：把两条路径的最终身份分别映射成 `1..K_A`/`1..K_B` 的整数标签
     ——**必须是 `resolve_final_slots` 解析后的最终槽号，不能是
     `scan_op_log` 返回的原始 `token_identity`（记录时的 `(slot,epoch)`）**
     ——`token_identity` 只记录"这个 token 当初被写入哪个 `(slot,epoch)`"，
     Ward 合并发生在**之后**，同一个最终簇完全可能由多个不同的记录时
     `(slot,epoch)` 合并而来；直接拿未解析的 `(slot,epoch)` 当标签，会把
     语义上已经属于同一个最终簇的 token 错误地拆成多个标签，人为压低一致
     率，哪怕两条路径的最终聚类结构完全等价。标签本身之后怎么再编号（例如
     把 `resolve_final_slots` 给出的物理槽号再映射成 `1..K_A` 这样紧凑的
     整数区间）不影响结果，但**前提必须是先经过 `resolve_final_slots`
     解析到最终身份**，建一张 `K_A×K_B` 的列联表（contingency table）
     `n_{ab}` = 同时满足"批量路径标签为 `a`"和"严格串行标签为 `b`"的
     token 数（一次分组统计即可得到，`O(T)`，`K_A`/`K_B` 是两条路径各自
     的簇数，量级是 `K_max`，远小于 `T`）；再算行和 `a_i=Σ_b n_{ib}`、
     列和 `b_j=Σ_a n_{aj}`。用标准 Rand Index 恒等式：

     ```
     agree_pairs = C(T,2) − Σ_i C(a_i,2) − Σ_j C(b_j,2) + 2·Σ_{ab} C(n_{ab},2)
     score       = agree_pairs / C(T,2)                    # C(n,2) = n(n-1)/2
     ```

     推导：`Σ_{ab} C(n_{ab},2)` 是"两条路径都同簇"的 pair 数；
     `C(a_i,2)−Σ_b C(n_{ib},2)`（对 `i` 求和）是"批量路径同簇、严格串行
     不同簇"的 pair 数，`C(b_j,2)` 侧同理；`C(T,2)` 减去这两类"至少一条
     路径同簇"的 pair 数、再加回被减了两次的"两条路径都同簇"，剩下的正是
     "两条路径都判不同簇"的 pair 数，与"两条路径都判同簇"相加即为一致
     pair 总数，就是上面的 `agree_pairs`——这是 Rand Index 的标准写法
     （sklearn `rand_score` 用的就是这个恒等式，不是新推导），整个计算量
     是 `O(T + K_A·K_B)`，`T=32768` 时也是毫秒级。
     **同时报告 Ward 事件本身的分歧，但"逐一比较候选对"必须先定义怎么对齐，
     不能只说"逐一比较"**——两条路径触发 Ward 合并的**次数**、**触发时刻**，
     乃至候选对本身用的槽号/`epoch`，都可能因为批量近似让某些 token 走了
     不同的路由决策而不同（谁先把 `K_max` 填满这件事本身可能发生在不同的
     token 上），裸槽号更是天然不可比（和上面成对一致率同样的理由）。
     **具体做法**：用 `algorithm-spec.md` §5.4"S0.8 的 Ward 事件比较需要
     合并前的成员快照"一节新增的 `scan_op_log_for_ward_events` 分别扫两条
     路径各自的 `op_log`，得到两份 `WardEvent` 列表，字段已经就是这里需要
     的东西，不用重新发明数据结构：
     - `trigger_token_idx` 就是**主键**，两条路径按它对齐各自的事件列表
       ——它精确定义为 `tok0`（触发这次建簇的 orphan 组里到达顺序最早的
       那个 token 的 `token_idx`，等价于紧随该 `WARD_MERGE` 之后那条
       `NEW_CLUSTER` 携带的 `token_idx`）。**这是一个 path-local 的对齐键，
       不是一个保证跨路径相等的量**——它在单条路径内部良定义、不依赖任何
       实现细节，不需要另外定义"哪个位置算触发点"；但若两条路径因为批量
       近似（Phase 1 冻结的 centroid 快照 vs 严格串行逐 token 重算）对同一
       段输入分出了不同的 orphan 分组，两条路径各自的 `tok0` 可能不同，此时
       按这个键会**匹配不上**（处理方式见下方 `ward_event_inserted`/
       `deleted`），不能假设它必然跨路径相等；
     - 每个事件的"候选对"不用槽号表示，直接用 `WardEvent.keep_sketch_before`/
       `free_sketch_before`（合并前 `keep`/`free` 两个簇各自的 MinHash 草图，
       不是精确 token 集合——`algorithm-spec.md` §5.4 同一节已经论证过存
       精确 `frozenset` 有 O(T²) 的内存/时间风险，草图是替代它的有界表示，
       `scan_op_log_for_ward_events` 已经在扫描时快照好），不依赖槽号或
       `epoch` 怎么编号；
     - `trigger_token_idx`（即 `tok0`）在两条路径都出现的事件算作
       **匹配**，对匹配上的事件用 `algorithm-spec.md` 同一节定义的
       `estimate_jaccard(...)` 分别估计 `keep`/`free` 两侧的 Jaccard
       相似度（均值）——这是一个**估计值**，标准误差上界 `0.5/√k`（默认
       `k=128` 时 ≈4.4%），`k` 必须和这个数字一起写进 eval metadata；连带
       报告 `keep_size_before`/`free_size_before`（精确整数，不经估计），
       帮助判断一个偏低的 Jaccard 发生在大簇还是小簇上；
     - `trigger_token_idx` 只在其中一条路径出现的事件，**不要**强行配对
       或直接丢弃平均掉——分别计为 `ward_event_inserted`（只在批量路径）/
       `ward_event_deleted`（只在严格串行参考）单独报告。"事件根本没有
       对应物"和"配对上但决策不同"是两种不同的失效模式，混在一起平均会
       互相掩盖。

     Ward 决策错一次会级联影响后续所有共享该簇的 token，比单个 token 路由
     错位后果更重，值得单独一行，不要只并进上面的成对一致率里被平均掉。
  2. **segment/PAD overhead**：`segment_count_ratio = 批量路径总 segment 数 /
     严格串行总 segment 数`，`pad_entry_ratio` 同理（用 `PAD_INSERT` 的
     `count` 字段求和）。**预期方向是确定的、不是双向噪声**——
     `algorithm-spec.md` §5.4"批量路由留下的一个未解决风险"那节已经论证过，
     `p_hi_c` 批内冻结只会让批量路径误判"隔了很久"从而**多开** segment，
     不会反过来少开，所以这两个比值预期 `≥ 1`，报告的重点是**大多少**，
     而不是"有没有偏差"。
  3. **最终 cache/readout 差异，拆成两个子项，硬性决策门只挂在 3b**——第 1
     项的成对一致率只给出一个标量一致度，不给出两条路径之间簇的一一映射；
     聚类本身若发生真实分歧（不只是槽号错位），"这个槽对应那个槽"这件事
     可能根本没有良定义的答案（比如批量路径把两个簇合并了、严格串行参考
     没合并，没有哪个单一的严格串行簇能对应这一个批量簇），直接拿第 1 项
     的身份匹配去逐字段比较 ladder 会在这种情形下产出没有意义的数字：
     - **3a（诊断，不是决策依据）：ladder 字段相对误差，只在"两条路径的
       簇 token 集合完全相同"的子集上比较**。对每条路径分别用
       `scan_op_log`/`resolve_final_slots` 得到每个存活簇的原始 token
       绝对位置集合，只对**两边存在完全相同 token 集合**的簇配对（集合
       相等，不是"重叠"或"相似"），逐字段比较 `k̄/v̄/w/p_lo/p_hi/sum_wp`
       等的相对误差；**落不进这个精确匹配子集的 token 单独报告成
       `unmatched_token_ratio`**，不强行拿 Jaccard 最高的对手配对去比较
       字段值——需要更细粒度诊断时可以用 Jaccard 相似度做**软匹配**
       （或匈牙利算法求最大权二分匹配）看"大致对应哪个簇"，但那是可选
       的调试辅助，产出的字段误差不作为正式报告数字，因为不同簇之间
       字段值天然没有可比性。
     - **3b（决策依据，唯一的硬性决策门）：两条路径各自处理完整条序列、
       最终 cache 状态都已构造完毕之后，做一次真实 attention 读出，比较
       批量路径与严格串行参考给出的注意力输出的相对 L2 误差**——是针对
       整条序列处理完毕后那个最终状态的**单次**测量，不是逐个 flush 批
       重复测量再平均，也不是"每处理完一批就测一次"的意思（具体用哪些
       query、选多大范围，见下方 blockquote）。**这一项不需要任何簇
       对齐**——直接对同一批 query 分别跑一次 attention、比较输出张量，
       两条路径的簇怎么编号、有没有一一对应完全不影响这个比较是否良定义，
       因此是唯一在"聚类本身发生真实分歧"时依然能给出干净数字的一项，
       也是唯一直接回答"这个近似值不值得用"的一项——前两项（1、2）加上
       3a 都是诊断信号（帮助归因"差在哪"），3b 才是决策依据。

       > **3b 需要真实 query，但上面"Stage 0 dump 规格"一节明确
       > `q_roped` 不整块落盘（"分块用完即弃"）——这两处必须显式对齐，
       > 不能各写各的、留下一个事后才发现的矛盾。** S0.8（含 3b）本来
       > 就已经不是"对着 Stage 0 那份静态 dump 事后做纯 CPU 分析"——它
       > 需要先跑两条路由/cache 构造路径（批量近似 + 严格串行参考），
       > 这一步只吃 `k_raw`/`v`/`pos`（机制 A 已经落盘的东西），和 q 完全
       > 无关，可以离线用 `algorithm-spec.md` §5.4/§5.18 那套 CPU 参考
       > 实现做（S0.1 已经在用同一类参考实现，不需要额外 GPU）。**3b
       > 真正要用到 q 的地方只有"拿这两份已经构造好的 cache 状态各做
       > 一次 attention 读出"这一步，决定：这一步复用机制 B 已有的、
       > 按 query block 处理、用完即弃的循环，不新增任何持久化。**
       > 具体做法：cache 构造（不需要 q，见上）在同一次 dump 里先完成；
       > 机制 B 本来就要为 `attn_mass_by_dist` 逐块算一次
       > `q_roped @ k_expanded.mT`，3b 在**同一个循环、同一个
       > `query_block`** 上，额外用两份已经就绪的 cache 状态各算一次
       > 读出、累积相对 L2 误差需要的分子/分母，算完这一块就跟着
       > `q_roped` 一起丢弃——q 全程不落盘，只是在它本来就要被用一次算
       > `attn_mass_by_dist` 的那个循环里"顺路多用一次"。**为了保证
       > 因果性，3b 只用序列尾部的 query，且精确定义用多少个：一个
       > 独立的 `tail_query_count` 参数**（默认借用 `flush_granularity`
       > 的量级，但是一个独立字段，不随 `block_size` 变化），窗口定义
       > 为"绝对位置落在 `[T − tail_query_count, T)` 的全部 query"（此时
       > 两条路径的最终 cache 状态都已经构造完毕，序列尾部的 query 因果
       > 地有权看到整个 cache）。
       >
       > **更正（这一轮修的）：上一版直接复用 `chunks(q_roped, block_size)`
       > 产生的最后一个 `query_block` 划定这个窗口，理由是"不引入新
       > 参数"——这个理由没有权衡代价，必须收回。** `block_size` 是机制 B
       > 为控制内存/计算峰值设的分块粒度，和"3b 该用多少个尾部 query 来
       > 估计误差"是两个不相关的问题：调 `block_size`（比如为了让 dump
       > 脚本跑得更快、或适配不同显存）会顺带改变最后一块的 query 数量和
       > 绝对位置集合，3b 的 L2 误差分子/分母的样本量因此跟着变，< 5%
       > 这个判据可能因为一次纯粹的性能调参而改变结论——一个硬性决策门
       > 不应该对一个和它要回答的问题无关的旋钮敏感，这比"多一个参数"的
       > 代价更值得付。`tail_query_count` 和 `block_size` 各自独立、互不
       > 派生。
       >
       > 机制 B 原有的按 `block_size` 分块的循环结构不变（它服务的是
       > 内存/计算峰值控制，和 `attn_mass_by_dist` 的其它职责一样）；3b
       > 的累积逻辑在这个既有循环内部新增一个与块边界无关的掩码：对每个
       > `query_block`，取 `abs_idx = block_start + arange(len(query_block))`，
       > `tail_mask = abs_idx >= T - tail_query_count`，只用 `tail_mask`
       > 选中的子集累积 3b 的 L2 误差分子/分母——`tail_query_count` 大于
       > `block_size` 时这个窗口会跨越不止一块，小于 `block_size` 时只
       > 覆盖最后一块的一部分，两种情形这个掩码写法都正确处理，不需要
       > 分支特判。3b 依然是这一次性的窗口给出的单个相对 L2 误差数字，
       > 不按 block 取平均——不需要让 cache 构造和机制 B 的循环逐块交错
       > 对齐，两个阶段谁先谁后不重要，只要 cache 构造在机制 B 处理到
       > 这个尾部窗口涉及的最后一块之前完成。**`block_size` 和
       > `tail_query_count` 必须一起写进 eval metadata**（与 §5.21-4
       > `s_h` 标定值同一纪律）——引用这个 5% 数字时，两者缺一都不算
       > 完整复现实验设置。
       >
       > **不选"额外持久化一份 q_roped（或它的抽样子集）供事后用"这条
       > 路**：那样会让 S0.8 依赖的输入和"Stage 0 只落盘
       > k/v/`attn_mass_by_dist` 直方图"这条既定原则产生一个例外，且
       > 抽样出来的 query 子集能不能代表真实 readout 误差是一个新的、
       > 未经验证的假设；复用机制 B 现成的循环不引入这个假设，也不需要
       > 改动上面的 dump 表。

  **决策门只挂在 3b 上**：注意力读出的相对 L2 误差应 < 5%（沿用原来的
  数字，但现在明确它挂在哪一项，且不再依赖 3a 那套需要精确 token 集合
  匹配、覆盖率可能不到 100% 的 ladder 字段比较）。**这个 < 5% 的结论只
  覆盖"尾部 query"这一种设定**——3b 按定义只用序列尾部窗口
  （`tail_query_count`）内的 query，不覆盖 query 在中部/前置的情形；
  下面 §7 消融表"query 位置"那一行是一个独立的、eval-time（Stage 2，
  真实模型输出）测量，回答的是同一个问题在中部/前置 query 下什么样，
  **不是 3b 的延伸，也不共享它的 5% 阈值**——3b 只代表尾部 query，不要
  拿它的结论去承担中部/前置 query 的判断，两者的结论不要互相借用。第
  1、2、3a 项没有独立的通过/失败阈值，是诊断输出——**但必须报告**，因为若 3b 超标，第 1/2/3a
  项决定了修法：如果是 cluster assignment 分歧主导（尤其 Ward 事件分歧），
  要收紧 Phase 1 的批量
  近似（比如缩小 flush 粒度）；如果主要是 segment/pad overhead 主导且
  cluster assignment 本身一致率很高，按 `algorithm-spec.md` §5.4 那节已经
  给出的方向，缓解应该是给 Phase 1 内部的 `p_hi_c` 加一次按簇分组的前缀
  扫描，而不是缩小 flush 粒度（缩小粒度对 segment/pad 膨胀这个问题没有针对
  性，代价却是实打实的，§11-B 摊还论证依赖较大的 flush 粒度）。`γ` 更保守
  只在 cluster assignment 分歧确实由 centroid 冻结导致漂移过大时才是对症的
  修法，不是对所有分歧类型都有效的万能旋钮——原来"分歧率高就调小 flush
  粒度或调保守 `γ`"这条建议没有说清楚"这两个旋钮分别对应哪种分歧"，容易在
  不对症的维度上白费力气，这次一并说清楚。

### Stage 1 — 实现

按 §5 落地，默认关闭时逐字节复现现状。**动手前先读 §5.19 和 §11。**

### Stage 2 — eval-time 探测

诚实的预期：新表示对现有 CPT 权重是分布外的，绝对分数可能不好看。价值在于**形状而非
绝对值**。

> **正确性检验不在这里，在 Stage 0/1。** `algorithm-spec.md` §5.14/§5.18 的
> `anchor_mode=z`（honest-decay）CPU 单测才是"新代码复现旧数学"的正式闸门——它绕开了
> level 0 宽度的差异，纯代数、逐位精确、不需要 GPU。下面的 A/B/C 三档全部是**eval-time
> 消融参考点**，不设通过/失败的容差。

| 配置 | 参数 | 期望 |
|---|---|---|
| A（`K_max=1` 参考点）| 强制单簇 | **不预期复现 0.1716/0.0827**——level 0 从"2-token 合并"改成单 token 是设计里不可选的改动（§2.1），即使实现完全正确数字也应该不同（大概率更好，因为分辨率更细）。这一行给后面 B/C 一个"仅簇轴退化"的基线，不是通过/失败判据 |
| B 纯语义 | `K=16, g_max=∞, η=0` | 隔离段机制的贡献。**`η=0` 不能省**——`algorithm-spec.md` §5.3 已经把"纯语义"精确定义为 `η=0 且 (g_max=∞ 或 ℓ_block=0)`，只设 `g_max=∞` 不设 `η=0` 时统一代价的 tie-break 仍会让 `c*` 偏离纯语义最近簇，这一档就不是真正的纯语义端点 |
| C 主实验 | `K=16`，扫 `λ_rel` 和 `(g_max, ℓ_block)` | niah 应显著高于 A |

```bash
DIAG_ARGS="--log_kv_semantic_clusters true --log_kv_cluster_k_max 16 \
  --log_kv_cluster_lambda_rel 1.0 --log_kv_seg_gap_max 4096 \
  --log_kv_seg_block_level 1" \
    bash majob.sh exp/qwen1.7b-32k/arc_warmup.yaml
```

沿用另一分支 §8.5 的坑：主 benchmark 参数不要传 `none`（LongBench 来自主列表）；
多配置串行跑（输出目录不按超参分开，靠 JSON 内的 metadata 区分）；新参数要一并写进
eval 的 metadata 字段。

### Stage 3 — CPT

Stage 2 有信号后再投入。v3 没有需要 warmup 的新标量（v2 的 `κ` 已被消解），训练侧
风险低于 v2；若 S0.7 触发了 Γ 广义化，那部分沿用 `second_order_scale` 的爬坡机制
（目标值别设太激进、warmup 别只给 10 步——另一分支 §4 真实踩过的崩溃教训）。

## 7. 消融表

每一行回答一个独立问题。所有行都必须附带**两笔内存账**（§4）。

| 旋钮 | 取值 | 回答的问题 |
|---|---|---|
| **`(g_max, ℓ_block)`** | 纯语义 → 完全分段 | **语义分组 vs 分段边界，谁贡献大？（S0.0 的 eval 版）** |
| **query 位置** | prompt 尾部 / 中部 / 前置 | **区分本方案与 eviction 类方法的关键设定**（§9-A）——尾部 query 是 retrieval-head 类方法的最佳工况。**这一行是独立的 eval-time 测量，不是 S0.8 3b 的延伸**——3b 只用固定的尾部 `tail_query_count` 窗口，其 <5% 决策门不覆盖、也不能借用来回答中部/前置 query 的表现，两者结论不互相代入 |
| `K_max` | 1 / 4 / 16 / 64 | 语义分组本身值多少分？1 是现状锚点。**覆盖默认值时要连带重算 `L_alloc`**（§5.6）|
| `K:B′` 分配 | 32×4 / 16×8 / 8×16 | 语义分辨率 vs 时序分辨率，总预算固定 |
| `anchor_mode` | `lo_hi_mid` / `lo_hi` / `mid` / `z` | 锚点表示 vs v2 的 z 统计量 |
| `λ_rel` | 0.5 – 2.0 | needle 隔离与簇纯度的平衡点 |
| `λ`（mass bias）| 0 / 1 | 大簇的计数质量补偿是否仍然正确 |
| `γ`（遗忘因子）| 0 / 0.5 / 1 | centroid 门控更新值不值 |
| rank-1 Σ/Γ | 关 / 现有构造 / delta-rule 构造 | 第三档取决于 S0.7 |
| vanilla memory-matched | B 调大到同 entry 数 | **排除"只是多用了内存"** |

**multi-needle 行必须同时报告 `K_max` 的绑定频率**（§5.6）：needle 数逼近 `K_max` 时
Ward 会把 needle 漏斗进同一个簇，分数下降到底是聚类不行还是预算被撑爆，不报这个数
无法归因。

指标沿用现有四项（ACC/LongBench/LongBench_e/niah@32768），**另加 multi-needle**——
单 needle 一旦从 0.08 提上去就会迅速失去区分度。v3 的锚点表示对 multi-needle 应有
额外优势（多个 needle 各自成簇、各自保留精确位置）。

