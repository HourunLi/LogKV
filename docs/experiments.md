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
| `q_roped` | `(nh, T, hs)` | fp16 | 机制 B（分块用完即弃，不必整块落盘）| §13.2 的注意力质量-距离曲线 |
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
| S0.8 | 批量化路由（§5.4）与严格串行版的分歧率 | 那个近似能不能用 |

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
- **S0.8**：分歧率应 < 5%。若更高，flush 粒度要调小，或 `γ` 要更保守。

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
| **query 位置** | prompt 尾部 / 中部 / 前置 | **区分本方案与 eviction 类方法的关键设定**（§9-A）——尾部 query 是 retrieval-head 类方法的最佳工况 |
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

