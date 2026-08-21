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

> **贯穿性硬要求（§2.5）**：Stage 0 的所有统计**必须逐层分开报告，且按统计量本身
> 的粒度选对下一级维度，不是统一"逐头"**——`k`/`v`/聚类/路由相关的统计按
> **(layer, KV group)**（聚类按 `(B, G)` 独立进行，一个 KV group 只有一份 `k`）；
> 依赖 `q` 的注意力质量统计（`attn_mass_by_dist`、S0.8 3b 的 `error_3b`）按
> **(layer, query head)**（`q` 是 per-query-head 的，见下方 S0.8 3b 一节）。

### Stage 0 — 离线可证伪（1 次 GPU dump + 主体 CPU 分析，S0.8 3b 例外——须在 dump 进程内完成）

只需要一次前向：对几条 32k 的 NIAH prompt dump 每层每 KV group 的 **pre-RoPE k / v**
（`(B, G, T, hs)`，不是每 query head 一份——GQA 下同一 KV group 被 `q_per_kv` 个
query head 共享）与 needle 的 token span（复用另一分支已有的 `log_kv_pin_diag.py`
定位逻辑）。

> **S0.0 是离线模拟，可以探索生产实现负担不起的配置**（例如 `ℓ_block = L_alloc`
> 的"完全连续分段"）。它的职责是回答科学问题：**可负担的区间里是否包含了大部分
> 价值**。

> **范围边界**：这个"1 次 GPU dump + CPU 后处理"的划分覆盖 S0.0、S0.2–S0.7
> 和 S0.8 的 1/2/3a 项；S0.1 是纯 CPU 单测，不读 dump；S0.8 3b 需要未落盘的
> `q_tail`，必须在 GPU dump 进程内完成。S0.8 目前仍缺 `cache_serial`/
> `cache_batch` 两份朴素 CPU 参考 cache，尤其 `cache_batch` 指 §5.4 Phase 1/2/3
> 的慢速参考实现，不是 Stage 1 的向量化生产实现。

#### Stage 0 dump 规格（这个脚本是本项目该写的第一段代码）

Stage 0 的全部结论都建立在这份 dump 上，所以它排在**任何生产代码之前**（§5.21-5）。

> **这份规格的机制 A（k/v）部分已经实现**，核心文件清单见 `CLAUDE.md` §0。
> **机制 B（下方）以及本节"落盘格式"要求的 `tail_query_count`/MinHash 三元组等字段
> 仍未实现**——现有脚本的 manifest 只有 k/v、`s_h`/`vh`、基础样本信息，不满足下面
> 完整 schema，不要把它当成已经覆盖了整份 Stage 0 dump 规格。
>
> **"dump 提供的数据够不够"和"有没有分析代码去算某个 S0.x 指标"是两回事。**
> 目前已经配了分析代码、能直接跑的是 S0.0、S0.3、S0.4、S0.5、S0.6、S0.2 口径①：
> S0.0/S0.4/S0.5/S0.2 口径①由 `SweepAccumulator`/`route_dpmeans_segments`
> 覆盖；S0.3 是 `unused/semantic_s0_needle_isolation.py`；S0.6 是
> `unused/semantic_s0_anchor_dedup.py`。S0.3/S0.6 已完成首批真实 dump 实跑，
> 并已补上 `K_max` + Ward clipped 探针；下一步是实跑 clipped gate。
> S0.7 的 supersession 判定逻辑还没写；S0.8 仍缺上述两份 CPU 参考 cache。

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
| `k_raw` | `(G, T, hs)` | **fp32**（见下方"更正"框，`float16` 仅作显式选项）| 机制 A | 全部聚类统计的输入 |
| `v` | `(G, T, hs)` | **fp32**（同上）| 机制 A | **S0.4 的 value 方差、S0.7 的 supersession** |
| `q_roped` | `(nh, T, hs)` | fp16 | 机制 B（分块用完即弃，不必整块落盘）| §13.2 的注意力质量-距离曲线；S0.8 3b 的 attention 读出比较也复用这个循环、同样不落盘，见下方 S0.8 3b 的说明 |
| `attn_mass_by_dist` | `(nh, n_bins)` | fp32 | 机制 B | 落盘的是这个分桶直方图，不是 `q_roped` 本身 |

> **更正（这一轮修的）：`k_raw`/`v` 的落盘 dtype 从 fp16 改成 fp32，这不是可选的
> 存储优化，是避免重新引入刚修过的一个 bug。** `s_h`/`vh` 标定（`RunningKeyScale.
> update()`）永远吃落盘前的 fp32 张量，若 `k_raw`/`v` 落盘时被量化成 fp16，
> DP-means 路由读到的就是量化过的数组，`d2 = ‖centroid − x‖²` 在 `λ_new` 边界
> 附近会因 fp16 舍入噪声翻转 cluster assignment——标定依据和实际路由输入精度不
> 一致。`unused/semantic_stage0_dump.py` 的 `--save_dtype` 默认值和
> `litgpt/semantic_s0.py` 的 `Stage0DumpRecorder` 默认值已经改成 `float32`；
> `float16` 仍是显式可选项（接受这个风险换存储体积），但**不是默认值，这张表
> 不应该继续把 fp16 写成唯一/默认选择**。32k、5 层子集下 fp16 约 700MB，fp32
> （新默认）约 1.4GB（见下方"层子集"一节），换算方式不变，只是单价翻倍。

> **不要 dump 完整 attention，也不需要 dump `q_raw`。** `(nh, T, T)` 在 32k 下是
> 16×32768² ≈ 172 亿个元素，落盘不可行，机制 B 的分块计算天生避开了它。`q_raw`
> （旋转前的 query）在这份规格里没有任何用途——**聚类只在 k 空间做**，从不碰
> `q`，所以只需要 `q_roped`，不必再引入一个 `q_raw` 制造歧义。

**维度约定**：
- **`B` 固定为 1**。batch 维保留会引入 padding 对齐问题，而 Stage 0 不需要吞吐。
- `G` 维**必须保留且分开统计**——§2.5 的逐头要求就落在这里，聚合会掩盖"少数头有效"。
- 层维同理（§11-I）。

**层子集**：按当前默认的 fp32 落盘（见上方"更正"框），全 28 层 × k+v 的体量是
`28×8×32768×128×4B×2 ≈ 7.5 GB/条 prompt`，4 条就 30 GB。**第一轮先 dump 层子集**
（建议 `{0, 7, 14, 21, 27}`，覆盖浅/中/深），约 1.4 GB；只有当子集显示层间差异
确实影响结论时，才做全量。（若显式传 `--save_dtype float16` 换回旧的存储优化，
这两个数字对半。）

**needle span 对齐**（最容易出错的一处）：
- span 必须是**分词后的绝对 token 下标**，与 `k_raw` 的 `T` 轴同一坐标系；
- NIAH 构造器给的是字符区间，需要经 tokenizer 的 offset mapping 转换，**转换代码要和
  eval 用的是同一份**，否则 S0.3 的隔离率统计全是噪声；
- 复用另一分支 `log_kv_pin_diag.py` 已有的定位逻辑，不要重写。

**落盘格式**：每层一个 `.npy`（或单个 `.npz`），外加一份 JSON manifest。
**manifest 是后续所有分析的唯一真相来源，schema 必须在这里集中列全，
不能散落在各处分析里各自约定各自需要的字段**——本文档后文陆续为不同
分析引入了新的必需字段，不集中到这一处很容易漏记，导致复现实验时缺
某个字段而不自知：

- 基础信息：prompt id、tokenizer、序列长度、needle 的 token span、
  层/头索引、模型 config hash；
- **`s_h` 标定值**（§5.21-4）；
- **`tail_query_count`**（S0.8 3b 的尾部窗口大小，见下方 S0.8 一节）
  ——直接影响 3b 那个 5% 数字本身，是复现它**必需**的字段；
- **MinHash 三元组：`hash_algorithm`（`"splitmix64-v1"`）、`k`
  （即 `WARD_EVENT_SKETCH_K`）、`master_seed`**（`algorithm-spec.md`
  §5.4"S0.8 的 Ward 事件比较需要合并前的成员快照"一节）——三者共同
  决定两次 dump 的 Ward 事件 Jaccard 估计是否可比，缺一都不算完整
  复现。

  `block_size`（机制 B 的分块粒度）**不属于这份"必需复现"清单**：
  `k_expanded` 从不按 `block_size` 切片，`softmax` 永远在完整 key
  维度上做，改 `block_size` 不改变 `attn_mass_by_dist` 或 3b 的任何
  输出值，只改变跑多快、峰值内存多大——记录它对调优/复现运行时表现
  有用，但缺了它不影响"能不能认定复现了同一个实验结果"，不要和上面
  几个字段混进同一个"必需"清单里。

| 编号 | 测什么 | 为什么 |
|---|---|---|
| **S0.0** | **扫 `(g_max, ℓ_block)`（固定 `η=0`，理由见 `algorithm-spec.md` §5.3 的更正框——`η>0` 会让"纯语义聚类"端点混入未受控的时序 tie-break）：一端纯语义聚类，一端完全连续分段。看槽内内容方差与锚点跨度的联合曲线** | **全课题最根本的问题：收益来自语义分组本身，还是仅仅来自更好的分段边界？** |
| S0.1 | §5.14 单测（含 mass bias **计数**守恒、`w=0` 恒等元）| 数学正确性，纯 CPU，不需要 dump |
| S0.2 | **`K_eff(n)` 整条曲线**（n 从 1k 到 32k）+ 每簇 segment 数，拟合饱和/log/幂律。**测两个口径,不是一个**：①纯 DP-means（不含 `η`/`g_max`/`γ`，不设 `K_max` 上限，离线 CPU 分析，衡量"内容本身有多少语义多样性"，喂曲线形状判定）；②生产三路路由（`η`/`g_max`/`γ` 按生产默认值打开，只是不设 `K_max` 上限，衡量"这套算法实际会尝试开多少簇"，喂 `algorithm-spec.md` §5.6 的 `c` 决策规则）。两者都与生产路径里被 `K_max` 截断后的实际簇数（§5.6）是不同的量，后者永远 `≤ K_max` | 预算故事成不成立；外推到 1M；**并决定 §5.11 走 Fenwick 还是扁平贪心** |
| S0.3 | needle 隔离率：所在簇的成员数分布（关键是 `≤ B′` 的比例）| §3 + §5.9 的核心机制成不成立 |
| S0.4 | 簇内 **key 方差**与 **value 方差**（两个都要）/ 现有位置槽内方差 | key 方差管分数侧，**value 方差管读出侧**，只测前者会高估收益（§11-D）|
| S0.5 | entry 的 `(p_hi − p_lo)` 跨度分布，随 `(g_max, ℓ_block)` 变化 | 验证段机制确实压住了跨度 |
| S0.6 | 锚点去重后的平均倍数 `E[M]` 与分布 | 值不值得为未来 gather/packed 实现投入（§4）——v1 本身的读出槽池是固定宽度，不受 `E[M]` 影响 |
| S0.7 | 簇内 value 的并存 vs 作废比例，**分相邻/远距离统计** | §2.4 开放问题的判定实验 |
| S0.8 | 批量化路由（§5.4）与严格串行版的分歧率——**拆成三项**（cluster assignment/Ward 事件、segment+PAD 开销、最终 cache/readout 误差），定义见下方决策门 | 那个近似能不能用 |

**决策门**：
- **S0.0**：如果"完全连续分段"已经拿到大部分收益，语义聚类这条线的边际价值有限，
  应当直接转向更简单的"语义分段"方案（工程量小一个量级、不需要 CPT）。
  **这个门开在最前面，就是为了避免在错误的复杂度上投入。**
  > **首次实证结果（2026-08-21，首轮跑，样本/层覆盖有限，不是最终确认）**：
  > 对比 `single_cluster_bprime_baseline`（同 ladder 机制、不聚类的对照组，
  > 即这道门真正要问的问题）,语义聚类的 key/value 方差中位数分别 ≈0.44×/
  > 0.71×,40/40 layer×group 全赢。**初步判定：不转向纯分段。** 完整数字、
  > 连带发现（entry 数 unclipped 涨 ~32–37×，跨 K_max 校准要用）、限定条件
  > 见 `CLAUDE.md` §14 2026-08-21 条。
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
  > **首次实证结果（2026-08-21，首轮跑，样本/层覆盖有限，不是最终确认）**：
  > `stage0_dump/s0_3.csv` 覆盖 `lambda_rel={0.875,1.0,1.125}` 与
  > `g_max={inf,8192,4096,2048,1024,256}`。全部候选的 token-level lift
  > 均显著高于随机（约 7–8×），所以 **S0.3 没有触发"就地停止"**。但
  > 绝对隔离率仍有限：`lambda_rel=1.0` 的 `needle_token_isolated_rate`
  > 约 11.7–12.3%，`span_any_token_isolated_rate` 约 61.5–64.0%，
  > `span_all_tokens_isolated_rate` 约 0.065–0.13%；`lambda_rel=0.875`
  > 把 token 召回提高到约 17.6–18.9%、`span_any` 提高到约 70.8–72.8%，
  > 但随机隔离率也从约 1.5% 涨到约 2.5–2.6%，且簇/entry 成本同步上涨。
  > 因此当前读法是：**语义信号成立，经常能从 needle span 里捞出至少一部分
  > token；但还不能指望完整 span 被隔离。**
  >
  > 同一 `lambda_rel` 内扫 `g_max` 的信息量很小：`g_max` 收紧不会提升
  > needle 隔离率，反而主要把 `segment_count_mean` 推高（如 `lambda_rel=1.0`
  > 下 `g_max=inf` 的 segment 均值约 311，`g_max=256` 涨到约 1672）。这也
  > 校正了此前"g_max 完全不影响簇归属"的过强说法：代码上 `g_max` 触发新
  > segment 时会通过 `gamma` 衰减改变后续 centroid 更新，所以 cluster_count
  > 可有轻微变化；但实测影响远小于 `lambda_rel`。
  >
  > **下一轮 gate（已实现，待实跑）**：S0.3 必须补 `K_max` + Ward clipped
  > 口径。核心字段是 `K_max_binding_rate`、`needle_token_isolated_rate`、
  > `span_any_token_isolated_rate`、`needle_token_ward_touched_rate` 和
  > `needle_cluster_merged_by_ward_rate`（列名以 CSV 实际输出为准）。如果
  > `lambda_rel=0.875` 的高召回主要来自大量新簇，但在默认附近的 `K_max=15/16`
  > 或 `K_max=32` 下高频绑定、Ward 反复触碰 needle 小簇，它不能进入生产主线。
- **S0.4**：key 方差应显著低于现有位置槽；**若 value 方差没有同步下降**，说明读出侧
  仍是 smear，收益要打对折，需要考虑按 `[k;v]` 联合聚类或簇内二次分裂。
- **S0.6（更正：不再是"预测会不会超过 vanilla"的决策门，那件事已经确定，
  与 `E[M]` 无关）**：CLAUDE.md §4 已经更正——`dedup_anchors` 固定返回
  `(..., S, 3)`，v1 读出时的物理槽池恒为 `entry × 3 = 4984`（不随 `E[M]`
  变化，无效/重复锚点靠 `slot_valid` 掩码而非收缩张量宽度剔除），`4984 >
  vanilla 3584` 是当前（无 gather）设计的**确定结果**，不是"`E[M]` 不好才
  会发生"的风险。S0.6 真正测的是**这项差距值不值得投入未来的 gather/
  packed 优化**：`E[M]` 越接近 1（越多 entry 去重后只剩 1~2 个锚点），
  gather 能把物理槽数拉回接近 `E[M]` 对应的逻辑锚点数（重新回到 vanilla
  之下）的空间就越大；`E[M]` 越接近 3，gather 的收益越小。**"砍掉第三
  锚点只留 `(p_lo, p_hi)`" 是一条独立于 `E[M]` 测量结果的设计选择**（把
  固定宽度从 3 降到 2，`1024+1320×2=3664`，仍然超过 vanilla 的 3584，只是
  差距更小），不是"`E[M]` 测出来逼近 3 之后才该考虑"的条件退路——两者是
  解决同一问题的不同思路，报告 S0.6 结果时不要把"实测 E[M]" 和"要不要退到
  两锚点"包装成因果关系。
  > **首次实证结果（2026-08-21，与 S0.3 同批 dump，首轮跑）**：
  > `stage0_dump/s0_6_rel1.csv` 与 `stage0_dump/s0_6_rel0.875.csv` 分别
  > 对应 `lambda_rel=1.0` 和 `0.875`。`lambda_rel=1.0` 的 `E[M]` 约
  > 2.04–2.10，`entry_count_mean_ratio_vs_single_cluster` 约 30.7–39.9×，
  > 当前 fixed-3 物理宽度约为 vanilla full 的 1.96–2.55×；`lambda_rel=0.875`
  > 的 `E[M]` 更低（约 1.93–2.00，说明更多小 entry 的 anchor 可去重），但
  > entry/fixed3 总成本更高：entry 相对 single-cluster 约 44.8–59.8×，
  > 物理宽度约为 vanilla full 的 2.87–3.83×。所以 **不要按最低 `E[M]`
  > 选配置**；低 `E[M]` 可能只是 entry 更碎、小 entry 更多，并不代表整体更省。
  >
  > 当前配置判断：`lambda_rel=1.0, g_max=inf` 是默认主线（S0.3 lift 约
  > 8.27×，S0.6 物理宽度约 1.96× vanilla full）；`lambda_rel=0.875,
  > g_max=inf/8192` 是高召回候选（S0.3 token 召回约 18.5–18.9%、span_any
  > 约 72.5–72.8%，但 S0.6 成本约 1.45–1.5× 于 `lambda_rel=1.0`）。`l_block`
  > 不宜过大，尤其 `l_block=2/3` 会显著增加 pad entry 和 fixed3 宽度。
  >
  > **下一轮 gate（已实现，待实跑）**：S0.6 同样必须带 `K_max`。除 `E[M]` 外，
  > 必须同时报告 `entry_count_mean`、`fixed3_anchor_count_mean`、
  > `current_scheme_physical_slot_count_mean_ratio_vs_vanilla_full`、
  > `ward_merge_count_mean`、`K_max_binding_rate` 和
  > `gather_savings_fraction_vs_fixed3`。
  > 选择配置时不能只按最低 `E[M]` 排名；`E[M]` 低但 entry 数暴涨，仍然是更贵的
  > 配置。

  推荐的 clipped gate 命令（worker 数按机器物理核心和实际吞吐调整，CPU 负载型任务
  通常先用 64 比盲目开满更稳）：

  ```bash
  python unused/semantic_s0_needle_isolation.py \
    --dump <stage0_manifest.json> \
    --output <out_dir>/s0_3_kmax_ward.json \
    --lambda_rel 1.0,0.875 \
    --g_max inf,8192,4096 \
    --k_max unclipped,15,16,32,64,128 \
    --b_prime 8 \
    --workers 64 \
    --parallel_unit group \
    --log_timing

  python unused/semantic_s0_anchor_dedup.py \
    --dump <stage0_manifest.json> \
    --output <out_dir>/s0_6_kmax_ward_rel1.json \
    --lambda_rel 1.0 \
    --g_max inf,8192,4096 \
    --l_block 0,1 \
    --k_max unclipped,15,16,32,64,128 \
    --b_prime 8 \
    --workers 64 \
    --parallel_unit group \
    --log_timing

  python unused/semantic_s0_anchor_dedup.py \
    --dump <stage0_manifest.json> \
    --output <out_dir>/s0_6_kmax_ward_rel0875.json \
    --lambda_rel 0.875 \
    --g_max inf,8192,4096 \
    --l_block 0,1 \
    --k_max unclipped,15,16,32,64,128 \
    --b_prime 8 \
    --workers 64 \
    --parallel_unit group \
    --log_timing
  ```

  实现审计状态：Ward ladder 合并已回归锁定为 `native+ejected` 整体按 `order` 排序；
  S0.6 analyzer 的 best-by-layer baseline 已改为按 layer 聚合，不再被多 KV group 的
  baseline 行覆盖。当前环境缺 `torch`，完整 pytest 未跑；目标脚本、analyzer、相关
  回归和 toy dump 端到端已通过。
- **S0.7**：若簇内 value 系统性作废的比例 > 30%，把 Γ 的 delta-rule 广义化提到
  Stage 1 范围内；否则记录结论并搁置 §2.4。
- **S0.8**：分歧率不是一个单一标量，必须拆成三项分别报告，理由是它们诊断的是
  不同层级的问题、且互相之间不是线性关系（cluster 分歧可能被 segment/pad 开销
  放大或抵消，两者都不直接等于最终读出误差）：
  1. **cluster assignment divergence（含 Ward 事件）**：

     > **前置校验（这一轮补的），必须先做**：`scan_op_log` 自己不做任何
     > 合法性检查（`algorithm-spec.md` §5.4"不改 `scan_op_log` 本身"一节
     > 明确这是刻意的设计，换取"足够简单、容易独立确信正确"），`NEW_
     > CLUSTER` 是否写进真正的空槽、`JOIN`/`NEW_SEGMENT` 是否指向 alive/
     > root 身份、`WARD_MERGE` 是否 self-merge 或引用 stale/non-root 操作
     > 数，这些全部不检查——一份非法的 `op_log` 会被解析成一组"看起来合法"
     > 的最终槽号，co-assignment/ARI/precision/recall 会在错误的输入上
     > 算出干净但没有意义的数字，且没有任何报错信号提示这一点。**在对
     > `cache_batch`/`cache_serial` 各自的 `op_log` 调用
     > `scan_op_log`/`resolve_final_slots` 之前，必须先对同一段 `op_log`
     > 各跑一遍 `scan_op_log_for_ward_events`（`algorithm-spec.md` §5.4，
     > 现在已经补齐了五类合法性断言：紧邻关系、缺失身份、self-merge、
     > stale/non-root 操作数、K 未满分支选中已占用槽），只要它跑完不
     > 抛异常就说明这段 `op_log` 满足全部已知的合法性约束**——它的
     > `WardEvent`/sketch/size 输出在这一步不需要用（第②小项"Ward 事件"
     > 已经在别处独立消费它们），只是借用它的断言做一次前置校验，通过后
     > 再放心调用 `scan_op_log`/`resolve_final_slots`。这个前置校验和下面
     > "Ward 事件"那一小项的计算**可以共享同一次 `scan_op_log_for_ward_
     > events` 调用**，不需要为了校验单独再跑一遍。

     用
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

     **仅凭这个原始一致率（下称 raw agreement）还不够：`K_max` 不大时，
     "两条路径都判不同簇"这一类 pair 会天然占绝大多数，把分数往 1 附近
     推，掩盖真实分歧，必须并列报告经过校正的指标。** 具体量级：
     `K_max` 在 32k 下默认是 15（§5.6），若两条路径各自把 token 大致
     均分成 15 个簇，即便两条路径的聚类**完全独立、互不相关**，随机
     情况下"两者都不同簇"这一类 pair 的比例期望约 `(1−1/15)²≈87.1%`，
     "两者都同簇"约 `(1/15)²≈0.4%`，两者相加，raw agreement 在完全
     无关的两个聚类之间也能到 **≈87.6%**——这个数字本身不能说明批量
     近似和严格串行参考**真的**一致，只是说明"很多 pair 靠『两边都判
     不同簇』这种容易凑巧对上的方式蒙对了"，`K_max` 越小这个基线越高、
     越容易把真实分歧藏进去。**必须并列报告 Adjusted Rand Index
     （ARI）**，用同一张列联表 `n_{ab}`/`a_i`/`b_j` 免费算出，不需要
     额外遍历：

     ```
     sum_nab  = Σ_{ab} C(n_{ab},2)          # 已经在算 agree_pairs 时算过
     sum_ai   = Σ_i C(a_i,2)                  # 同上
     sum_bj   = Σ_j C(b_j,2)                  # 同上
     expected = sum_ai * sum_bj / C(T,2)
     max_idx  = (sum_ai + sum_bj) / 2
     ARI = (sum_nab - expected) / (max_idx - expected)
     ```

     ARI 对"随机蒙对"做了归一化（两个独立随机聚类的期望 ARI 是 0，完全
     一致是 1），不会像 raw agreement 那样被 `K_max` 偏小、"大多数 pair
     天然不同簇"这类结构性因素撑高。**同时并列报告 same-cluster 口径的
     precision/recall/F1**（只看"至少一条路径判同簇"这个子集，天然回避
     "两边都判不同簇"这一类被小 `K_max` 放大的多数 pair，对 ARI 的结论
     做交叉验证，且能看出分歧的具体方向）：

     ```
     precision = sum_nab / sum_ai   # 批量路径判同簇的 pair 里，严格串行
                                       # 参考也判同簇的比例
     recall    = sum_nab / sum_bj   # 严格串行参考判同簇的 pair 里，批量
                                       # 路径也判同簇的比例
     f1        = 2 * precision * recall / (precision + recall)
     ```

     **上面三个公式都有零分母边界，32k 下正常路由几乎不会撞见，但小合成
     场景的实现单测（极端场景：全 singleton、两条路径都退化成单个大簇）
     会撞见，必须显式定义，不能让 NaN 悄悄混进报告——这正是本文档一贯在
     防的那类问题，不能因为"实践中大概率不会发生"就在这里放过去**：
     - **ARI 的 `max_idx - expected == 0`**：只在 `sum_ai == sum_bj`
       且两者都取到各自可能的两个极值之一时发生——精确对应"两条路径的
       聚类形状退化成完全相同的极端结构"（都是全 singleton，或都是单个
       大簇纳入全部 token）。经验证 `sklearn.metrics.adjusted_rand_
       score` 在这类输入上返回 `1.0`（用 `all_singleton`/`all_same_
       cluster` 等场景实测确认，不是凭印象转述）——两条路径退化成同一种
       最简单结构，判"完全一致"符合直觉，不是需要回避的边界，本规格
       采用相同约定：**`max_idx == expected` 时 `ARI := 1.0`**。
     - **precision 的 `sum_ai == 0`**：批量路径把每个 token 都分进了
       独立的单点簇，"批量路径判同簇的 pair"这个集合本身是空集，此时
       `sum_nab` 也必然是 0（`sum_nab ≤ sum_ai` 恒成立），precision 是
       真正的 0/0，没有自然的数值可填——**定义为未定义（报告
       `NaN`，标记 `precision_undefined=True`）**，不强行给一个会被
       误读成"精确"的数字。
     - **recall 的 `sum_bj == 0`**：对称情形（严格串行参考退化成全
       singleton），**同样定义为未定义（`NaN`，
       `recall_undefined=True`）**，独立于 precision 是否也未定义
       （两条路径可能只有一条退化）。
     - **f1**：`precision`/`recall` 任一未定义则 `f1` 未定义
       （`NaN`）；两者都定义且都精确为 0（`sum_ai>0` 且
       `sum_bj>0` 但两条路径没有任何共同的同簇 pair）时
       `f1 := 0`（标准调和平均"两个已定义的 0 调和平均仍是 0"的约定，
       不是另一个 0/0——`f1` 公式本身只有在 `precision+recall==0` 时
       才有除零风险，而这恰好就是这个分支覆盖的情形）。

     raw agreement、ARI、`(precision, recall, f1)` 四组数字一起报告，
     不用其中一个代替另一个——raw agreement 直觉最直接但容易被小
     `K_max` 撑高，ARI 修正了这一点但数值本身不直观（可以为负），
     precision/recall 能看出分歧具体偏向"批量路径过度合并"（precision
     低，批量路径把本该分开的簇并到了一起）还是"批量路径过度拆分"
     （recall 低，批量路径把本该同簇的 token 分开了），三者合起来看
     比任何单一数字更不容易被误读，也更方便和 Ward 事件那一类专门诊断
     "过度合并"的指标互相印证。
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
       `best_orientation_jaccard(event_a, event_b)`（不是直接调
       `estimate_jaccard` 分别比较 `keep`/`free`）估计两侧的 Jaccard
       相似度——**`keep`/`free` 是物理保留槽/释放槽这个实现细节的产物，
       不保证两条路径方向一致**：两条路径若合并的是同一对语义簇但选了
       相反方向的 keep/free（一条把 C 当 keep、另一条把 X 当 keep），
       直接按标签比较会把两个不同的簇错误地凑一起，产出虚假的低 Jaccard，
       即便这次合并本身是强对齐（`trigger_token_idx` 相等）。
       `best_orientation_jaccard` 把两种配对方向都试一遍、取总相似度更高
       的一种，返回 `side_1_jaccard`/`side_2_jaccard`（不再叫
       keep/free——它们只在这一次结果内部有意义，不是跨路径可比的固定
       标签）、`orientation_flipped`（是否选中了交换方向）、
       `orientation_margin`/`orientation_ambiguous`（两种方向总分之差，
       以及这个差是否小到分不清方向——`algorithm-spec.md` 同一节已经
       论证过单个 Jaccard 估计本身就有 ≈0.5/√k 的标准误差，两种方向
       总分接近时 `orientation_flipped` 可能只是抽样噪声，`ambiguous`
       为真的事件不该被当作"方向真的翻转了"去归因，只把 `side_1/2_
       jaccard` 当点估计使用），四者都要报告，不要只留相似度丢掉方向
       信息——翻转（且不 ambiguous）本身也是一个诊断信号（翻转频繁
       可能说明两条路径的 Ward 候选选择存在系统性差异）。这是
       一个**估计值**，标准误差上界 `0.5/√k`（默认 `k=128` 时 ≈4.4%），
       `k` 必须和这个数字一起写进 eval metadata；连带报告
       `keep_size_before`/`free_size_before`（精确整数，不经估计，用来
       判断一个偏低的 Jaccard 发生在大簇还是小簇上——size 本身不受
       keep/free 方向影响，两条路径各自的 `(keep_size, free_size)` 无序对
       仍可直接比较）；
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
     - **3b（决策依据，唯一的硬性决策门）：两条路径各自处理到
       `cutoff = T − tail_query_count` 为止、把这一步的 cache 状态当作
       冻结前缀，再把尾部 `tail_query_count` 个 token 当作真正意义上的
       in-flight chunk 拼接上去，做一次真实 attention 读出，比较批量
       路径与严格串行参考给出的注意力输出的相对 L2 误差**（具体构造见
       下方 blockquote 的"更正"）——是针对这一个 `cutoff` 参考点的
       **单次**测量，不是逐个 flush 批重复测量再平均，也不是"每处理完
       一批就测一次"的意思（具体用哪些 query、选多大范围，见下方
       blockquote）。**这一项不需要任何簇
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
       > 实现做（`algorithm-spec.md` 的实现单测——不是 S0.1，见其"实现单测"
       > 更正框——已经在用同一类参考实现，不需要额外 GPU）。**3b
       > 真正要用到 q 的地方只有"拿这两份已经构造好的 cache 状态各做
       > 一次 attention 读出"这一步，只用序列尾部一个固定大小的窗口——
       > 窗口大小是独立的 `tail_query_count` 参数，不派生自
       > `block_size`（见下方"更正"）。**
       >
       > **更正（这一轮修的，P0）：上一版"序列尾部的 query 因果地有权
       > 看到整个 cache"这句话只对 `tail_query_count=1`（字面意义上的
       > 最后一个位置）成立，`tail_query_count>1` 时是错的，设计必须
       > 重做，不能只改措辞。** 反例：尾部窗口里任意一个非末尾位置
       > `p ∈ [T−tail_query_count, T−1)`，真实自回归 serving 下它只该
       > 看到"刚摄入 token p 那一刻"的 cache 状态；但"两条路径各自处理
       > 完整条序列之后的最终 cache 状态"已经包含了 `p` 之后全部
       > `T−1−p` 个 token 的压缩贡献。
       >
       > **上一版曾经尝试证明"只要 `recent_count ≥ tail_query_count`，
       > 用最终 cache 就安全"——这个精确度不够，必须再收紧一轮。** 那版
       > 论证止步于"pooled 里的原始 token 不含来自 `p` 之后的内容"（可以
       > 证明成立：给定 `recent_count ≥ tail_query_count`，任意 pooled
       > 成员的位置都 `< T−recent_count ≤ T−tail_query_count`，早于每一个
       > 尾部 query），但这只排除了**内容泄漏**（不该出现的原始 token
       > 混进读出集合），没有排除**表示粒度泄漏**：`p` 和 `T−1` 之间还有
       > `T−1−p` 个 token，它们在真实"处理到 p 那一刻"根本还没被摄入，
       > 但在最终 cache 里已经把 recent window 挤过了一轮——`p` 之前那些
       > 原本"as of p"还该待在 recent window 里、以精确 `w=1` 槽形式
       > 存在的 token，到了最终 cache 里可能已经被后续（`p` 之后）的
       > flush 事件挤出、压缩进了 pooled 层级。这些内容依然**早于** `p`
       > （不违反因果，读出结果不会看到未来 token），但**被更粗的粒度
       > 代表了**——用最终 cache 服务尾部窗口里除最后一个 query 外的
       > 任何 query，读出用的表示和"真实 as-of-p 状态"不是同一个东西，
       > 会让 3b 测量的到底是"批量近似 vs 严格串行的路由分歧"还是"用了
       > 错误粒度表示的副作用"变得含糊——即使两者都不改变`<5%`判定的
       > 方向（下面会说明为什么这层差异不足以让已经跑过的历史结论作废），
       > 继续依赖一个需要这么多篇幅才能讲清楚"为什么安全"的构造，不如
       > 换一个从构造上就不需要这层论证的方案。
       >
       > **修法：不再复用最终 cache，3b 为自己单独构造一对"只处理到
       > `cutoff = T − tail_query_count`"的 CPU 参考 cache（`cache_
       > batch_prefix`/`cache_serial_prefix`），把尾部当成真正意义上
       > "尚未提交"的 in-flight chunk 通过 `append_exact_tokens` 拼上去，
       > 再用 `causal_tail=tail_query_count` 读出。** 这精确复现了
       > `p=cutoff`（尾部窗口里最早的那个 query）此刻的真实 cache 状态
       > ——`[0, cutoff)` 已提交、`[cutoff, T)` 是这一次 forward 正在
       > 处理、还没写回 cache 的当前 chunk，和生产路径
       > `CausalSelfAttention._log_kv_training_forward()` 处理任意一个
       > 流式 chunk 时的状态完全同构（都是"冻结的已提交前缀 + 当前 in-
       > flight chunk"）。`causal_tail` 的三角掩码则保证尾部窗口内部
       > 每个 query 只看到自己和更早的尾部位置，不看到更晚的——三点合起来
       > （前缀严格早于 `cutoff`、`causal_tail` 挡住尾部内部的"未来"、
       > 附加的尾部 token 从未被压缩过，精确保留）对尾部窗口里的**每一个**
       > query 都精确成立，不再需要"pooled 内容是否碰巧早于 p"这类需要
       > 额外证明的不变量，也不再需要 `recent_count ≥ tail_query_count`
       > 这条前提——这条前提连同它要求的"两条路径分别断言""硬失败换更小
       > 的 `tail_query_count`"等配套机制一并作废，不再需要维护。
       > **`cache_batch_prefix`/`cache_serial_prefix` 是 3b 专用的一对
       > 独立实例**，用同一套 §5.3/§5.4 CPU 参考实现构造，只是喂给它们
       > 的输入截短到 `k_raw[:cutoff]`/`v[:cutoff]`/`pos[:cutoff]`——不
       > 是 S0.8 第 1/2/3a 项比较用的那对 `cache_batch`/`cache_serial`
       > （那对依然处理完整条序列，1/2/3a 关心的是"完整序列处理完之后
       > 的最终路由/cache 结构分歧"，用完整序列是对的，不受这条更正
       > 影响）。**惟一新增的前置条件是 `T > tail_query_count`**（否则
       > `cutoff ≤ 0`，前缀为空，需要另挑一条更长的 prompt，同样硬
       > 失败不做静默处理）——这比原来的 `recent_count ≥ tail_query_
       > count` 更弱、更不容易触发。
       >
       > **`k_tail_roped` 需要现算，不能照抄前缀部分的现成物化逻辑**：
       > dump 只落盘 pre-RoPE 的 `k_raw`（机制 A），尾部 `[cutoff, T)`
       > 这批 token 要作为 in-flight **精确**槽拼接，按既有约定
       > （`algorithm-spec.md` §5.14"in-flight chunk 必须用 post-RoPE
       > k"）必须是 post-RoPE。这不需要模型前向的中间变量——`apply_rope`
       > 是纯位置索引的函数，尾部每个 token 的绝对位置已知（`cutoff`
       > 到 `T−1`），直接对 dump 出来的 `k_raw[cutoff:T]` 在各自绝对
       > 位置上调用标准 `apply_rope` 即可得到 `k_tail_roped`，和
       > `materialize_anchor_keys` 给单点（`M=1`）entry 物化 key 用的
       > 是同一个原语，不是新写一套逻辑。`v_tail` 不需要 RoPE，直接取
       > `v[cutoff:T]`。
       >
       > CPU 参考实现（`cache_batch_prefix`/`cache_serial_prefix` 和
       > S0.8 其它子项共用的 `cache_batch`/`cache_serial`）依然必须真正
       > 维护一个 recent window 缓冲区，不能只做 `op_log` 重放/只保留
       > 压缩后的层级——这是"滑窗、按到达顺序追加、溢出时把最老的
       > `flush_granularity` 个 token 推出去做压缩"这条既有语义（同
       > CLAUDE.md §10）对任何 CPU 参考实现的通用要求，`algorithm-
       > spec.md` §5.4"必须补的单测"用到的同一套参考实现同样要满足，
       > 不是 3b 专属的新增负担。
       >
       > **更正（这一轮修的，P0）：具体做法上一版还有两个坑，不是加了
       > `causal_tail` 就完事——这两条在新构造下依然成立，一并保留。**
       > ①**忘了真正拼接 in-flight chunk，`causal_tail` 会遮错对象**：
       > 必须把 `k_tail_roped`/`v_tail` 真正通过 `append_exact_tokens`
       > 拼进 `cache_batch_prefix.get_attention_state()` 的返回值，不能
       > 只是算出来却不用——`causal_tail` 遮住的是"调用方传入的张量
       > 最后 `causal_tail` 个位置"，如果那里不是真正拼接过的尾部
       > token，三角因果掩蔽的就是不相关的内容。②**`get_attention_
       > state()` 的返回值本来就是"压缩 levels（老到新） + recent
       > window（原始顺序，`w=1` 精确槽）"拼接后的完整状态**——`cache_
       > batch_prefix`/`cache_serial_prefix` 各自的 recent window（可能
       > 还留着 `cutoff` 之前的一些精确 token）已经在这份返回值里，
       > `append_exact_tokens` 只需要把**尾部**（`[cutoff, T)`）接在
       > 后面一次，不要重复拼接已经在返回值里的内容。
       >
       > **更正（这一轮修的，P0）：3b 的比较对象一度被错误地混进了"稠密
       > 侧 ground truth"，必须收回，不是措辞问题，是要删掉一整段设计。**
       > 上一版在这里加了"稠密侧（mechanism B 给出的 ground truth）...
       > 用来算 3b 的分子/分母另一半"——这和 3b 从一开始的定义（本节
       > 最上面："比较批量路径与严格串行参考给出的注意力输出的相对 L2
       > 误差"；S0.8 的条目定义同样只说"批量化路由与严格串行版的分歧率"）
       > 直接矛盾：**3b 比较的自始至终是两个压缩 cache（批量近似、严格
       > 串行参考）各自的读出结果，不涉及任何稠密 attention 输出。**
       > 这处矛盾的根源是把两件不相关的事混成了一件——"3b 蹭机制 B 的
       > 循环拿 `q_roped`，省一次重复算 RoPE 的开销"和"3b 拿机制 B 算出
       > 的 dense score/probs 当比较对象"——前者是设计里一直都有的，
       > 后者从来不是。**修法**：`q_tail` 只从机制 B 循环里"顺路"累积，
       > 机制 B 自己的 `scores`/`probs`/`attn_mass_by_dist` 全程只服务
       > 它自己原来的目的（§13.2 的距离-质量曲线），3b 拿到完整
       > `q_tail` 之后在循环**外面**独立做两次压缩侧读出、互相比较，和
       > 机制 B 的 dense 计算结果没有任何关系。
       >
       > **更正（这一轮修的，P0/P1）：上一版这段伪代码本身还有三处会让
       > 实现直接崩溃或悄悄算错的问题，光是接上 `causal_tail`、剔除
       > dense 比较还不够。**
       > ① **`log_kv_slot_attention` 调用传的是错误的位置参数。** 现有
       > 签名（`algorithm-spec.md` §5.14"虚拟槽展开必须扣上
       > `causal_tail`/`mask` API"一节）是
       > `(q, slot_k, slot_v, slot_w, scale, mask=None, lam=1.0, causal_tail=0,
       > slot_valid=None, M_s=None, ...)`——`scale` 排在 `slot_w` 之后、
       > `mask` 之前，`lam` 排在 `mask`/`causal_tail` 之间（**这一轮补的**：
       > 上一版这里的签名片段漏抄了 `lam`，容易让人以为它被移除了——它一直
       > 都在，下面的调用也一直显式传它，只是这个片段本身抄漏了），
       > `slot_valid`/`M_s` 是排在更后面的具名参数。上一版
       > `log_kv_slot_attention(q_tail, *cache.get_attention_state(),
       > causal_tail=...)` 把 `get_attention_state()` 返回的 5 元组
       > `(slot_k,slot_v,slot_w,slot_valid,M_s)` 整个展开成位置参数，
       > 第 4、5 个位置会把 `slot_valid` 误传成 `scale`、把 `M_s` 误传成
       > `mask`——两者类型都不对（`scale` 该是 `float`，收到一个 bool
       > 张量；`mask` 该是 `(T_q,S)` bool 或 `None`，收到一个 int 张量），
       > 而真正的 `slot_valid`/`M_s` 关键字参数反而没被传上，各自取默认值
       > `None`。②**`tail_mask`/`block_start` 全部按错了轴索引。**
       > `q_roped` 是 `(nh,T,hs)`（"Stage 0 dump 规格"一节的维度约定：
       > `B` 固定为 1，直接省略这一维），`chunks(...)` 按 T 轴（dim 1）
       > 切块，但 `len(query_block)` 在 PyTorch 里对一个张量返回的是
       > `.shape[0]`，也就是 `nh`（16）而不是这一块的 T 长度——
       > `abs_idx`/`block_start` 全部算错；`query_block[tail_mask]`
       > 同样是按 dim 0（`nh` 轴）索引，而 `tail_mask` 的长度是这一块的
       > T 长度，两者对不上，多数情况下会直接因形状不匹配报错，不会
       > 悄悄算错，但同样会阻塞实现。③**`q_tail` 缺 batch 维。**
       > `torch.cat(q_tail_parts, dim=-2)` 拼出的是
       > `(nh, tail_query_count, hs)`，但 `log_kv_slot_attention` 的 `q`
       > 要求显式 `(B, nh, T_q, k_dim)`
       > （`litgpt/log_kv_cache.py:1477-1481`）——dump 约定里 `q_roped`
       > 不带 batch 维，从它切出的 `q_tail` 也不带，必须显式补回去。
       > 三处一起修，`scale` 直接复用机制 B 循环里已经在用的同一个
       > `scale`（不是新引入的量）：
       >
       > ```python
       > q_tail_parts = []
       > block_start = 0
       > for query_block in chunks(q_roped, block_size):   # 机制 B 已经在跑
       >                                                      # 的循环，不
       >                                                      # 新增一次遍历；
       >                                                      # query_block:
       >                                                      # (nh, blk_len, hs)
       >     ...（attn_mass_by_dist 的既有累积，见"Stage 0 dump 规格"，原样不变）...
       >     blk_len = query_block.shape[1]        # T 轴是 dim 1，不是 dim 0
       >                                              # （dim 0 是 nh）——张量上
       >                                              # 的 len() 返回 shape[0]，
       >                                              # 这里必须显式取 shape[1]
       >     abs_idx = block_start + arange(blk_len)
       >     tail_mask = abs_idx >= T - tail_query_count
       >     if tail_mask.any():
       >         q_tail_parts.append(query_block[:, tail_mask, :])   # 按 T 轴
       >                                                                # （dim 1）
       >                                                                # 取子集，
       >                                                                # 不是按
       >                                                                # dim 0
       >                                                                # （nh 轴）；
       >                                                                # 可能跨
       >                                                                # 不止一
       >                                                                # 块，也可能
       >                                                                # 只是某一
       >                                                                # 块的一部
       >                                                                # 分，两种
       >                                                                # 情形都正
       >                                                                # 确累积
       >     block_start += blk_len
       >
       > q_tail = torch.cat(q_tail_parts, dim=-2).unsqueeze(0)
       >     # 先拼成 (nh, tail_query_count, hs)，再补一个 batch 维变成
       >     # (1, nh, tail_query_count, hs)——log_kv_slot_attention 的 q
       >     # 要求显式 (B, nh, T_q, k_dim)。3b 的两次 log_kv_slot_attention
       >     # 调用在循环外面、只做一次，不逐块调用。
       > q_tail = q_tail.cpu().float()
       >     # q_tail 来自本轮 GPU 前向的 q_roped，是 CUDA 张量；cache_batch/
       >     # cache_serial 是 algorithm-spec.md §5.4/§5.18 的纯 CPU 参考实现
       >     # 构造出来的，get_attention_state() 返回的是 CPU 张量。两者不搬到
       >     # 同一设备就直接相乘，log_kv_slot_attention 内部的矩阵乘法会立即
       >     # 报 device mismatch——这不是可以事后再补的细节，是这段代码能不能
       >     # 跑起来的前提。选择把 q_tail 挪到 CPU（而不是把 cache_batch/
       >     # cache_serial 挪到 GPU）：q_tail 只有 (1, nh, tail_query_count, hs)
       >     # 这么大，搬一次的代价可以忽略；反过来搬 cache 状态需要额外维护
       >     # 一条"参考实现也能在 GPU 上跑"的路径，而参考实现的全部存在意义
       >     # 就是"足够简单、容易独立确信正确"，不值得为这里的比较步骤破例。
       >     #
       >     # 更正（这一轮修的）：上一版说 dtype 不受 .cpu() 影响、留在 q_roped
       >     # 原始的 fp16/bf16——这在 CPU 上是乐观假设，不是安全默认值。fp16 在
       >     # CPU 后端的 matmul/softmax 支持历来不完整（不同 PyTorch 版本下常见
       >     # "not implemented for 'Half'" 这类报错，或者能跑但退化成极慢的逐
       >     # 元素路径），bf16 支持更好但也不是所有算子、所有版本都覆盖，3b 不
       >     # 应该依赖"这台机器的 PyTorch/CPU 后端恰好支持"这个未经检查的前提。
       >     # 改为显式 .float()（fp32）——fp32 在 CPU 上是全算子通用支持，不
       >     # 存在这一类阻塞风险。这不会让比较失真：3b 测的是"批量近似路由 vs
       >     # 严格串行参考"的分歧，跟这一步用什么精度算无关，用比生产更高的
       >     # 精度做这次比较只会让读出的数值噪声更小，不会掩盖真实分歧，
       >     # 和 relative_l2 自己已经对 out_batch/out_serial 做 .float() 是
       >     # 同一个精神。cache_batch/cache_serial 的 k̄_raw/v̄/w 同样要转
       >     # fp32 才能和 fp32 的 q_tail 相乘，不能指望参考实现"恰好"存的就是
       >     # 这个 dtype。
       > assert T > tail_query_count   # cutoff = T - tail_query_count 必须 > 0，
       >                                  # 否则前缀为空，换一条更长的 prompt
       > cutoff = T - tail_query_count
       >
       > # cache_batch_prefix/cache_serial_prefix：3b 专用的一对独立 CPU 参考
       > # cache，和 S0.8 第 1/2/3a 项用的 cache_batch/cache_serial 是不同实例
       > # ——用同一套 §5.3/§5.4 CPU 参考实现构造，只是喂给它们的输入截短到
       > # [0, cutoff)，不是完整的 [0, T)：
       > #
       > # 更正（这一轮修的，P1）：上一版写的 k_raw[:cutoff]/v[:cutoff] 按的是
       > # dim 0 切片，但"dump 什么"表（本节前面）已经钉死 k_raw/v 的落盘形状是
       > # (G, T, hs)——dim 0 是 KV group（默认 G=8），dim 1 才是时间轴 T。
       > # k_raw[:cutoff] 字面上会切掉除前 cutoff 个 KV group 之外的一切（当
       > # cutoff 是几千的 token 数量级、G 只有 8 时，这个切片要么整个越界、
       > # 要么静默切出一个形状对不上下游的怪张量），不是按时间戳截断。pos 是
       > # 例外：它是纯位置索引，形状 (T,)，没有 G 这一维，pos[:cutoff] 原来就是
       > # 对的，不用改。改成显式按 dim 1 取子集：
       > cache_batch_prefix  = build_cache_batch(k_raw[:, :cutoff, :], v[:, :cutoff, :], pos[:cutoff])
       > cache_serial_prefix = build_cache_serial(k_raw[:, :cutoff, :], v[:, :cutoff, :], pos[:cutoff])
       >
       > # 尾部 [cutoff, T) 作为 in-flight 精确槽，必须是 post-RoPE——dump 只有
       > # pre-RoPE 的 k_raw，用标准 apply_rope 在各自绝对位置上现算，和
       > # materialize_anchor_keys 给单点 entry 物化 key 是同一个原语。同样按
       > # dim 1（T 轴）取尾部子集，不是 dim 0：
       > # 更正（这一轮修的，P3）：apply_rope 要求 cos/sin 恰好是三维
       > # （见 litgpt/model.py:2081 的显式 `if cos.dim() != 3: raise
       > # ValueError`），但 cos_cache[cutoff:T]/sin_cache[cutoff:T] 是从
       > # 按位置索引的 cache 里切出来的，形状是 (tail, hs)，只有 2 维，
       > # 直接传会立即报错，不是静默算错。k_raw[:, cutoff:T, :] 本身已经
       > # 是 (G, tail, hs) 3 维（dim 0=G 权当 apply_rope 签名里的"B"，
       > # 与本节其它地方把 k_raw/v 当 (G,T,hs) 处理一致），补一个前导
       > # 维度让 cos/sin 变成 (1, tail, hs) 即可，dims_diff=0，直接靠
       > # 前导维 1 对 G 做标准 broadcasting，不需要额外 reshape：
       > k_tail_roped = apply_rope(k_raw[:, cutoff:T, :], cos_cache[None, cutoff:T, :], sin_cache[None, cutoff:T, :])
       > v_tail = v[:, cutoff:T, :]
       >
       > # log_kv_slot_attention(q, slot_k, slot_v, slot_w, scale, mask=None,
       > # lam=1.0, causal_tail=0, slot_valid=None, M_s=None, ...)——scale 是
       > # 位置参数、排在 slot_w 之后，slot_valid/M_s 是具名关键字参数。
       > # get_attention_state() 返回类型是 algorithm-spec.md §5.14"slot_
       > # valid/M_s 在语义模式下不是可选项"一节新增的 CacheAttentionState
       > # （具名结构，不是裸位置元组），按字段名取值，不再有"第几个位置对应
       > # 哪个参数"这个问题；with_stats 显式传 False——3b 只验证批量近似
       > # 路由/compaction 与严格串行参考的一阶读出是否一致，不覆盖 Σ/Γ 二阶
       > # 修正（后者已经由 algorithm-spec.md 同一节"Σ/Γ 锚点物化"的独立单测
       > # 覆盖，理由见那节）；lam 显式传 log_kv_lambda（本次实验实际配置的
       > # mass bias 系数，不能让它默认落到 1.0）——§7 消融表把 λ∈{0,1} 列为
       > # 一个独立扫描轴，λ=0 时 mass bias 整项不加（log_kv_cache.py:1625
       > # 的 `if lam != 0.0:` 门控），若 3b 悄悄固定用默认值 1.0，扫 λ=0 时
       > # 3b 比较的就不是这次实验实际配置的读出行为：
       > # 更正（这一轮修的，P2）：上一版把 append_exact_tokens 当成"收三个裸
       > # 张量、吐三个裸张量"的老式函数调用——但 algorithm-spec.md §5.14
       > # （CacheAttentionState 定义那一节）已经钉死它的新契约是"收一个
       > # CacheAttentionState、吐一个 CacheAttentionState"，输入输出都要迁移，
       > # 不是只迁移 get_attention_state() 一个函数。改成按新契约调用：先用
       > # NamedTuple 的 _replace() 只转 dtype（slot_valid/M_s 等其余字段原样
       > # 透传，_replace 不碰未指定的字段），再整个 state 传给
       > # append_exact_tokens，返回值也是一个完整 CacheAttentionState，直接
       > # 按字段名取给 log_kv_slot_attention，不再有裸位置元组：
       > state_batch_f32 = state_batch._replace(
       >     slot_k=state_batch.slot_k.float(),
       >     slot_v=state_batch.slot_v.float(),
       >     slot_w=state_batch.slot_w.float(),
       > )
       > state_batch_full = append_exact_tokens(state_batch_f32, k_tail_roped.float(), v_tail.float())
       >     # 尾部拼在 prefix 的 pooled+recent 之后；state_batch_full.slot_valid/
       >     # M_s 原样透传自 state_batch_f32，只覆盖 pooled 前缀宽度，不需要为
       >     # 拼接的尾部额外扩展（log_kv_slot_attention 对 S_pooled 之外的位置
       >     # 隐式按 slot_valid=True/M_s=1 处理，和 exact 后缀天然 w=1 是同一条
       >     # 既有约定）
       > out_batch = log_kv_slot_attention(
       >     q_tail, state_batch_full.slot_k, state_batch_full.slot_v,   # scale 复用机制 B
       >     state_batch_full.slot_w, scale,                                # 循环里已经在用
       >     lam=log_kv_lambda,                                             # 的同一个 scale；
       >     causal_tail=tail_query_count,                                  # lam 是本次实验
       >     slot_valid=state_batch_full.slot_valid,                        # 实际配置的值，
       >     M_s=state_batch_full.M_s,                                       # 不留给默认值 1.0
       > )
       > state_serial = cache_serial_prefix.get_attention_state(with_stats=False)
       > state_serial_f32 = state_serial._replace(
       >     slot_k=state_serial.slot_k.float(),
       >     slot_v=state_serial.slot_v.float(),
       >     slot_w=state_serial.slot_w.float(),
       > )
       > state_serial_full = append_exact_tokens(state_serial_f32, k_tail_roped.float(), v_tail.float())
       > out_serial = log_kv_slot_attention(
       >     q_tail, state_serial_full.slot_k, state_serial_full.slot_v, state_serial_full.slot_w, scale,
       >     lam=log_kv_lambda,
       >     causal_tail=tail_query_count,
       >     slot_valid=state_serial_full.slot_valid,
       >     M_s=state_serial_full.M_s,
       > )
       > error_3b = relative_l2(out_batch, out_serial)   # 3b 的分子/分母；
       >                                                    # 两次调用用的是
       >                                                    # 同一个 q_tail；
       >                                                    # relative_l2 的
       >                                                    # 精确定义见下方
       > ```
       >
       > **`relative_l2` 此前只是一个未定义的函数名，补上精确定义**——
       > 分母取谁、要不要 fp32、`eps` 怎么取、按 head/layer 先平均还是
       > 整体 Frobenius，四个问题都需要钉死，否则 <5% 这个硬性决策门
       > 本身就没有良定义的算法：
       >
       > ```python
       > def relative_l2(out_batch, out_serial, eps=1e-6):
       >     """out_batch/out_serial: (B, nh, tail_query_count, v_dim)，两次
       >     log_kv_slot_attention 调用的直接输出（activation dtype，如
       >     fp16/bf16）。
       >
       >     误差必须在 fp32 里算，不能在原 dtype 上直接相减取范数——
       >     tail_query_count·v_dim 量级的求和在 fp16 下自带可观的舍入
       >     噪声，量级可能和我们想测量的信号（批量近似 vs 严格串行的
       >     真实分歧）相当，混进去会让 <5% 的判定本身不可信。
       >
       >     分母固定取 ||out_serial||，不是对称范数、也不是取两者较大值
       >     ——out_serial 是这次比较里的参考真值（严格串行参考），3b 问
       >     的是"批量近似偏离参考多少"，不是"两者互相偏离多少"这种两个
       >     地位对等的量，与 `torch.testing.assert_close(actual,
       >     expected, ...)` 系比较固定以 expected 为参考的惯例一致。
       >     `eps` 只防止 `out_serial` 本身接近全零时除零，不改变分子
       >     分母的选择。
       >
       >     范数按 (tail_query_count, v_dim) 联合展平后取，返回形状
       >     (B, nh)——每个 (batch, query head) 一个标量，不在函数内部
       >     跨 head 或跨 layer 平均。这是刻意的：按 §2.5 的贯穿性要求，
       >     `error_3b` 依赖 `q`，属于"逐 (layer, query head)"这一档
       >     （不是聚类/路由统计的 (layer, KV group) 档），3b 的 <5%
       >     决策门同样逐 (layer, query head) 判定，不允许几个语义头很好的分数把
       >     某个头很差的分数平均掉——调用方对每一层单独调用本函数一次
       >     （层是外层循环，不在这个函数内部），拿到的 (B, nh) 结果按
       >     (layer, head) 网格汇总报告，不产出单独的整体聚合数字。
       >     """
       >     a = out_batch.float()
       >     b = out_serial.float()
       >     num = torch.linalg.norm((a - b).flatten(start_dim=-2), dim=-1)   # (B, nh)
       >     den = torch.linalg.norm(b.flatten(start_dim=-2), dim=-1)          # (B, nh)
       >     return num / den.clamp_min(eps)
       > ```
       >
       > 这也顺带回答了"dense readout（`out_dense=probs@v`、GQA 展开、
       > dtype/softcap 口径）要不要补全"这个单独提过的问题——**不需要**，
       > 因为 3b 从来不吃 dense 的最终读出结果，`attn_mass_by_dist` 需要
       > 的只是 `probs` 本身（分桶累加），机制 B 现有伪代码对它自己的
       > 目的已经是完整的，缺的从来不是 dense readout 公式，是"3b 不该
       > 向它要东西"这条边界。压缩侧两次调用**直接、原样传递生产函数
       > `get_attention_state()` 的返回值给 `log_kv_slot_attention`，不
       > 做任何手工重建或平行实现**（GQA 折叠、`slot_valid`、`M_s`、
       > `λ·log(w) − log(M)`（`−log(M)` 不受 `λ` 门控，§2.3）、fp32
       > 分数缓冲这些细节因此全部自动保持一致，不需要在这里重新枚举）。
       >
       > `tail_query_count > block_size` 时 `q_tail` 会跨越不止一块——
       > 上面"先逐块累积、循环结束后统一 `cat`"的写法对这种情形和
       > `tail_query_count ≤ block_size` 的情形处理方式完全相同，**不
       > 需要 `tail_query_count ≤ block_size` 这条约束，也不需要为 3b
       > 单独引入一个 ring buffer**——机制 B 本来就要完整跑一遍全部
       > block（服务 `attn_mass_by_dist`），跑到覆盖尾部窗口的那几块时
       > 顺路多存一份切片，循环天然会覆盖到全部需要的位置。**`block_size`
       > 不影响 3b 的结果，只影响需要几次循环迭代才能凑齐 `q_tail`**——
       > 3b 复现实验只需要记录 `tail_query_count`；`block_size` 是否要
       > 记录是 `attn_mass_by_dist`/§13.2 那条独立诊断线的复现需求，和
       > 3b 的 5% 数字无关，不要求两者一起进 metadata。
       >
       > **不选"额外持久化一份 q_roped（或它的抽样子集）供事后用"这条
       > 路**：那样会让 S0.8 依赖的输入和"Stage 0 只落盘
       > k/v/`attn_mass_by_dist` 直方图"这条既定原则产生一个例外，且
       > 抽样出来的 query 子集能不能代表真实 readout 误差是一个新的、
       > 未经验证的假设；上面的做法全程只在内存里累积一个
       > `(nh, tail_query_count, hs)` 的小切片、用完即弃，不落盘，不
       > 引入这个假设。
       >
       > **这个决定的直接推论（这一轮补的）：3b 不能是一个独立于 GPU
       > dump 的后处理脚本，必须内嵌在 dump 脚本本身里。** 上面"S0.8
       > （含 3b）本来就不是纯 CPU 后处理"那段说 `cache_batch`/
       > `cache_serial` 的构造只吃 `k_raw`/`v`/`pos`（不需要 GPU），这句
       > 话容易被读成"构造 cache 这一步可以在任何时候、任何进程里做"
       > ——但既然 `q_tail` 只在 mechanism B 的循环里短暂存活、用完即弃，
       > 3b 最后那一步真实读出比较就只能在 `q_tail` 还没被丢弃时发生，
       > 也就是**这条 prompt 的 mechanism B 循环跑完、进入下一条 prompt
       > 或脚本退出之前**。具体顺序：dump 脚本处理某一层时，mechanism A
       > 的 hook 先给出这一层完整的 `k_raw`/`v`（一次性可得的中间激活，
       > 不需要等待），紧接着 mechanism B 逐块跑过 `q_roped`（累积
       > `attn_mass_by_dist` 和 `q_tail`）；`q_tail` 一凑齐，脚本必须
       > **立即**（还在同一次 dump 调用栈里）用刚拿到的 `k_raw`/`v`/
       > `pos` 跑一遍 §5.4/§5.18 的 CPU 参考实现构造出
       > `cache_batch`/`cache_serial`，再做上面两次
       > `log_kv_slot_attention` 调用算出 `error_3b`，然后才能丢弃
       > `q_tail`、移动到下一层/下一条 prompt。**S0.8 的其余部分（第
       > 1/2/3a 项，以及不依赖 `q` 的 cache 构造本身）不受这条约束**
       > ——它们不碰 q，可以在 dump 完全结束、进程退出之后，用另一个
       > 独立的纯 CPU 脚本随时对着已经落盘的 `k_raw`/`v`/`pos` 重跑；
       > "离线"这个词对它们是准确的，但不适用于 3b 最后这一步读出比较。

  **决策门只挂在 3b 上**：注意力读出的相对 L2 误差应 < 5%（沿用原来的
  数字，但现在明确它挂在哪一项，且不再依赖 3a 那套需要精确 token 集合
  匹配、覆盖率可能不到 100% 的 ladder 字段比较）。**这个 < 5% 的结论只
  覆盖"尾部 query"这一种设定**——3b 按定义只用序列尾部窗口
  （`tail_query_count`）内的 query，不覆盖 query 在中部/前置的情形；
  下面 §7 消融表"query 位置"那一行是一个独立的、eval-time（Stage 2，
  真实模型输出）测量，回答的是同一个问题在中部/前置 query 下什么样，
  **不是 3b 的延伸，也不共享它的 5% 阈值**——3b 只代表尾部 query，不要
  拿它的结论去承担中部/前置 query 的判断，两者的结论不要互相借用。

  **多 prompt 聚合口径（此前未定义）**：`relative_l2` 对每条 prompt 各自的
  每一层给出一个 `(B, nh)` 网格，不跨 head/layer 平均（见 `relative_l2` 的
  docstring）；Stage 0 dump 的是"几条 32k NIAH prompt"（复数），所以完整的
  `error_3b` 结果是一个 `(prompt, layer, head)` 三维网格，此前没有说清楚这
  个网格怎么收敛成一个"过/不过"的判断。**决定：硬性决策门挂在
  `max` 上——`max` over `(prompt, layer, head)` 的 `error_3b` 必须
  < 5%，任意一个 prompt 的任意一层任意一个头超标就算 S0.8 在 3b 这一项
  失败，不是 warning。** 理由和"不跨 head/layer 平均"同源（§2.5 的贯穿性
  要求）：均值/`p95` 这类统计量会把"某个头系统性有问题"稀释进一堆好头里，
  而 3b 存在的意义正是暴露这类头/层特定的失效，取 `max` 是唯一不会把问题
  平均掉的选择。**但只报一个 `max` 数字不够诊断**：同时报告完整网格的
  均值、`p95`、以及取到 `max` 的那个 `(prompt, layer, head)` 三元组本身
  （方便直接定位是哪条 prompt、哪一层、哪个头出的问题，不需要事后再翻
  整个网格去找）——`max` 决定过没过，分布统计和最差点身份帮助归因"差在
  哪"，两者都要报，不能只留一个。第
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
| `K_max` | 1 / 4 / 16 / 64 | 语义分组本身值多少分？**`K_max=1` 是单簇消融参考点，不是现有 LogKV 的数值锚点**——即使实现完全正确也不预期复现现有 LogKV 的 0.1716/0.0827（`algorithm-spec.md` §5.18"K_max=1 是退化边界"一节），正确性检验走 `anchor_mode=z` 的 CPU 单测，不在这张表里。**覆盖默认值时要连带重算 `L_alloc`**（§5.6）|
| `K:B′` 分配 | 32×4 / 16×8 / 8×16 | 语义分辨率 vs 时序分辨率，总预算固定 |
| `anchor_mode` | `lo_hi_mid` / `lo_hi` / `mid` / `z` | 锚点表示 vs v2 的 z 统计量 |
| `λ_rel` | 0.5 – 2.0 | needle 隔离与簇纯度的平衡点 |
| `λ`（mass bias）| 0 / 1 | 原始 vanilla `log(w)` 那部分计数质量补偿开/关，值不值——`−log(M)` 这个锚点展开候选数校正项不受这个开关影响，两档下都无条件生效（§2.3），扫这一行不会像旧公式那样连带改变 anchor-count 校正的行为 |
| `γ`（遗忘因子）| 0 / 0.5 / 1 | centroid 门控更新值不值 |
| rank-1 Σ/Γ | 关 / 现有构造 / delta-rule 构造 | 第三档取决于 S0.7；**"现有构造"/"delta-rule 构造"两档在 Stage 2 还不能跑**——`algorithm-spec.md` §5.14"S0.8 3b 明确只走 with_stats=False"一节：3b 只验证过路由/compaction 的一阶分歧，从没验证过批量近似路由下 Σ/Γ 聚合状态本身是否也和严格串行参考一致，Stage 1 因此把 `second_order_scale` 默认锁在 0（即这一行的"关"），要跑另外两档必须先有类似 3b 的独立验证（那节称为 3c，未展开设计） |
| vanilla memory-matched | B 调大到同 entry 数 | **排除"只是多用了内存"** |

**multi-needle 行必须同时报告 `K_max` 的绑定频率**（§5.6）：needle 数逼近 `K_max` 时
Ward 会把 needle 漏斗进同一个簇，分数下降到底是聚类不行还是预算被撑爆，不报这个数
无法归因。

指标沿用现有四项（ACC/LongBench/LongBench_e/niah@32768），**另加 multi-needle**——
单 needle 一旦从 0.08 提上去就会迅速失去区分度。v3 的锚点表示对 multi-needle 应有
额外优势（多个 needle 各自成簇、各自保留精确位置）。
