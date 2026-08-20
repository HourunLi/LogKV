# SemanticLogKV 算法规格与实现方案

> 本文件是 [`../CLAUDE.md`](../CLAUDE.md) 的第 §5 章，独立成文只为便于阅读和维护。
> **章节编号沿用全局编号**，所以文档间的交叉引用（`§2.3`、`§11-A`、`§5.19-2` 等）
> 全部有效。设计动机和为什么这样选，看 CLAUDE.md 的 §2；实验协议看
> [`experiments.md`](experiments.md)；风险与未决问题看
> [`risks-and-open-questions.md`](risks-and-open-questions.md)。
>
> **动手写代码之前先读 §5.19（实现 tips 与易错点）和 §5.20（复用边界），
> 以及 risks 文档的 §11（技术难点清单）。**
>
> **本文件目前 100% 是设计规格，不是已实现代码——`litgpt/`、`tests/` 里不存在
> 下面参数表、buffer 清单、`op_log` 等任何字段或分支（`LogKVStreamTrainingAttention`
> 这个类名确实存在，但那是它在语义簇设计之前、位置分桶时代就有的版本，见
> CLAUDE.md §0"当前阶段"）。文中大量"更正""这一轮修的"字样，改的都是**规格
> 文本自身的逻辑漏洞**，不是已运行代码的 bug——不要把这些当成"代码已经在跑，
> 只是在修 bug"的证据。连 Stage 0 的 dump 脚本（§5.21-5，本项目该写的第一段
> 代码）都还没有写。**

## 5. 算法规格与实现方案

### 5.1 记号与参数

> **下表是设计参数，不是已存在的 CLI 开关**——`log_kv_semantic_clusters` 等
> 字段在当前代码里尚不存在，见本文件顶部的"未实现"说明。

> **记号冲突警告**：早期版本用 `λ` 同时表示 mass bias 系数和 segment 阈值，用 `τ`
> 表示 cosine 阈值。现已统一如下，`λ` **只**保留给现有代码已占用的 mass bias。

| 符号 | 参数名 | 默认 | 含义 |
|---|---|---|---|
| — | `log_kv_semantic_clusters` | `false` | 总开关，关闭时逐字节复现改动前行为 |
| `K_max` | `log_kv_cluster_k_max` | **`max(4, ⌈log₂N⌉)`** | 簇数上界。**默认由 `N` 推导，但允许实验显式覆盖**（消融就是在扫它，见 §5.6）|
| `B′` | `log_kv_cluster_entries` | `8` | ladder 每级 entry 数；同时是**无损阈值**（§5.9）|
| `L_alloc` | 无 CLI 开关 | 见 §5.12 | 每簇 ladder 层数。**纯推导量**，且必须用**生效的** `K_max` 算，不能用默认值 |
| `λ_rel` | `log_kv_cluster_lambda_rel` | `1.0` | 新簇距离阈值（相对量，见 §5.2）|
| `η` | `log_kv_seg_eta` | `1.0` | join cost 时序权重，**只影响候选排序**（§5.3）|
| `g0` | `log_kv_seg_g0` | `2048` | 时序项的饱和尺度 |
| `g_max` | `log_kv_seg_gap_max` | `4096` | 开新 segment 的间隔阈值（**S0.0 主扫描之一**）|
| `ℓ_block` | `log_kv_seg_block_level` | `1` | 段对齐保护到第几层（**S0.0 主扫描之一**，§5.11）|
| `γ` | `log_kv_seg_forget` | `0.5` | 跨段的 centroid 计数遗忘因子（§5.5）|
| `λ` | `log_kv_lambda`（现有）| `1.0` | **mass bias 系数**，沿用现有语义 |

沿用项目惯例：`demo.py`/`eval.py` 走 `run_cli()` 签名自省与 `_o()`，**YAML 新字段
默认必须写 `null`**（否则会被同名 CLI 参数静默覆盖，见 §12），不进 `eval.sh`/
`majob.sh` 核心列表，走 `DIAG_ARGS` opt-in。

**`log_kv_semantic_clusters=True` 必须拒绝 `importance_pooling=True` 与
`pin_size>0`，在构造时硬失败，不是静默忽略**：

```python
if log_kv_semantic_clusters and (importance_pooling or pin_size > 0):
    raise ValueError(
        "semantic clusters 与 importance_pooling/pin 尚未定义共存语义，"
        "构造时二选一"
    )
```

理由：`compact()` 现有的 `imp1`/`imp2` 路径（`log_kv_cache.py:238-246`、
`695-706`）会让槽内加权均值偏离纯 `w`-计数均值，改变的正是 Ward 合并代价、
DP-means 分配阈值、mass bias `log(w/M)` 全部依赖的那个"`w` 就是真实权重"的前提。
两套机制不是不能共存，是共存的数学还没推导——**pin 系列本身也是另一条独立技术
路线（CLAUDE.md 顶部已声明），在没有专门推导之前默认禁止组合，比默认允许后产出
无法解释的分数更安全。**

**同样，`log_kv_semantic_clusters=True` 必须拒绝 `rope_interleave=True`**：

```python
if log_kv_semantic_clusters and config.rope_interleave:
    raise ValueError(
        "semantic clusters 的 _rotate_at_anchors 目前只实现了 split-half 布局的"
        "RoPE（对应 apply_rope），未覆盖 rope_interleave=True 用的 "
        "apply_rope_interleave 布局；两者数据排布不同（偶奇分组 vs 前后半），"
        "混用会静默产生错误的旋转"
    )
```

理由：§5.14 的 `_rotate_at_anchors` 复刻的是 `apply_rope`（`model.py:2074` 附近）的
split-half 数学——`rot = cat(-x[h:], x[:h])`。但 `model.py:810-812` 显示模型层还有
一条 `config.rope_interleave=True` 时启用的 `apply_rope_interleave` 路径，先把
`[x0,x1,x2,x3,...]` 重排成 `[x0,x2,x4,...,x1,x3,x5,...]`（偶奇分组）再做旋转
（`model.py:2106-2128`），是**完全不同的数据排布**，不是同一个函数换个参数。
Qwen3-1.7B（本项目的目标模型）的 `rope_interleave` 默认 `False`，走 `apply_rope`，
`_rotate_at_anchors` 的实现覆盖了这个模型；但仓库里其它配置确实会打开这个开关
（如 `tests/test_yarn.py`），如果不做这条校验，换一个用 interleaved RoPE 的模型
跑 semantic clusters，锚点旋转会静默算错，且不会有任何报错——和 `ℓ_block`、
`importance_pooling`/`pin` 是同一类"宁可拒绝、不可算错"的风险，处理方式对齐：v1
硬性拒绝，覆盖 interleaved 布局留作后续泛化（把 `_rotate_at_anchors` 拆成两个
分支，复用 `apply_rope_interleave` 同款的重排逻辑），不在当前范围内。

**同类的第三条：`log_kv_semantic_clusters=True` 只支持 `CausalSelfAttention`，
遇到 `MultiheadLatentAttention` 必须硬失败。** `litgpt/model.py` 里这两个是完全
独立的类（`656` 起 / `1579` 起）——本节从 §5.2 的度量到 §5.14 的锚点物化，通篇
假设的都是 `CausalSelfAttention` 的显式 K/V-per-group 布局（clustering 直接作用于
`k_raw: (B,G,T,hs)`）。MLA 的 KV 是低秩 latent（`kv_lora_rank`）+ 独立的
`qk_rope_head_dim`/`qk_nope_head_dim` 拆分，压根没有一份"每 KV 头一条"的原始
`k_raw` 可供聚类——把本节的设计直接套上去在概念上就不成立，不是"实现起来麻烦"，
是**没有对应的输入**。构造时校验 `isinstance(attn_module, CausalSelfAttention)`，
否则 `raise NotImplementedError`；MLA 需要的话是一次独立的设计，不是本方案的
参数扩展。

### 5.2 分簇：度量与阈值

**用 qk-norm 之后、RoPE 之前的 key 上的平方欧氏距离，不做 cosine/L2 归一化。**
理由不是习惯，而是这一个量同时是三件事：

> **更正（这一轮修的）：上一版"原始（未归一化）pre-RoPE key"这个措辞会被读成
> "qkv 投影的原始输出，qk-norm 之前"，与 §5.21-1 的定案直接矛盾。** "未归一化"
> 这里唯一想表达的是"不做 cosine 需要的 L2/单位范数归一化"（呼应"不用 cosine"），
> 不是"跳过 qk-norm"；但"原始"和"未归一化"连用，读者很自然会把它理解成"没有
> 经过任何处理的 key"——而 qk-norm 本身就是一种归一化，"未归一化"这个词精确地
> 诱导了这个误读。§5.21-1 已经用一整节钉死"pre-RoPE k 精确地说是 qk-norm 之后、
> apply_rope 之前，不是 qkv 投影的原始输出，取错位置会让簇的度量落在未归一化的
> 空间里、`s_h` 标定直接失效"——本节和 §5.21-1 描述的是同一个量，只是这里的
> 措辞不小心写成了 §5.21-1 明确警告过的那个错误读法。改成"qk-norm 之后、RoPE
> 之前"，不再用"原始（未归一化）"这个会引发歧义的说法。

- 槽内池化误差对分数的影响是 `q·(k_i − k̄)`，所以**控制误差的量就是欧氏距离**；
- DP-means 的分配准则 `‖k − μ_c‖² > λ_new`；
- Ward 合并代价 `(n_a n_b)/(n_a+n_b)·‖μ_a − μ_b‖²`，即簇内平方和的增量，也正是现有
  `_pair_rank1_stats` 已经在算的量。

**一个度量贯穿路由、合并、误差分析三处。** 副作用：attention sink（巨范数 token）
因为距离所有 centroid 都远而自动隔离——正是想要的行为（对比重要性池化是把 sink
加权**放大**，方向相反）。

**阈值必须相对化，否则跨层不可比**：`k` 的范数在不同层、不同 KV group 之间差好几个
数量级。

```
λ_new = λ_rel · s_h        s_h = 该 (layer, KV group) 上 E‖k − k̄‖² 的估计
```

> **口径统一：这里的粒度是 KV group（`G=8`），不是 query head（`nh=16`）**。聚类
> 只在 k 空间做（§2.5），而一个 KV group 只有一份 `k`——GQA 下 `q_per_kv=2` 个
> query head 共享同一份 k/v（`model.py:857-860`），所以"每个头一个 `s_h`"这个
> 说法本身就有歧义：在 query-head 粒度上，两个共享同一 KV group 的 query head
> 天生就该有相同的 `s_h`（它们看到的是同一份 `k`），单独给它们估计两份值既浪费
> 又制造出两个理应恒等却可能因为标定噪声而不等的常量。**`s_h` 的物理粒度是
> KV group**，后文所有"per-head"措辞统一改读作"per-KV-group"，buffer 形状
> `(n_layer, G)`（§5.13）已经是对的，是文字表述滞后了。

**`k̄` 的定义**：`k̄` 是该 (layer, KV group) 在**整个标定集**上的全局均值——不是
按 prompt 分别求均值再平均，也不是逐层逐组维护的运行均值。具体地，标定 pass 对
每个 (layer, KV group) 累积**标定集里所有 prompt、所有位置**的 `k_raw`，先算出
一个全局 `k̄`，再算 `s_h = mean_t ‖k_t − k̄‖²`（对全体标定 token 取平均，一次
过、离线、非流式）。理由：`s_h` 要刻画的是"这个 (layer, KV group) 的 key
分布本身有多分散"，是一个**总体统计量**，不是"某条 prompt 内部有多分散"（后者
会被单条 prompt 的话题数系统性左右，不是我们想要的、跨 prompt 稳定的尺度参考）；
也不是运行均值（那是在线估计的做法，已经在下面被否决，理由见 §5.21-4）。

**`s_h` 已定案：v1 用离线标定**（完整取舍见 §5.21-4）。在留出样本上跑一次前向，
记录每 (layer, KV group) 的 `E‖k − k̄‖²`，存成常量随 config 走，**并写进 eval
metadata**。在线估计会让阈值随序列演化、引入 §11-E 的新变体，且序列开头估计未
收敛——而 §5.6 说早期路由错误不可恢复，所以它降级为后续消融项。**这一条如果不做，
S0.2 扫出来的 `K_eff` 曲线在层间完全没有可比性**，整个 Stage 0 会得出无意义的
结论。

### 5.3 分簇：三路判定

有两个**性质不同**的决策——"这内容新不新"（花一个 centroid 槽）和"时序断没断"
（花 ladder 局部性）。硬揉成一个阈值就是重犯 v3 的错。正确结构是：
**统一代价用于挑候选，两个阈值用于两个决策。**

```
对每个 flush 出来的 token x（绝对位置 p）:

  d_c = ‖k_x − μ_c‖²  +  η · φ(p − p_hi_c)        # 统一代价，仅用于排序
        φ(g) = g / (g + g0)                        # 有界饱和：时序项不能无限支配
  c* = argmin_c d_c

  if   ‖k_x − μ_{c*}‖² > λ_new :   → 开新 cluster              # 语义 novelty
  elif p − p_hi_{c*}   > g_max  :   → 同 cluster，开新 segment   # 时序被打断
  else:                              → 同 cluster，当前 segment
```

`η` 在这里的作用是**平局裁决**：两个簇语义上差不多近时，优先给最近活跃的那个。
`φ` 必须有界饱和，否则很老的簇会被赋予近乎无穷的惩罚，等价于硬性禁止回访，那就退回
v3 的 fork 行为了。

> **重要更正（曾经写错）**：早期版本说"`η → ∞` 退化成纯连续分段"，并据此把扫 `η`
> 定为 S0.0。**这是错的**——`η` 只出现在排序代价里，`η → ∞` 的实际效果是"总是先看
> 最近活跃的簇"，而它多半语义上不匹配，于是几乎每个 token 都开新簇，是一种退化行为，
> 既不是纯语义也不是纯分段。真正在两端之间插值的是 **`(g_max, ℓ_block)`**：
> `g_max → ∞` 或 `ℓ_block = 0` 是纯语义聚类；`g_max` 小且 `ℓ_block` 大是连续分段。
> S0.0 已相应改为扫这一对。
>
> **第二处需要收紧：即使 `g_max → ∞` 或 `ℓ_block = 0`，只要 `η > 0`，"纯语义聚类"
> 这个说法仍然不精确。** `(g_max, ℓ_block)` 控制的是**要不要分段**，但 `c*` 本身
> 由统一代价 `d_c = ‖k_x−μ_c‖² + η·φ(p−p_hi_c)` 的 argmin 决定——只要 `η>0`，
> `c*` 就可能不是纯语义最近的那个簇（时序 tie-break 把它换成了另一个），而
> `novelty` 判据 `‖k_x−μ_{c*}‖² > λ_new` 恰恰是在 `c*`（不是纯语义最近簇）上
> 判定的。也就是说 `g_max→∞`/`ℓ_block=0` 只保证"不因为时序打断而开 segment"，
> **不保证 novelty 判定不受时序影响**——`η` 仍然可能让一个 token 被判定为"该开
> 新簇"或"该并入某簇"，而这个判定和纯语义距离矩阵的 argmin 不一致。
>
> **两种处理方式，选第一种**：①把"纯语义聚类"端点精确定义为
> `η=0 且 (g_max=∞ 或 ℓ_block=0)`——`η=0` 时 `d_c` 退化成纯语义距离
> `‖k_x−μ_c‖²`，`c*` 就是真正的语义最近簇，此时"纯语义聚类"这个描述才是字面
> 精确的；S0.0 扫 `(g_max, ℓ_block)` 时应固定 `η=0`，而不是沿用生产默认值——
> `η` 的作用（打破近似语义平局）是生产路由要的性质，不是"分离语义贡献与分段
> 贡献"这个受控实验要的性质，混进去会让 S0.0 的"纯语义"端点带有一个未受控的
> 时序污染源。②退一步只把"纯语义"重新定义成"不分段"（不做上述收紧，接受
> `η` 的残余影响）——**不采用**，因为 S0.0 存在的意义就是要干净地回答"收益
> 来自语义分组本身，还是仅仅来自更好的分段边界"（CLAUDE.md §0），一个端点里
> 混入未受控的时序因素会让这个问题本身变得不适定。

### 5.4 分簇：批量化路由（三阶段）

朴素实现要逐 token 串行（centroid 边分配边更新），代价是单次前向 45 万次 Python
迭代，训练完全不可行（§11-B）。解法基于一个频率观察：**"加入已有簇"是常态，
"开新簇"是罕见事件**。

```
Phase 1（并行，覆盖绝大多数 token）:
    冻结 centroid 与 p_hi_c，一次算出统一代价矩阵
        D[t,c] = ‖k_t − μ_c‖² + η·φ(p_t − p_hi_c)      # (m, K)，就是 §5.3 的 d_c
        S[t,c] = ‖k_t − μ_c‖²                            # (m, K)，纯语义距离，同一次算出
    c*[t]   = argmin_c D[t,c]                              # 每个 token 的统一代价赢家
    s*[t]   = gather(S[t,:], c*[t])                        # c* 自己的纯语义距离
    所有 s*[t] ≤ λ_new 的 token 直接分配到 c*[t]（是否新开 segment 仍按 §5.3 的
    第二个判据 p_t − p_hi_{c*[t]} > g_max 决定）

Phase 3a（紧接 Phase 1 之后、Phase 2 开始前，向量化，**不是**批末）:
    用 Phase 1 刚产出的主操作，批量更新它们各自目标簇的
    centroid/n_eff/n_total/p_hi_c/current_segment（§5.5）——见下方"Phase 3
    拆成 3a/3b"一节，这一步必须在这里执行，不能推迟到 Phase 2 之后

Phase 2（串行，只处理 orphan）:
    s*[t] > λ_new 的 token 需要开新簇，它们之间还可能互相成簇
    （s*[t]/c*[t] 沿用 Phase 1 算出的值，不因 Phase 3a 随后更新了 centroid
    就重新判断——见下方"orphan 集合冻结"说明）
    在这批 orphan 内部跑一个小 DP-means（O(m_orphan²) 的距离矩阵即可）
    每个 orphan（组）的主操作一旦写入本地缓冲，立即（Phase 3b，内联，不等
    批末）用同一套 §5.5 公式更新它目标槽的元数据——Ward 合并、centroid 混合
    因此全程读到的都是本批目前为止的真实状态，见下方"Phase 3 拆成 3a/3b"
    一节
```

> **orphan 集合冻结：Phase 3a 更新的 metadata 不反过来影响它，这是一个
> 明确决定，不是留白。** Phase 1 算出的 `c*[t]`/`s*[t]`（进而
> `direct[τ]`/orphan 划分）只在 Phase 1 这一步计算一次，此后原样沿用到
> 这一批处理完毕——即使 Phase 3a 紧接着把某个簇的 `centroid`/`n_eff`
> 更新了（可能让某个原本 `s*[t] > λ_new` 的 orphan，相对更新后的 centroid
> 已经不再超过阈值），Phase 2 也**不会**拿这份更新后的 metadata 去重新
> 判断该 token 是不是还该算 orphan。Phase 3a 写出的 metadata 只服务两处
> 消费者：Ward 代价矩阵（`Δ[a,b]` 需要**当前**的 `n_total`/`μ`，包含本批
> Phase 1 已经贡献的部分，见 §5.6）和 Phase 3b 自己的 §5.5 在线更新
> （需要**当前**的 `n_eff`/`μ` 作为"批前"状态）——从不用于重新检验 Phase 1
> 已经做出的 direct/orphan 分类。
>
> **为什么必须冻结，不能重判**：如果允许重判，一个被重新判定为"该并入某
> 既有簇 X"的 orphan（X 恰好是本批 Phase 1 也写过 direct token 的簇），会
> 在本地缓冲里给 X 产生一条 `JOIN`——但物理写入顺序是"Phase 1 整组先写、
> Phase 2 整组后写"（见下方"为什么……满足这三条契约"一节），这条本该按
> 真实到达顺序排在 X 某些 Phase 1 成员之间的 `JOIN`，会被无条件排到 X 在
> 本地缓冲里全部 Phase 1 成员的**后面**，直接违反契约第 1 条（"同一逻辑
> 簇自己的主操作序列必须按真实到达顺序出现"，`compact()` 的"时间序相邻
> 配对"正确性依赖这一条）。这不是这条契约偶尔失效的一个例外，是它赖以
> 成立的前提被直接打破——契约成立依赖"同一逻辑簇的主操作在一个 flush
> 批内只可能整体来自 Phase 1 或整体来自 Phase 2"（下方论证），允许重判
> 会让这个前提不再自动成立，下方那整节证明也随之失效。
>
> **这不是一个新的近似类别，是"Phase 1 冻结 centroid"这个已经接受、已经
> 在 S0.8 测的近似（本节开头"这是一个近似"那段）的直接延伸**——批前冻结
> 意味着 Phase 1 的分类从一开始就没有用到"本批最新"的 centroid，Phase 3a
> 的更新只是让这份"过时"在批内更明显了一点，不改变近似的性质。S0.8 的
> cluster assignment divergence 指标（`experiments.md` §6）用的是"批量
> 路径 vs 严格串行、每个 token 都用当时最新 centroid 重算一次"的对拍，
> 天然覆盖这类分歧，不需要为它单独定义新的度量。

> **更正（曾经写错）**：早期版本的 Phase 1 判据是"`min_c ‖k−μ_c‖² ≤ λ_new` 的
> token 直接按 §5.3 分配"——**这不等价于 §5.3**。§5.3 的 `c*` 由统一代价 `d_c`
> （语义 + 时序 tie-break）的 argmin 决定，不是纯语义距离最小的那个簇；如果纯语义
> 最近的簇 X 因为太久没被访问、时序项把它排到了后面，而统一代价选中的是簇 Y，
> 旧判据会把这个 token 当作"安全批量分配"处理，但分配去向和判定阈值用的却是两个
> 不同的簇（判"要不要开新簇"用 X 的语义距离，`c*` 却该是 Y）。**这不是近似误差，
> 是判据本身选错了对象**。修法是先按统一代价选出 `c*`，再取 `c*` 自己的语义距离
> 去比较 `λ_new`——和上面重写的伪代码一致，计算量不变（`S` 和 `D` 本就是同一次
> einsum 的副产品，多一次 `gather` 可忽略）。

**摊还论证**：Phase 2 的总执行次数被"新簇事件总数"界住，即 `E[K]`，**不是 `O(T)`**。
所以串行部分的总代价是 `O(K)` 而非 `O(T)`，flush 粒度因此可以从 2 提到 128。

**这是一个近似**（批内冻结 centroid**和** `p_hi_c`）。**必须测它与严格串行版的
分歧率**——S0.8（拆成 cluster assignment/Ward 事件、segment+PAD 开销、最终
cache/readout 误差三项分别测，指标定义见 `experiments.md` §6 的 S0.8 决策门，
不是一个单一标量）。`p_hi_c` 冻结这条单独更值得关注：见 §5.21-2 里对它的
展开分析（同批内连续同簇 token 会因为看到"批前"的 `p_hi_c` 而被误判成隔了
很久）。

#### op_log 跨 Phase 的顺序契约：token 归属改成显式字段，不再依赖隐式位置

**§5.21-2 已经承认"op_log 里的线性 token 顺序是一个逻辑顺序，不是 forward 实际
执行的物理顺序"，但 backward 重放的 `token_ptr` 却假设两者是同一个顺序——这是
一个真实的矛盾，不是文字表述问题，必须先在这里解决，下面 Phase 1 的具体公式才
有稳固的地基。** 具体反例：一个 flush 批是 `[direct τ0, orphan τ1, direct τ2]`
（`τ0`/`τ2` 被 Phase 1 判定为 direct，直接路由；`τ1` 是 orphan，交给 Phase 2）。
按现有设计，Phase 1（向量化）先把它处理的 direct token 的 ops 写进本地缓冲
（`τ0`、`τ2`），Phase 2（串行）随后处理 orphan、把 `τ1` 的 op 追加在后面——本地
缓冲里三条主 op 的物理顺序是 `[τ0 的 op, τ2 的 op, τ1 的 op]`，**不是**
`[τ0, τ1, τ2]` 这个真实到达顺序。但重放循环的 `token_ptr` 只是一个从 0 开始的
递增计数器，它会把这三条 op 依次对应到 `k_raw[0], k_raw[1], k_raw[2]`——也就是
`τ0, τ1, τ2` 的原始数据——第二条 op（实际是 `τ2` 的 `JOIN`）会被错误地喂上
`τ1` 的 `(k_raw, v, pos)`，第三条 op（实际是 `τ1` 的 `NEW_CLUSTER`）会被错误地
喂上 `τ2` 的数据。**这不是边界情形，只要一个批里 direct 和 orphan 交替出现就会
发生。**

**修法：主操作的 `arg2` 字段改存显式 `token_idx`（该 token 在这次 forward 处理
的整条序列里的绝对位置），重放直接用 `op.token_idx` 索引 `k_raw`/`v`/`pos`，
不再依赖任何隐式递增的 `token_ptr`。** `arg2` 对 `NEW_CLUSTER`/`NEW_SEGMENT`/
`JOIN` 这三类主操作此前恒为 `-1`（未使用），现在改存 `token_idx`——不需要扩大
op 的 4-int32 格式。结构操作（`WARD_MERGE`/`PAD_INSERT`/`CARRY`）不消费具体
token，`arg2` 保留原有语义（`count`/`resulting_count`/`-1`），不受影响，也
不需要 `token_idx`。

**这个修法同时回答了"op_log 的物理顺序到底要不要等于真实到达顺序"这个问题
——不需要，只要每条主操作显式携带自己的 `token_idx`。** 精确的顺序契约改写为
三条，都不要求跨 Phase 的全局到达顺序：

1. **同一个逻辑簇（`(slot, epoch)` 身份）自己的主操作序列，必须按真实到达
   顺序出现**——这是 `compact()`"时间序相邻配对"数学成立的前提（§5.21-2 已经
   论证过），Phase 1 的 `main_idx` 公式（见下方）和 Phase 2 的串行处理各自
   保证了这一点。
2. **结构操作紧邻它服务的主操作**（`WARD_MERGE` 在其 `NEW_CLUSTER` 之前，
   `PAD_INSERT` 在其 `NEW_SEGMENT` 之前）——已有规则，不变。
3. **不同逻辑簇之间的主操作，互相的相对顺序不重要**——因为 Phase 3a/3b 逐簇
   独立处理元数据、`compact()`/`_binary_carry()` 的数学也是逐簇独立的，两个
   不同簇的 op 谁先谁后不影响任何计算结果。

**为什么"Phase 1（direct）全部先写，Phase 2（orphan）再写"满足这三条契约，
不需要让两组在本地缓冲里按真实到达顺序交错**：关键前提是**同一个逻辑簇的
主操作，在一个 flush 批内只可能全部来自 Phase 1 或全部来自 Phase 2，不会
两边都有**：

- Phase 1 只处理 `direct[τ] = True`（`s*[τ] ≤ λ_new`）的 token，把它们路由到
  **既有**簇（批前就 alive 的槽）。**这个划分只在 Phase 1 算一次，Phase 3a
  随后更新 centroid 不会让它反悔**（上方"orphan 集合冻结"说明）——下一条
  "orphan 永远不会 JOIN 一个 Phase 1 本批也在写的既有簇"正是建立在这个
  划分不会被事后改写这个前提上，如果允许重判，这条就不再成立。
- Phase 2 只处理 orphan（`s*[τ] > λ_new`），它们要么加入本批 Phase 2 内部
  mini DP-means 分到的**新**临时簇（一个从未在这个 `(slot,epoch)` 身份下出现
  过的全新逻辑簇），要么（在 Ward 合并腾位的情形下）落进一个**被释放又复用**
  的槽——复用之后是一个新的 `epoch`，逻辑上同样是全新身份，和被合并走的旧
  身份不是同一个簇（`scan_op_log`/`resolve_final_slots` 那节已经用
  `(slot,epoch)` 讲过这一点）。orphan **永远不会** JOIN 一个 Phase 1 本批也
  在写的既有簇——按定义，orphan 到所有既有簇（含被 Phase 1 命中的那些）的语义距离都
  `> λ_new`，否则它就不是 orphan。
- Ward 合并本身（`ward_merge_only` 步骤 1）只**读写 `keep_slot` 的聚合元数据**
  （μ/n_eff/n_total/p_hi/current_segment 的合并），**不会给 `keep_slot` 产生
  新的主操作**——orphan 自己的主操作全部落在 `free_slot`（腾出来的槽）上，不
  是 `keep_slot`。所以即使 `keep_slot` 恰好是本批 Phase 1 也写过 direct token
  的既有簇，Ward 合并也不会在它的主操作序列里插入任何 Phase 2 的东西，
  `keep_slot` 自己那条主操作序列依旧完整地、只由 Phase 1 贡献。

  所以：**每个逻辑簇的主操作序列要么整个来自 Phase 1（保序由下面的
  `main_idx` 保证），要么整个来自 Phase 2（保序由串行处理本身保证）**，
  两组之间没有交叉。"先写完 Phase 1 那一组、再写 Phase 2 那一组"这个物理
  顺序，天然满足契约第 1 条——不需要让两组在缓冲区里按 τ 交错。

**这不只是一个记录格式的决定，它同时钉死了 forward 的真实执行顺序，回答了
"Ward 合并的物理状态和重放顺序如何闭合"这个问题**：既然 op_log 的物理顺序
就是"Phase 1 组、然后 Phase 2 组"，为了让"重放严格按 op_log 顺序执行"与
"forward 实际发生的事情"逐位一致（§5.21-2 的既有要求），**forward 自己也
必须按这个顺序真的去做**——即：Phase 1 的向量化路由**和**它对应的向量化
ladder 物理写入（§5.21-3 的掩码并行 carry）在这一步内一起完成，产出一个已经
完整反映全部 direct token 内容的 ladder；Phase 2 才开始运行，它看到的、Ward
合并要读要写的就是这个已经被 Phase 1 更新过的真实 ladder，不是批前快照。
**"Phase 1 只记日志、不动 ladder，等 Phase 2 跑完再统一物理写入"和"Phase 1
先把全部 direct token 物理写入、Ward 合并提前看到本该更晚到达的 token"这两种
读法都不对**——正确的分界线不是"日志 vs 物理写入"，是"Phase 1 整体 vs
Phase 2 整体"：Phase 1（日志 + 物理写入一起）作为一个原子步骤完整跑完，
Phase 2（日志 + 物理写入一起，含 Ward 合并）才开始，两者之间没有交错，也
没有谁"抢跑"谁。这个顺序和 op_log 的物理顺序完全对应，所以 replay（严格按
op_log 顺序执行）自动等价于 forward 的真实执行——不需要另外证明，这是构造
出来的，不是巧合。

#### Phase 1 缺持久 segment 状态，且批内同簇的 segment id / PAD_INSERT count 不能各算各的

**这是比"`p_hi_c` 冻结"更基础的一处空白：`JOIN(cluster, segment)` 和
`NEW_SEGMENT(cluster, new_seg)` 都要求写一个具体的 `segment` 整数，但 §5.13 的
buffer 表里从来没有过一个持久存着"这个簇当前 segment id 是多少"的 buffer。**
`p_hi_c` 之前为 Phase 2 的 orphan 组新增了 `local_p_hi`/`local_segment` 这套私有
状态（见上面 Phase 2 那一节），但那套状态只覆盖 Phase 2**新建**的簇——Phase 1
批量分配到**既有**簇的 token 完全不在它的覆盖范围内，而 Phase 1 恰恰是每批处理
token 最多的路径。缺了这块状态，实现者会不知道 `JOIN` 该写哪个 segment id，也
无法在同一簇本批内触发多个 `NEW_SEGMENT` 时给它们分配互不冲突的递增 id。

**新增持久 buffer `current_segment: (B,G,K_max)` int32**——该簇当前最新的
segment id（下一次开新段时用 `current_segment + 1`），进 §5.13 的簇级元数据表，
和 `p_hi_c` 同一档：**Phase 3a/3b 是它唯一的写入点**（见下方"Phase 3 拆成
3a/3b"一节——walk 到一条某簇的 `NEW_SEGMENT` 时，就把该簇的 `current_segment`
更新成这条 op 的 `segment` 字段——因为 Phase 3a 对 Phase 1 的主操作做的是按簇
分组、保序的向量化扫描，Phase 3b 对 Phase 2 的主操作是逐条内联执行，两者各自
内部都保持真实到达顺序，这个赋值天然收敛到"目前为止"最后一次 `NEW_SEGMENT`
的值，不需要额外的 `max`），`ward_merge_only` 步骤 1 合并两个既有
簇时取 `current_segment_new = max(current_segment_a, current_segment_b)`（和
`p_hi_new = max(p_hi_a, p_hi_b)` 同一个模式）。

> **`current_segment_new = max(...)` 保证的是什么、不保证什么，必须说清楚，
> 不能只说"不会撞车"——那句话不准确。** `max` 只保证**合并之后、`keep_slot`
> 这个身份未来还会产生的新 segment id**，严格大于合并前 `a`/`b` 双方各自
> 出现过的任何旧 id——这防的是"未来新 id 撞上过去旧 id"。它**不**保证、也
> **不需要**保证"合并前 `a` 和 `b` 两段独立历史里的旧 segment id 互不相同"
> ——`a` 和 `b` 在合并前都是从 0 开始独立计数的（`NEW_CLUSTER` 的 `arg1=0`
> 约定），`a` 的 segment 0 和 `b` 的 segment 0 是两个完全不同的时间段，数值
> 相同纯属巧合，这从一开始就不是、也不需要是同一个命名空间。**`op_log` 里
> 这两段历史的 op 各自带着自己原本的 `cluster` 字段（`a`/`b` 各自的槽号，
> 没有被重定向改写，见前面"op_log 永不改写"一节），所以它们天然是可区分
> 的——真正需要小心的场景是"同一个物理槽先后被两条不同的逻辑簇使用"（先是
> 簇 X 用槽 5，X 被合并走或整体让位后，槽 5 又被拿去 `NEW_CLUSTER` 建立
> 一个和 X 无关的新簇 Y），这种情况下 X 和 Y 的 segment 计数都从 0 开始，
> `(cluster=5, segment=0)` 这个裸 key 确实会撞——但这正是"槽位复用"问题
> 本身（`scan_op_log`/`resolve_final_slots` 那节已经用 `(slot, epoch)`
> 版本化身份解决过一次），不是 segment 机制独有的新问题。**结论：任何需要
> 跨这类边界做分组/画图的分析工具，必须按 `(slot, epoch, segment)`（复用
> `scan_op_log` 已经维护的同一套 `epoch`）做 key，不能只用裸
> `(cluster, segment)`；而跨 `a`/`b` 两段历史的真实时间先后顺序，只能从
> `op_log` 本身的线性位置读，不能靠比较两个 lineage 各自的 `segment` 数值
> 大小去推断——它们不是同一个可比较的计数序列。**

**但只有持久 buffer 还不够——同一批内，同一簇的多个 token 各自决定"是否开新段"
时，必须知道彼此的决定，否则会产生和 Phase 2 那个 bug 同构的错误：** 若 Phase 1
用同一份批前冻结的 `p_hi_c` 独立判断每个 token，这个判断本身已经是接受下来的
近似（上面"这是一个近似"那段）；但**分配 segment id 和计算 `PAD_INSERT` 的
`count` 不能重复这个近似**——如果本批里 token A、B（同簇，A 更早到达）都被判定
为"开新段"，若两者都直接读批前持久的 `current_segment`/`level_count[cluster,0]`
计算，会给出完全相同的 `segment id` 和 `PAD_INSERT count`，而正确结果应该是 B
在 A 已经开的那个新段之后再开一段、B 的 padding 也应该把 A 插入的 pad 和 A 自己
的 entry 算进去。这本质上和 Phase 2 orphan 组内后续成员的问题是**同一类 bug**，
只是发生在 Phase 1 的向量化路径上，必须用向量化的方式解决，不能退回逐 token
串行（那样就违背了 Phase 1 存在的全部意义，见本节开头"45 万次 Python 迭代"）。

**解法：两个都是标准的向量化分段扫描（segmented scan）原语，不需要任何数据
相关的有界循环轮数——这一点和 §5.21-3 的 ladder carry 不同，值得说明为什么。**
**先定死一个此前没写清楚的过滤步骤：下面所有公式只跑在 direct 子序列上，
orphan 完全不参与。** 本批 `m` 个 token 按到达顺序排好，绝对批内位置记为
`τ = 0..m-1`；Phase 1 开头的统一代价矩阵（本节最上方的伪代码）已经给出
`c*[τ]`（每个 token 的目标簇）和 `s*[τ]`（`c*[τ]` 自己的语义距离），对**全部**
`m` 个 token 都算了一遍（这一步没法只对 direct 算，因为还不知道谁是 direct）。
但从这里开始：

```
direct[τ]  = (s*[τ] ≤ λ_new)                         # τ = 0..m-1，覆盖全批
direct_idx = 把 {τ : direct[τ]=True} 按 τ 递增排好的列表，长度 m_direct
```

下面 `rank`/`last_new_rank_before`/`steps_since`/`main_idx` 等全部公式里的
`t`，指的是**压缩后 direct 子序列的下标**（`t = 0..m_direct-1`），不是原始批内
位置 `τ`——读取 `c*[t]`/`new_seg[t]` 实际上是 `c*[direct_idx[t]]`/
`new_seg[direct_idx[t]]`（gather），`new_seg[τ]` 由 §5.3 第二判据算出（是否
开新段，仍然用批前冻结的 `p_hi_c`，这部分近似不变）。**orphan（`direct[τ]=False`
的 token）不会出现在任何 `t` 位置上，既不贡献 `nsg_incl`/`steps_since` 这类
按簇分组的前缀扫描，也不占用 Phase 1 本地缓冲的任何一行**——它们的语义距离
`> λ_new`，根本没有一个"既有簇"可供它们贡献 segment/pad 计数，混进这些公式
只会污染真正 direct token 所在簇的统计（例如让一个 haystack 簇的 `nsg_incl`
被一个凑巧 `c*` 相同、但实际是 orphan、根本不会真的 JOIN 这个簇的 token
污染）。orphan 的主 op 完全由 Phase 2 产生，参见上面"op_log 跨 Phase 的顺序
契约"一节。

**Segment id（对应上面新增 `current_segment` 要解决的问题）**：

```
nsg_incl[t] = 按 c*[t] 分组、按 t 排序，对 new_seg 做"组内包含自身"的前缀和
              # 标准 segmented inclusive cumsum：先按 (c*[t], t) 排序（稳定排序，
              # 组内保序=到达顺序），组内做 cumsum，再按原顺序 scatter 回去。
              # 这是 GPU 上有现成实现的原语（等价于 segment_csr / groupby 的
              # inclusive cumsum），不是新发明的算法。
op.segment[t] = current_segment[c*[t]] + nsg_incl[t]
```

这一个公式**同时**给 `JOIN` 和 `NEW_SEGMENT` 算出正确的 `segment`：`new_seg[t]`
为真时，`t` 自己的 `+1` 被计入 `nsg_incl[t]`，落在新值上；为假时，`nsg_incl[t]`
不含自身贡献，等于"目前为止本批同簇已经开过几段"，正是 `JOIN` 应该沿用的当前
段号。

**PAD_INSERT 的 count（对应 §5.11，也是这一轮要修的第②项）**：

`count` 依赖"这个新段开始前，level 0 当前逻辑占用数 mod `2^ℓ_block`"，这个量
本身是一个会在 `B′` 处折返的计数器（entry 满了触发 carry），看起来像需要模拟
一个有状态的过程。**但有一个关键化简：`PAD_INSERT` 的定义就是把这个 mod 计数器
补到 `≡0`，所以每次新段事件（pad 完、该 token 自己的 entry 落地后）计数器必然
精确回到 `1 mod 2^ℓ_block`——一个固定值，与之前发生过多少次 pad、pad 了多少个
都无关。** 于是这个计数器唯一依赖的是"距离本批内该簇上一次开新段过去了几个
真实 entry"（batch-start 之前的持久状态只在本批还没发生过新段事件时才需要）：

```
rank[t]        = 按 c*[t] 分组、按 t 排序，t 在自己这个簇的批内子序列中的
                  0-indexed 组内下标（第一个同簇 token 是 0，第二个是 1，……）

last_new_rank_before[t] = 严格排在 t 之前、且 new_seg=True 的同簇 token 里，
                  组内下标最大的那个；若不存在（本批内 c*[t] 在 t 之前还没
                  开过新段），取哨兵值 -1
                  # 标准 segmented "reset scan"：对 new_seg=True 的位置打上
                  # 自己的组内下标、其余位置置 -1，做一次**exclusive**（不含
                  # 自身）的组内前缀 max。exclusive 是关键——如果拿 inclusive
                  # 前缀 max 直接用，t 自己若 new_seg=True 会把自己算进去，
                  # 得到 last_new_rank_before[t] == rank[t]，下面的公式会用
                  # t 自己的下标减自己，产出荒谬的负数或零。exclusive 前缀
                  # max 可以直接用 inclusive 前缀 max 整体右移一位得到
                  # （标准技巧，同样是现成的向量化原语，不需要逐 token 串行）

steps_since[t] = rank[t] - last_new_rank_before[t] - 1   # 注意这个 "-1"，
                  # 两个分支都要减：
                  #   有上一次新段（last_new_rank_before[t] = r' ≥ 0）：
                  #     steps_since = rank[t] - r' - 1，不是 rank[t] - r'——
                  #     r' 这个 token 自己的那一步已经被"重置到 1"吸收掉了，
                  #     不能再算一次
                  #   本批内还没开过新段（哨兵 last_new_rank_before[t]=-1）：
                  #     steps_since = rank[t] - (-1) - 1 = rank[t]，也就是
                  #     "t 前面有多少个同簇批内成员"——同一个 "-1" 在两个
                  #     分支里自动给出正确结果，不需要为两个分支分别记两条
                  #     不同的公式

base_mod[t]  = 1                                   如果 last_new_rank_before[t] ≠ -1
                                                     （本批内 c*[t] 之前已经开过新段）
             = level_count[c*[t], 0] mod 2^ℓ_block  否则（本批这个簇还没开过新段，
                                                     用批前持久值）
prev_mod[t]  = (base_mod[t] + steps_since[t]) mod 2^ℓ_block
count[t]     = (-prev_mod[t]) mod 2^ℓ_block         # 只有 new_seg[t]=True 且
                                                       count[t] > 0 才产生
                                                       PAD_INSERT op（沿用既有
                                                       "count=0 不产生 op"规则）
```

> **这个 "-1" 不是可以省略的细节，是这条公式唯一容易做错的地方，必须显式钉死
> 并给出推导。** 从头验证：设一批内某簇的子序列按组内下标排好，`M_r` 表示
> 处理完第 `r` 个成员之后（即将处理第 `r+1` 个成员之前）的 mod 计数器值，
> `M_0 = level_count[cluster,0] mod 2^ℓ_block`（批前持久值）。递推关系是
> `M_{r+1} = 1`（若第 `r` 个成员 `new_seg=True`）或 `(M_r+1) mod 2^ℓ_block`
> （否则）——这就是"每次新段事件后计数器精确回到 1"这条化简的直接展开。
> 设 `r'` 是严格小于 `r` 的、最近一次 `new_seg=True` 的组内下标：从
> `M_{r'+1}=1` 开始，之后 `r-1-r'` 个成员全是 `new_seg=False`（否则 `r'`
> 就不是"最近一次"），每个贡献 `+1`，所以 `M_r = 1 + (r - 1 - r') =
> r - r'`。而 `prev_mod[t]` 就是 `M_{rank[t]}`——代入 `steps_since[t] =
> rank[t] - r' - 1` 和 `base_mod[t]=1`，`prev_mod[t] = 1 + (rank[t]-r'-1)
> = rank[t]-r'`，和直接展开算出的 `M_{rank[t]} = rank[t]-r'` 完全一致。
> 若不减这个 "1"（即错误地令 `steps_since = rank[t]-r'`），会算出
> `prev_mod[t] = rank[t]-r'+1`，比真值多 1，`count[t]` 也会系统性算错——
> 且两个分支（有无上一次新段）用同一个哨兵 `-1` 统一处理后，错误会以
> **同样的方向**（多算 1 步）同时出现在两个分支里，不会只在一个分支里
> 暴露，容易在单测覆盖不到"本批第一次开新段"这个边界情形时被放过。

**为什么这不需要像 §5.21-3 的 carry 那样上有界轮数的循环**：§5.21-3 的 carry
需要模拟"进位可能级联多少层"，这个深度虽然有静态上界（`L_alloc`）但每一层都要
真的算一遍，是不可避免的多轮迭代。这里不同——上面的化简把"计数器怎么折返"这个
本来需要模拟的细节，替换成了一个**无状态的组内相对位置查询**（"上一次归零点在
哪"），归零点本身用 `new_seg` 这个已知的布尔掩码直接定位，不需要真的把计数器
从 0 敲到 `B′` 再折返地模拟一遍——这是能把它压成一次分段扫描而不是有界多轮
迭代的根本原因，值得记录下来，否则容易被误认为"这里也要照抄 §5.21-3 的多轮
掩码 carry"。

**必须有 CPU 参考实现和对拍单测（S0.1，纯 CPU 可测）**：和 §5.21-2 对批量
ladder 写入的要求同一个模式——写一个逐 token 串行的朴素参考实现（对每个**到达
的 direct token**，先过滤掉 orphan，再读当前 `current_segment`/
`level_count[cluster,0]`，立即决定 `segment`/`count`，立即"执行"更新，供下一个
token 读到最新值），断言它和上面向量化公式在任意合成批次（含同簇多次开新段、
含多个不同簇交错到达、**含 direct 和 orphan 交替出现**）上逐位一致。这条测试
没通过之前，"segment id / PAD_INSERT count 的批量化是对的"这个论证是未经验证
的假设。

**上面只算出了每个 token 的 `segment`/`count`，还没说这些 op 怎么写进本地缓冲
——`PAD_INSERT` 和它服务的 `NEW_SEGMENT` 是两条 op，主操作永远只有一条，每个
token 展开出的 op 数量不一样，这是一个变长写入问题，Phase 1 是向量化路径，
不能逐 token 决定"我这条写在第几行"。**

```
extra[t]     = new_seg[t] and (count[t] > 0)   # 这个 token 除了主操作，还需要
                                                 # 额外一条 PAD_INSERT
main_idx[t]  = base + t + inclusive_prefix_sum(extra)[t]   # base 是本地缓冲
                                                              # 提交前的 local_op_len
                                                              # （Phase 1 总是本批
                                                              # 第一个写入者，实践中
                                                              # base=0，但公式写通用
                                                              # 形式）
若 extra[t]：pad_idx[t] = main_idx[t] - 1
token_idx[t] = batch_start_offset + direct_idx[t]   # 写进主 op 的 arg2，见前面
                                                      # "op_log 跨 Phase 的顺序
                                                      # 契约"一节——batch_start_
                                                      # offset 是这个 flush 批
                                                      # 第一个 token 在整条序列
                                                      # 里的绝对位置，direct_idx[t]
                                                      # 把压缩后的 t 映回批内原始
                                                      # 位置 τ，两者相加才是 replay
                                                      # 要用来索引 k_raw/v/pos 的
                                                      # 全局绝对 token 位置
```

`inclusive_prefix_sum(extra)[t]`（**含 `t` 自己**，标准的向量化前缀和，不是
新原语，§5.4 别处已经在用同类操作）是关键：它等于"到 `t` 为止，包括 `t` 自己，
总共有多少个 token 需要额外的 `PAD_INSERT` 槽位"。用一个小例子验证为什么必须是
inclusive（含自身）而不是 exclusive（不含自身）：三个 token `t=0,1,2`，只有
`t=1` 需要 pad（`extra=[F,T,F]`）。inclusive 前缀和是 `[0,1,1]`：

```
main_idx[0] = 0+0+0 = 0        # t=0 主操作 -> 本地缓冲第 0 行
main_idx[1] = 0+1+1 = 2        # t=1 主操作 -> 第 2 行；pad_idx = 2-1 = 1
                                #   -> 第 1 行是 t=1 的 PAD_INSERT，恰好紧邻在
                                #      它服务的 NEW_SEGMENT（第 2 行）之前
main_idx[2] = 0+2+1 = 3        # t=2 主操作 -> 第 3 行，紧接第 2 行，没有空隙
```

四行（0,1,2,3）恰好装下 `3` 个主操作 + `1` 个 pad，`local_op_len` 本批结束后
推进到 `4 = m + Σextra`。**如果错用 exclusive 前缀和**（不含自身），`t=1` 的
`main_idx` 会算成 `0+1+0=1`，`pad_idx=0`——但第 0 行已经被 `t=0` 的主操作占了，
两个 token 的 op 写进同一行，互相覆盖。**inclusive 是必须的，不是随意选择**：
`t` 自己若需要 pad，这条 pad 本该排在它主操作的前一行，而 `t` 自己贡献的那个
`+1`（inclusive 才有）恰好把它的主操作向后多推一行，腾出这个空位；exclusive
少算了 `t` 自己这一份，主操作和它自己的 pad 会挤到同一行。

**这条公式只覆盖生产路径的主操作 + `PAD_INSERT`。** 调试 build 的 `CARRY`
不参与这个索引分配——它不需要精确的行间位置关系（§5.21-2 已经说明重放从不读
`CARRY`），要不要把它也塞进这套向量化索引分配是调试工具自己的问题，不在这里
展开。**必须补的单测**（和上面 segment/count 的对拍单测同一批数据即可复用）：
断言按这套公式写出的本地缓冲，逐行类型和参数与逐 token 串行执行的参考实现
（每次决定一个 op 就模拟"追加"一次，行号自然递增）完全一致，尤其要覆盖"连续
多个 token 都需要 pad"和"本批最后一个 token 需要 pad"这两个边界情形。

**这个技巧的适用范围不止这里，顺带记一笔——这一轮已经从"留给以后"变成必须
履行的义务，见下方"Phase 3 拆成 3a/3b"一节**：Phase 3a 对 `n_eff`/`centroid`
的在线更新（§5.5）在没有 `γ` 衰减重启（即本批内该簇没有 `NEW_SEGMENT`）时会
逐项相消、化简成一个简单的批量加权和。

> **更正（这一轮修的）：上一版说"一旦出现 NEW_SEGMENT，就需要和上面完全
> 同构的『segmented reset scan』才能向量化……套用上面 steps_since/reset
> 的思路，不是另一个新问题"——这是错的，两者的"重置"不是同一类操作，不能
> 直接套用。** `PAD_INSERT` 的重置落在一个固定值上（mod 计数器精确回到
> `1`，与之前发生过多少次 pad、pad 了多少个都无关）——这正是"距离上一次
> 重置过了几步"这个无状态的相对量足以决定当前值的根本原因：重置把对
> 更早历史的依赖彻底切断了。但 `n_eff` 的 `γ` 衰减一般是"把累积历史打一个
> 折扣"（`n_eff_pre ← γ·n_eff`），**不是像 `PAD_INSERT` 那样固定重置到
> 同一个值**——`γ∈(0,1)` 时衰减后的值依然依赖**衰减前**的完整历史（不是
> 某个固定常数），而那段历史可能又包含更早的一次衰减。**`γ=0`（§5.5
> 明确支持的合法配置，"归零"分支）是个特例，需要单独说清楚，不能顺着
> 上面的逻辑简单推广成"γ=0 时就退化成 PAD_INSERT 那种固定重置，可以用
> steps_since"**：`γ=0` 时 `n_eff` 单独看确实精确归零、随后从固定值 `1`
> 重新计数，表面上像一次固定重置；但 `centroid`（`μ`）不是这样——它是
> 段内各成员 `k` 的加权和，即使 `n_eff` 归零后从 `1` 重新计数，`μ` 依然
> 需要真正累积每个成员的**内容**，不是一个能被压缩成"距离上一次重置几步"
> 这一个计数器的量。所以不论 `γ` 取哪个值（包括 `γ=0`），`steps_since`
> 这类单层查询本身都不足以求出 `μ`，必须知道每次衰减发生**那一刻**的
> 累积值（`n_eff` 侧）以及此后每个成员的实际内容（`μ` 侧），这是一条真正
> 跨越整个批内子序列的递推，不能被压成一个无状态的相对位置查询。完整
> 推导挪到下方"Phase 3a 的具体做法"一节给出，不再是"留一个指针，不重复
> 推导"，那节也给出了 `γ=0` 时上面这套仿射扫描仍然精确成立的说明（`A`
> 直接算出 `0`，公式自动退化成"只看衰减之后的内容"，不需要单独分支）。

**这一步不再是可以无限期推迟的优化项**：下方"Phase 3 拆成 3a/3b"一节会说明，
Phase 3a 必须在 Phase 2 开始之前完成，否则 Ward 合并会读到本批 Phase 1 贡献
缺失的 stale metadata。

#### Phase 2 的 Ward 合并会让 Phase 1 已经写下的 op 指向错误的槽——必须显式防止

**这是比 `p_hi_c` 冻结更严重的一类冲突，此前完全没处理。** Phase 1 用**批前**的
`alive`/`centroid` 快照把一批 token 里的大多数直接分配到某个槽（比如槽 X），并
已经把这些 `JOIN`/`NEW_SEGMENT` op 写进了本批的**本地** op 缓冲（还没提交进跨批
持久的 `op_log`，见下方"本地缓冲 vs 持久 op_log"）。**但 Phase 2 处理 orphan
时若触发 `K_max` 满的 Ward 合并（§5.6 的合并过程），会释放并复用某个槽**——如果被
释放复用的恰好是槽 X（Phase 1 这一批已经往里面写过东西的槽），Phase 1 写下的那些
`op.cluster=X` 就会产生歧义：它们该被理解成"合并前的旧簇 X"（内容已经被
`WARD_MERGE` 转移进了保留槽），还是"合并后占据槽 X 的新簇"（Phase 2 因为这个
orphan 而建立的、和旧簇 X 毫无关系的另一个语义身份）？两种理解在 replay 时会产出
完全不同的 cache 结构，而 op_log 本身的字段（`WARD_MERGE(keep_slot, free_slot)`
只记了合并这一步，没有记"这次合并是否使某个本批更早的 op 的 `cluster` 字段失效"）
不足以消歧。

> **上一轮的修法是"屏蔽 touched 槽 + 无候选时把 orphan 顺延到下一个 flush 批"，
> 这个修法本身有一个更严重的问题，必须推翻重做**：语义簇路径的"flush"复用的是
> 现有 recent window 溢出即驱逐的语义（`log_kv_cache.py:1169-1188`
> `add_recent`）——token 一旦被 recent window 挤出去，就**没有"还留在 recent
> window 里等下一批"这个中间状态**，它必须在同一步里进入某个真实的 (簇,段)。
> "顺延"意味着这个 orphan 本批完全不被写入任何 cache 槽位，但它已经不在 recent
> window 里了——这是一个"flush 出来了但 cache 里暂时没有"的语义空洞：这个 token
> 在顺延期间对 attention 到底可不可见？算不算 recent？要不要进 `op_log`？连续
> 顺延多批时因果顺序怎么保证？**这些问题没有一个能用现有架构自然回答**，因为
> "顺延"发明了一种现有系统里根本不存在的第三态（既不在 recent window、也不在
> ladder cache）。引入一个新的 staging buffer 来装它是可能的，但代价高、状态机
> 复杂，且没有必要——下面给出一个不需要这种 buffer 的修法。

**决定：让 Ward 合并保持完全不受限（和 §5.6 单 token 场景一模一样，不额外屏蔽
任何槽），且 `op_log`（不论本地缓冲还是持久存储）里任何已经追加的条目永不改写。**
这样每个 orphan 在它自己所在的批次内**总能**拿到一个真实槽位——`K_max ≥ 2` 时
Ward 永远有候选（§5.6 已证明的性质，不受本批是否有槽被"摸过"影响），deferred/
顺延这个概念因此整个不再需要，上面那一整段语义空洞问题随之消失。

> **上一轮在这里给出的修法是"向量化重定向改写"，这一轮把它推翻。** 上一轮的
> 判断是：重定向不仅无害，还能换来"`op_log` 对每个 token 的最终归属自解释"这个
> 额外好处，代价可控。**这个判断错了——重定向不是无害的锦上添花，它会产生真实的
> 语义错误，而且错误恰好发生在它本该覆盖的那个场景里。** 下面先给反例，再说
> "自解释"这个真实需求该怎么在不改写历史的前提下满足。

#### 为什么重定向是错的：一个具体反例

延续本节开头的场景：本批内，orphan 组 1 先触发 `NEW_CLUSTER(B)` 建立新簇 B；
紧接着，orphan 组 2 因为 `K_max` 已满也需要建新簇，Ward 代价矩阵在"全部 alive、
非对角线槽"里选出的最便宜候选恰好是 `(keep_slot=C, free_slot=B)`。**这不是需要
凑巧才会发生的边界情形，是大概率事件**：B 刚建立，`n_total` 极小（可能只有
1~2），Ward 的尺寸加权代价 `(n_a n_b)/(n_a+n_b)·‖μ_a−μ_b‖²` 对这种簇天然给出
全场最低代价——这正是 §5.6 那个"multi-needle 会被系统性漏斗到同一个簇"警告框
已经论证过的同一种偏好，只是这次发生在**同一个 flush 批之内**、发生在**刚刚
才被本批自己创建**的簇身上。而 Ward 候选池"全部 alive、非对角线的槽"从设计上
就没有排除这种候选，也不应该排除（排除它就是重新引入上一轮已经推翻的
`touched_mask`）。

按上一轮的重定向规则：orphan 组 1 更早写下的 `NEW_CLUSTER(B, 0, -1)`（以及组内
其余成员的 `JOIN`/`NEW_SEGMENT(B, ...)`）会被改写成 `arg0=C`，且改写后的条目
**留在原来的时序位置**（在 `WARD_MERGE(C, B)` 之前）。重放这个被改写过的序列会
产生两个独立的错误，任何一个都足以致命：

1. **段号错位**：重定向只碰 `arg0`，不碰 `arg1`。改写后的 `NEW_CLUSTER(C, 0, -1)`
   仍然带着 `arg1=0`——这个 `0` 的原意是"新簇 B 的第一段"，但重定向之后它被
   直接喂给重放循环的 `append_to_ladder(..., cluster=C, segment=0)`。C 是一个
   已经运行了很久的既有簇，真实的 segment 计数可能已经是 47——把这个刚到达的
   token 记成"C 的 segment 0"，直接违反 segment 单调性，下游任何按段号判断
   新旧的逻辑（§5.11 的对齐填充、§5.5 的 `γ` 衰减边界）读到的都是一个荒谬的
   历史段号。
2. **`WARD_MERGE` 引用的槽在重放状态里从未被创建过**：`WARD_MERGE` 类型本身被
   排除在重定向目标之外（这条没错），所以 `WARD_MERGE(C, B, -1)` 原样留在序列
   里。但 B 的建立事件（`NEW_CLUSTER(B,...)`）已经被重定向改写成指向 C 了——
   在重放重建出的状态里，**槽 B 从未被 `alive` 过**。重放执行到
   `WARD_MERGE(C, B)` 这一步时，`ward_merge_only(C, B)` 试图合并一个从未存在
   过的簇 B：若实现老实断言 `alive[free_slot]`，重放直接崩溃；若不做这条断言，
   会用槽 B 里的垃圾/零值内容去"合并"进 C，悄悄丢弃或污染 C 的真实 ladder 内容。

这两处都不是"补一个字段、加一个特判"就能堵住的漏洞——只要 Ward 允许合并本批内
刚建立的簇（而它必须允许，否则 §5.6"`K_max≥2` 时永远有候选"这条不变量就不成立），
重定向就会持续撞上"改写一个建立事件，却没有一并处理它的其它字段、以及依赖它
存在性的后续操作"这个结构性矛盾，修一个漏洞会长出下一个。

**正确的选择是完全不做重定向，`op_log` 是一份纯粹的、只追加、永不改写的执行
日志。** 这不需要放弃任何已经证明成立的性质——本节开头就已经论证过：**结构
操作和主操作共享同一条未经改写的时间线时，严格按 `op_log` 顺序重放本身就足以
正确重建 forward 的 cache 结构**。`WARD_MERGE(C, B)` 执行到的那一刻，B 在重放
状态里是刚刚被前面那些**未经改写**的 `NEW_CLUSTER(B,...)`/`JOIN(B,...)`真实
建立起来的、内容真实的簇，`ward_merge_only` 合并的是两个真实存在的 ladder，和
forward 当时发生的事情逐位一致——这不是一个需要额外机制去"凑出来"的性质，是
"不改写历史"这条最朴素的不变量的直接推论。**上一轮把"要不要重定向"和"重放
正不正确"混成了一个问题去解决，答案错在这里：重放正确性不需要重定向，重定向
反而会破坏它。**

#### 重定向想买的"自解释"，改用只读派生拿，不进 `op_log`

重定向除了（错误地）想保证正确性外，还有一个真实的动机：省得每次想知道"某个
token 最终在哪个簇里"都要完整重放一遍。**这个便利可以完全在 `op_log` 之外拿到，
不需要付出改写历史的代价：**

> **更正（这一轮修的）：上一版 `derive_final_cluster` 直接对裸槽号做并查集，
> 槽位复用会被错误合并成同一个身份，必须改成版本化身份。** 具体反例：
> `token0` 进入 `NEW_CLUSTER(B)`；之后 `WARD_MERGE(C, B)` 把 B 合并进 C（B 变成
> `free_slot`）；再之后又一次 `NEW_CLUSTER(B)`——B 被后续某个 orphan 复用，建立
> 了一个和"旧 B"毫无关系的全新簇，`token1` 属于这个新簇。上一版的
> `parent[find(op.free_slot)] = find(op.keep_slot)` 只认槽号，`token1` 记录时
> 的槽号也是 `B`，会被同一次并查集查找路由到 `C`——`derive_final_cluster` 把
> "旧 B 被并入 C"和"后来复用同一物理槽建立的新 B"错误地当成了同一个身份，
> `token1` 的最终归属被算错。**这正是 §5.4 反例（同批 Ward 合并选中刚建立的
> 簇）在离线派生工具这一侧的镜像问题**——正确执行路径（forward/replay）不会
> 犯这个错，因为它们严格按时间顺序执行、槽的"当前身份"永远是最后一次写入决定
> 的；但这个派生函数是**离线、跳过时间顺序、只看 `WARD_MERGE` 边**做并查集，
> 裸槽号不足以区分"同一个物理槽在不同时期代表的不同簇"，必须显式引入版本。
>
> **修法：给每个槽维护一个只在这个函数内部使用的 `epoch` 计数器，身份是
> `(slot, epoch)` 而不是裸 `slot`。** 每次遇到 `NEW_CLUSTER(slot, ...)`（无论是
> 该槽第一次被建立，还是被 Ward 合并释放后又一次被复用）就把该槽的 `epoch`
> 加一，这个 token 及此后同槽的 `JOIN`/`NEW_SEGMENT` 都记在这一代身份下；
> `WARD_MERGE(keep_slot, free_slot)` 只 union **当前这一代** `free_slot` 的身份
> 到**当前这一代** `keep_slot` 的身份，不触碰 `free_slot` 的 `epoch`——`epoch`
> 只在下一次真正的 `NEW_CLUSTER` 复用这个槽时才前进，从而让"旧 B"（被合并走的
> 那一代）和"新 B"（后来复用建立的下一代）天然是两个不同的并查集节点，不会被
> 误连。`epoch` 只是这个只读函数内部的临时簿记，不是持久 buffer，不影响
> forward/backward/replay 的任何状态——真正的执行路径本来就不需要它。

> **更正（这一轮修的）：函数目前只对"从冷启动开始的完整 op_log"安全，必须
> 显式声明这个前提，且给出能处理切片的方式，不能只靠调用方自觉。** 若拿
> batch-local 本地缓冲、或 `op_log` 的某个中间区间去单独调用它，
> `epoch.get(slot, 0)` 这个默认值会把"这段切片开始之前、在更早的部分里已经
> 被复用过的槽"重新当成"这段里第一次出现，epoch 0"处理——这正是这一轮修的
> 那个槽位复用 bug 的另一种触发方式：不是同一次调用内部弄错，而是**跨调用
> 边界**弄错。函数本身不可能从一段裸切片里推断出"这个槽在切片开始前已经被
> 用过几次"，必须由调用方显式提供。**修法：把 `epoch`/`parent` 做成可选的
> 种子参数，函数变成可以对同一条 `op_log` 分段串联调用的形式**，而不是只能
> 一次性喂完整日志。

> **更正（这一轮修的）：分段串联会让早期 token 的 final slot 过期，必须把
> "扫描/累积"和"解析出最终槽号"拆成两个函数，不能在扫完一段就提前算
> `final_slot`。** 上一版每次调用都返回本段的 `final_slot`（对本段
> `token_identity` 里的每个 token 立即用当前的 `parent` 状态跑一次
> `find()`）——问题是**后续段里的 `WARD_MERGE` 可能继续改变早期 token 的
> 最终归属**：假设第 1 段处理完时，`find()` 把 token 500 解析成槽 C，函数
> 照原设计把 `final_slot[500] = C` 返回给调用方；但第 2 段里发生了
> `WARD_MERGE(D, C)`——槽 C 的内容被合并进 D。**token 500 真正的最终归属
> 现在是 D，但调用方手里存的 `final_slot[500]` 还是过期的 C，而且没有任何
> 机制通知调用方这个值已经失效。** 只传 `epoch`/`parent` 状态不够，因为
> 早期 token 的 `(slot, epoch)` 版本化身份（`token_identity` 里的值）已经
> 在返回 `final_slot` 时被"解析掉"、没有保留下来——即使调用方后续想用新的
> `parent` 重新解析 token 500，也拿不到它当初的版本化身份了。**修法：把
> 函数拆成 `scan_op_log`（只扫描、累积 `token_identity`，从不调用
> `find()`、从不产出任何"最终"结果）和 `resolve_final_slots`（对累积下来
> 的**全部** `token_identity` 统一跑一次 `find()`）——调用方的职责变成
> "先把所有关心的段都 `scan_op_log` 完、把每段的 `token_identity` 累积到
> 一个字典里，最后才调用一次 `resolve_final_slots`"，中间任何一次
> `scan_op_log` 的返回值都不包含"最终归属"这个概念，也就没有"过期"的
> 可能性——这不是这个函数的一个可选优化，是唯一避免过期结果的做法。**

```python
def scan_op_log(
    op_log,
    initial_epoch: dict[int, int] | None = None,
    initial_parent: dict[tuple[int, int], tuple[int, int]] | None = None,
) -> tuple[dict[int, tuple[int, int]], dict[int, int], dict[tuple[int, int], tuple[int, int]]]:
    """只读、离线，扫一段 op_log（完整日志，或任意切片，比如一个 flush 批的
    本地缓冲）。**只累积状态，从不解析"最终归属"**——见上面的更正框，final
    slot 的解析被故意挪到一个单独的函数，且只应该在扫完全部你关心的段之后
    调用一次。从不写回 op_log，也不是 forward/backward 正确性契约的一部分。
    对 WARD_MERGE 的合并方向做一次并查集(union-find)，身份是 (slot, epoch)
    而不是裸 slot——见上方更正框，槽位复用后的新身份不能被误认成被合并走的
    旧身份。

    前提（函数不做任何隐式校验，调用方必须自己保证）：
    1. `op_log` 参数必须是**有效前缀**（按 §5.13 的 `op_log_len` 截断，不是
       整个静态 `(OP_max, 4)` buffer）。
    2. 主操作必须携带真实的 `token_idx`（arg2，§5.21-2 这一轮的更正——旧格式
       `arg2` 恒为 `-1` 的日志不能喂给这个函数，`token_identity` 直接用
       `op.token_idx` 做 key，不再用递增计数器去猜"这条 op 对应哪个 token"，
       原因见 §5.4"op_log 跨 Phase 的顺序契约"）。
    3. 若这是从冷启动（所有槽从未 `alive` 过）开始的完整日志：
       `initial_epoch`/`initial_parent` 留默认 `None`。若这只是一段切片：
       必须传入这一段开始之前的 `epoch`/`parent` 状态——用上一段调用的
       返回值直接传进来即可串联。

    返回三元组：(本段的 token_identity：token_idx -> 记录时的版本化身份；
    本段结束时的 epoch 状态；本段结束时的 parent 状态)。后两者原样传给下一段
    调用的 initial_epoch/initial_parent；第一项**必须由调用方自行累积**（和
    之前所有段的 token_identity 合并到一个字典里），不能丢弃，因为
    `resolve_final_slots` 需要看到全部 token 才能给出不会过期的答案。"""
    epoch = dict(initial_epoch) if initial_epoch else {}    # 槽号 -> 当前 epoch
    parent = dict(initial_parent) if initial_parent else {}  # (槽号,epoch) -> (槽号,epoch)
    def find(v):
        while parent.get(v, v) != v:
            parent[v] = parent.get(parent[v], parent[v])
            v = parent[v]
        return v

    token_identity = {}               # token_idx -> 记录时的版本化身份 (slot, epoch)
    for op in op_log:
        if op.type == NEW_CLUSTER:
            epoch[op.cluster] = epoch.get(op.cluster, -1) + 1   # 首次建立 -1->0，
                                                                  # 每次复用再 +1
            token_identity[op.token_idx] = (op.cluster, epoch[op.cluster])
        elif op.type in (JOIN, NEW_SEGMENT):
            token_identity[op.token_idx] = (op.cluster, epoch.get(op.cluster, 0))
        elif op.type == WARD_MERGE:
            keep_v = (op.keep_slot, epoch.get(op.keep_slot, 0))
            free_v = (op.free_slot, epoch.get(op.free_slot, 0))
            parent[find(free_v)] = find(keep_v)   # 只 union 当前这一代，不碰 epoch 本身

    return token_identity, epoch, parent


def resolve_final_slots(
    token_identity: dict[int, tuple[int, int]],
    parent: dict[tuple[int, int], tuple[int, int]],
) -> dict[int, int]:
    """把（可能横跨多次 scan_op_log 调用累积下来的）token_identity，用**最终**
    的 parent 状态解析成每个 token 的最终物理槽号。**只应该在扫完你关心的
    整段 op_log 之后调用一次**——如果后面还会有更多 op_log 段要 scan，这里
    算出来的结果可能被后续的 WARD_MERGE 弄过期，不要提前调用、也不要缓存
    中间结果当作最终答案。"""
    def find(v):
        while parent.get(v, v) != v:
            v = parent.get(v, v)
        return v
    return {t: find(v)[0] for t, v in token_identity.items()}
```

**返回值是 `dict[int, int]`——每个 token 最终的物理槽号，不是 `(slot,
epoch)`。** 这不是文档偷懒省略了 epoch，是可以证明安全的：**在
`resolve_final_slots` 被调用的那一刻（扫完全部你关心的段之后），同一个
物理槽号不可能同时有两个不同 epoch 的身份还是并查集的根**。因为按顺序
契约，任何复用某个槽号的 `NEW_CLUSTER`（把该槽 `epoch` 从 `e` 推进到
`e+1`）必然紧邻在释放这个槽号的 `WARD_MERGE` 之后——也就是说，`epoch=e`
的身份在 `epoch=e+1` 出现之前就已经被 `parent[find((slot,e))] =
find(keep_v)` 指向别处，不再是根。所以对 `token_identity` 里出现的任意
版本化身份，`find(v)[0]` 在同一时刻至多对应一个仍是根的身份——`[0]`
天然无损，不会把两个不同的最终簇错误地折叠成同一个标签。**下游（比如
`experiments.md` 的成对共簇一致率）可以放心只用这个整数做标签，不需要、
也不应该退回未解析的 `(slot, epoch)`**——`experiments.md` 引用这个函数
时统一说"最终物理槽号"，不要再写"`(slot, epoch)` 身份"，两种说法字面上
指的不是一回事，容易诱导实现者去改函数签名塞回 epoch，而那其实是多余的。

**典型用法（串联多段）**：

```python
all_identity, epoch, parent = {}, None, None
for segment in op_log_segments:               # 完整日志，或按 flush 批切片
    identity, epoch, parent = scan_op_log(segment, epoch, parent)
    all_identity.update(identity)              # 累积，不丢弃

final_slot = resolve_final_slots(all_identity, parent)   # 只在这里、扫完全部
                                                            # 段之后调用一次
```

`scan_op_log` 本身仍然可以按段串联调用（`epoch`/`parent` 是这段计算的全部
"记忆"，从上一段的返回值原样传进下一段），这部分不变；变化的只是"什么时候
可以安全地把结果当作最终答案"——答案是"扫完全部你关心的段之后，且只调用
`resolve_final_slots` 一次"，不是"每扫完一段就问一次"。

**不新增任何持久 buffer,不改变 §4/§5.13 的内存账目**——`epoch`/`parent` 是
调用方自己持有、自己决定生命周期的普通 Python 对象，不属于 cache 状态的一
部分。S0.8 对拍、调试工具需要"token 最终去了哪"时调用它,forward/backward 的
正确性路径永远不依赖它、也不会被它影响。

#### S0.8 的 Ward 事件比较需要合并前的成员快照——`scan_op_log` 不提供，需要一个专用姊妹函数

`experiments.md` §6 S0.8 决策门的 Ward 事件分歧统计要求对齐两条路径各自的
`WARD_MERGE` 事件、比较"合并前 `keep`/`free` 两个簇各自包含的原始 token
集合有多相似"，但上面 `scan_op_log` 的 `WARD_MERGE` 分支只做了
`parent[find(free_v)] = find(keep_v)`——它从不维护"某个 `(slot,epoch)`
身份此刻实际持有哪些 token"这个反向索引，扫到 `WARD_MERGE` 的那一刻吐不出
任何可以拿来算相似度的东西。**不要改 `scan_op_log` 本身去做这件事**——
它的返回值和调用方式已经被 `resolve_final_slots`、S0.1 的重放正确性
测试等多处依赖，它足够简单可信的原因正是"只维护解析最终归属需要的最小
状态"，把 S0.8 才需要的东西塞进去会破坏这一点。新增一个专供 S0.8 用的
姊妹函数（下面会看到，它维护的不是精确 token 集合本身，而是一个有界的
近似表示——理由见下方更正框）：

> **更正（这一轮修的）：存完整 `frozenset[int]` 有 O(T²) 的内存/时间风险，
> 必须换成有界的近似表示。** 上一版每次 `WARD_MERGE` 都把 `keep`/`free`
> 两侧**当前完整的** token 集合各复制进一条 `WardEvent`；`keep` 侧尤其
> 危险——它往往是一个持续吸收新成员的大簇（Ward 的尺寸加权本就偏好把
> 小簇并进大簇，§5.6），体积可以逼近整条序列。上面 §5.21-2 的 `OP_max`
> 摊还论证已经证明过 `WARD_MERGE` 事件数在病态输入下最坏是 O(T)——O(T)
> 个事件、每个最坏 O(T) 个成员，总量是 O(T²)：Stage 0 的 dump 规格
> （`experiments.md` §6）目标是 32k token/prompt，用 Python `frozenset[int]`
> （每个元素算上对象头和哈希表槽位约 50~80 字节）估算这个乘积能到数十
> GB，会直接把 Stage 0 拖死——这不是危言耸听的边界情形，是 `OP_max`
> 论证里已经证明过的同一个最坏情况在这里的另一次出现，必须给出有界方案，
> 不能假设"实践中不会这么糟"。
>
> **修法：把精确 `frozenset` 换成固定宽度的 MinHash 草图**，只服务它
> 唯一的下游消费者——`experiments.md` S0.8 要的是 Jaccard 相似度这一个
> 标量，不是集合本身，而 Jaccard 允许有界误差的估计（这一项是诊断信号，
> 不是决策门，决策门只挂在 3b，见 `experiments.md` §6）。
>
> **更正（这一轮修的）：叫它"MinHash（bottom-k）"是名不副实，必须纠正
> 措辞，不能只是笔误放过。** "bottom-k" 特指**单个**哈希函数、取整个集合
> 哈希值里最小的 `k` 个（一个无序的 k 元子集），它的 Jaccard 估计量和
> 合并规则都是"取两边 `2k` 个候选值里最小的 `k` 个,再看有几个来自两边
> 交集"，比这里写的实现更省样本但更复杂。**下面的伪代码实际实现的是
> 经典的『`k` 个独立哈希函数各自取 min』方案**（Broder 1997 的原始
> MinHash，有时也称 k-hashes MinHash）：`k` 个位置逐一独立，每个位置
> 存一个标量最小值，合并是逐位 `elementwise_min`，Jaccard 估计量是"两个
> 草图逐位相等的比例"——这是两种不同的、不能混用估计公式的实现，写成
> "bottom-k" 会让实现者去查 bottom-k 的估计量/合并公式，套到这里就是错的。
> 全文统一改称 **"k-hashes MinHash"** 或直接说"MinHash 草图"，不再用
> "bottom-k"这个限定词。
>
> MinHash 的关键性质是**集合并运算下精确合成，不是"近似的近似"**：
> `sketch(A ∪ B) = elementwise_min(sketch(A), sketch(B))` 对任意有限宽度
> `k` 都精确成立；唯一的误差来自用有限宽度估计 Jaccard 本身，标准误差
> 上界 `0.5/√k`（在真实 Jaccard=0.5 处最大，`k=128` 时 ≈4.4%）。这让它可以
> 直接嵌进现有的增量式更新逻辑：每个身份的存储从"正比于它当前成员数"变成
> 恒定的 `k` 个 `uint64`，总内存 O(事件数 × k)，与 `T` 完全无关——不是把
> 常数因子改小，是把渐近复杂度本身改掉。代价：`keep_sketch_before`/
> `free_sketch_before` 不再能重建出精确成员列表，只能估计 Jaccard；若某次
> 调试确实需要某个具体事件的精确成员，用 `scan_op_log`/`resolve_final_slots`
> 重放到该事件的 `trigger_token_idx` 为止即可精确重建（多付一次 O(T) 重扫，
> 但只在真的需要时才付，不是默认路径）。精确的**成员数**（不是成员本身）
> 代价是 O(1)，单独维护一个计数器即可，不需要靠草图估计。`k`（草图宽度）
> 是 S0.8 专用的调试/分析参数，不属于核心算法（不进 §5.13 buffer 表、不
> 影响 forward/backward）。
>
> **更正（这一轮修的）：光写 `k` 不够复现，`minhash_of_token` 不能当成
> 像 `apply_rope` 那样"不用规定内部细节"的原语——`apply_rope` 是唯一
> 确定的数学操作，任何正确实现都逐位一致；但"一个哈希函数"不是唯一
> 确定的，不同实现（Python `hash()`、不同 PRNG、不同 murmur/xx 变体）
> 给出完全不同的输出，两次 Stage 0 dump 如果用了不同的哈希实现，草图
> 之间就不可比、连带 Jaccard 数字失去意义，而这条错误没有任何报错信号，
> 只会安静地污染诊断数字。必须钉死到"任何人照此实现都逐位复现"的程度。**
> 采用 [SplitMix64](https://prng.di.unimi.it/splitmix64.c)（Vigna 提出，
> Java `SplittableRandom` 的种子扩展器，公开、简单、无歧义的标准 64 位
> 混合函数）：全部算术按 `uint64` 回绕语义（mod `2**64`）执行，Python
> 实现必须显式 `& 0xFFFFFFFFFFFFFFFF` 或使用 `numpy.uint64`/
> `numpy.uint64` 数组，**不能用裸 Python `int`**（无溢出，静默不复现
> C/numpy 版本的回绕结果）。

```python
MASK64 = 0xFFFFFFFFFFFFFFFF

def _splitmix64_next(state: "uint64") -> "tuple[uint64, uint64]":
    """标准 SplitMix64 一步：返回 (推进后的 state, 本步输出)。全部按 uint64
    回绕语义。"""
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    z = z ^ (z >> 31)
    return state, z

WARD_EVENT_SKETCH_K = 128   # MinHash 草图宽度；S0.8 专用调试参数，见上方更正框，
                              # 标准误差上界 0.5/sqrt(128) ≈ 4.4%（Jaccard=0.5 处最大）
WARD_EVENT_HASH_ALGO = "splitmix64-v1"   # 必须和 k、master_seed 一起写进 eval
                                           # metadata——三者共同决定草图可不可比
WARD_EVENT_MASTER_SEED = 0   # 唯一的自由参数；k 个子种子从它确定性推导

def make_minhash_seeds(k: int, master_seed=WARD_EVENT_MASTER_SEED) -> "list[uint64]":
    """从单个 master_seed 迭代 SplitMix64 k 次，派生 k 个互相独立的子种子——
    这是 SplitMix64 的标准用法（同一算法既用来出草图哈希、也用来扩展自己的
    种子），不需要另外维护一份种子列表。"""
    state, seeds = master_seed, []
    for _ in range(k):
        state, seed_i = _splitmix64_next(state)
        seeds.append(seed_i)
    return seeds

MINHASH_SEEDS = make_minhash_seeds(WARD_EVENT_SKETCH_K)   # 固定种子，整个 Stage 0
                              # dump 全程复用同一份——不同事件的草图必须用同一族哈希
                              # 函数才可比
EMPTY_SKETCH = np.full(WARD_EVENT_SKETCH_K, MASK64, dtype=np.uint64)
                              # 空集的草图：全体取 min 的幺元，任何真实 token 的
                              # 哈希值都比它小，第一次插入即正确覆盖。**不再用作
                              # "身份缺失时的默认值"**——见下方 strict 校验一节，
                              # 这个语义已被证明是错的

def minhash_of_token(token_idx: int, seeds=MINHASH_SEEDS) -> "np.ndarray[K]":
    """单个 token 的 MinHash 草图：k 个独立哈希值，h_i(token_idx) =
    splitmix64_next(seeds[i] ^ token_idx)[1]——把第 i 个子种子和 token_idx
    异或后过一轮 SplitMix64。"""
    return np.array([_splitmix64_next(s ^ np.uint64(token_idx))[1] for s in seeds],
                     dtype=np.uint64)


class WardEvent(NamedTuple):
    trigger_token_idx: int | None       # 见下方"触发 token 的定义"。只有仍在
                                          # scan_op_log_for_ward_events 内部
                                          # 传递、尚未被下一条 NEW_CLUSTER 消费
                                          # 的 pending 事件才可能是 None——顺序
                                          # 契约（§5.4 第 2 条）保证任何被真正
                                          # append 进函数**返回的** ward_events
                                          # 列表的 WardEvent，这个字段恒为 int，
                                          # 因为 append 只发生在
                                          # `pending._replace(trigger_token_idx=...)`
                                          # 之后，从不直接 append 带 None 的
                                          # pending 本身（见下方实现）
    keep_identity: tuple[int, int]      # 合并后保留的 (slot, epoch)
    free_identity: tuple[int, int]      # 被合并、释放的 (slot, epoch)
    keep_sketch_before: "np.ndarray[K]" # 合并前 keep 的 MinHash 草图（uint64[K]）
    free_sketch_before: "np.ndarray[K]" # 合并前 free 的 MinHash 草图（uint64[K]）
    keep_size_before: int               # 合并前 keep 的精确成员数（不经草图，
                                          # 单独维护的计数器，无估计误差）
    free_size_before: int               # 合并前 free 的精确成员数


def estimate_jaccard(sketch_a, sketch_b) -> float:
    """标准 MinHash Jaccard 估计量：两个草图里逐位相等的比例。无偏，标准
    误差上界 0.5/sqrt(K)（K=WARD_EVENT_SKETCH_K，在真实 Jaccard=0.5 处
    最大）。"""
    return float(np.mean(sketch_a == sketch_b))


def best_orientation_jaccard(event_a: WardEvent, event_b: WardEvent):
    """两个已经按 trigger_token_idx 匹配上的 WardEvent，给出方向无关的相似度。

    `keep`/`free` 是"物理保留槽/释放槽"这个实现细节的产物——`ward_merge_only`
    合并 (C, X) 时哪一个当 keep、哪一个当 free 由代价矩阵/槽位选择逻辑决定，
    不是语义上稳定的有序对。若两条路径（批量近似、严格串行参考）合并的是
    同一对语义簇，但各自选择了相反方向的 keep/free（批量路径把 C 当 keep、
    严格串行把 X 当 keep），直接按标签比较
    `estimate_jaccard(event_a.keep_sketch_before, event_b.keep_sketch_before)`
    会把两个不同的簇错误地凑在一起比较，产出一个虚假的低 Jaccard——事件本身
    明明是同一次合并的强对齐（trigger_token_idx 相等），却被误判成"两侧都差"。

    修法：把两种配对方向都试一遍，取总相似度更高的一种；`keep`/`free` 不再是
    两个跨路径可比的固定标签，只作为"同一次结果内部" side_1/side_2 的
    区分——orientation 是否发生翻转单独报告，不藏进相似度数字里。

    **更正（这一轮修的）：单选"总相似度更高的一种"没有考虑到每个 Jaccard
    本身就是一个标准误差 ≈0.5/√k 的估计值。** 两种方向的总分很接近时，
    `orientation_flipped` 报告的可能只是抽样噪声，不是两条路径的 Ward
    候选选择真的存在系统性差异——如果不加区分地把这个标志当成可信信号去
    归因（比如"orientation 翻转频繁，说明两条路径的合并方向选择不一致"），
    会把噪声误读成结论。**修法**：新增 `orientation_margin`（两种方向总分
    之差的绝对值）和 `orientation_ambiguous`（margin 小于一个基于 `k` 的
    噪声阈值时为真）。阈值推导：单个 `estimate_jaccard` 的标准误差上界是
    `0.5/√k`；`margin` 是两个"各由两个独立 Jaccard 估计求和"的量之差，
    四个估计项近似独立，方差可加，`margin` 的标准误差因此约为
    `sqrt(4)·(0.5/√k) = 1/√k`——取约两倍标准误差作为阈值，
    `orientation_ambiguous = margin < 2/√k`（`k=128` 时 ≈17.7%）。
    `ambiguous=True` 时 `orientation_flipped` 这个二元判断本身不可信，
    不该被用来做进一步归因；`side_1_jaccard`/`side_2_jaccard`（仍然取
    总分更高的那一种配对）依然是当前最优的点估计，不受这条标记影响——
    ambiguous 标记的是"方向"这个判断不可信，不是"相似度数值"不可信。
    """
    fwd = (estimate_jaccard(event_a.keep_sketch_before, event_b.keep_sketch_before),
           estimate_jaccard(event_a.free_sketch_before, event_b.free_sketch_before))
    swapped = (estimate_jaccard(event_a.keep_sketch_before, event_b.free_sketch_before),
               estimate_jaccard(event_a.free_sketch_before, event_b.keep_sketch_before))
    fwd_sum, swapped_sum = sum(fwd), sum(swapped)
    margin = abs(fwd_sum - swapped_sum)
    ambiguous = margin < 2.0 / np.sqrt(WARD_EVENT_SKETCH_K)   # 见上方推导
    flipped = swapped_sum > fwd_sum
    side_1, side_2 = swapped if flipped else fwd
    return dict(side_1_jaccard=side_1, side_2_jaccard=side_2,
                orientation_flipped=flipped, orientation_margin=margin,
                orientation_ambiguous=ambiguous)


def scan_op_log_for_ward_events(
    op_log,
    initial_epoch: dict[int, int] | None = None,
    initial_parent: dict[tuple[int, int], tuple[int, int]] | None = None,
    initial_sketches: dict[tuple[int, int], "np.ndarray[K]"] | None = None,
    initial_sizes: dict[tuple[int, int], int] | None = None,
    initial_pending: "WardEvent | None" = None,
):
    """只读、离线，S0.8 专用，不是 forward/backward/S0.1 依赖的东西——和
    scan_op_log 是两份独立的扫描逻辑，只是复用同一套 (slot,epoch) 身份和
    union-find 记号，**不能**和 scan_op_log 混用同一份 epoch/parent 状态
    跨函数传递（sketches/sizes/pending 是这个函数独有的状态，scan_op_log
    从不维护它们）。

    在 scan_op_log 的基础上额外维护两个反向索引：sketches（身份 -> 当前
    成员集合的 MinHash 草图）和 sizes（身份 -> 当前精确成员数），随
    NEW_CLUSTER/JOIN/NEW_SEGMENT 增量更新；每遇到一次 WARD_MERGE，在真正
    执行合并之前，把 keep/free 双方**此刻**的草图与成员数各自快照进一条
    WardEvent（快照只是复制 K 个定长数值 + 两个整数，代价与真实成员数
    无关），再把两者合并（free 并入 keep：草图取逐位 min，成员数相加，
    见上方更正框"精确合成"），供后续继续扫描时使用。

    trigger_token_idx 按 §5.4"op_log 跨 Phase 的顺序契约"第 2 条（结构
    操作紧邻它服务的主操作）填上：任何 WARD_MERGE 后面紧跟的下一条 op
    必然是它服务的 NEW_CLUSTER，用一个 pending 指针记住"刚追加、还没等到
    trigger_token_idx 的 WardEvent"，下一次遇到 NEW_CLUSTER 时直接填上它
    的 token_idx，不需要向前看（lookahead）；一个 orphan（组）的建簇请求
    最多触发一次 Ward 合并（腾出恰好一个槽），所以 pending 在被下一条
    NEW_CLUSTER 消费之前不会被第二次覆盖——这不是需要容忍的情形，用
    assert 钉死。

    **可以和 scan_op_log 一样分段串联调用，但 pending 必须显式作为
    种子参数/返回值传递，不能像上一版那样是函数内部的局部变量。** 如果
    调用方按任意边界切片 `op_log`（`scan_op_log` 的 docstring 明确允许
    "完整日志，或任意切片"），切片边界完全可能恰好落在某个 WARD_MERGE
    和它服务的 NEW_CLUSTER 之间——如果 pending 只是局部变量，扫完这一段
    时这个尚未补上 trigger_token_idx 的 WardEvent 会随函数返回直接丢失，
    且没有任何信号告诉调用方"漏了一个事件"。**修法**：pending 做成第六个
    可选种子参数，返回值里也带上它，和 epoch/parent/sketches/sizes 走同一
    套"调用方负责在段之间原样传递"的纪律。**这里没有一个类似
    `resolve_final_slots` 的额外"finalize"步骤**——WardEvent 一旦被
    append 进 ward_events 就是最终结果，不会像 token_identity 那样被
    后续操作弄过期；调用方唯一要做的事是：**扫完你关心的最后一段之后，
    显式断言最终返回的 `pending is None`**——`WARD_MERGE` 后面按顺序
    契约必然紧跟它的 `NEW_CLUSTER`，一份扫到底的完整日志不可能以一个
    悬空的 `WARD_MERGE` 收尾，这条断言失败只可能是调用方自己的切片/拼接
    逻辑有 bug（漏了一段、段的顺序错了），不是这个函数的正常行为，也不是
    这个函数自己能替调用方判断的事（它不知道这是不是"最后一段"）。"""
    epoch = dict(initial_epoch) if initial_epoch else {}
    parent = dict(initial_parent) if initial_parent else {}
    sketches = dict(initial_sketches) if initial_sketches else {}   # 身份 -> 草图
    sizes = dict(initial_sizes) if initial_sizes else {}             # 身份 -> 精确成员数
    def find(v):
        while parent.get(v, v) != v:
            parent[v] = parent.get(parent[v], parent[v])
            v = parent[v]
        return v

    def _require(store: dict, ident, what: str):
        """严格查找,替代此前的 `.get(ident, EMPTY_SKETCH)`/`.get(ident, 0)`
        防御式默认值——更正见下方 blockquote。缺失就地报错,不垫一个"看起来
        合法"的空草图/0 计数。"""
        if ident not in store:
            raise AssertionError(
                f"scan_op_log_for_ward_events: 引用了未知的簇身份 {ident}"
                f"（缺失于 {what}）——多半是 initial_sketches/initial_sizes 没有"
                f"正确传递，或 op_log 切片边界不合法")
        return store[ident]

    ward_events: list[WardEvent] = []
    pending = initial_pending   # 刚 append、还缺 trigger_token_idx；可能是
                                  # 上一段传进来的，也可能在本段内产生
    for op in op_log:
        if pending is not None:   # 见下方更正框：顺序契约必须在这里现场校验
            assert op.type == NEW_CLUSTER, (
                f"scan_op_log_for_ward_events: 顺序契约违反——WARD_MERGE 之后"
                f"必须紧跟它服务的 NEW_CLUSTER，但遇到了 {op.type}")
            assert op.cluster == pending.free_identity[0], (
                f"scan_op_log_for_ward_events: 顺序契约违反——NEW_CLUSTER 写入"
                f"的槽 {op.cluster} 不是这次 WARD_MERGE 刚释放的槽 "
                f"{pending.free_identity[0]}（§5.4 Phase 2 步骤 4 的既有约定："
                f"释放出来的槽立即交给这个 orphan 的新簇使用，这里违反了这条"
                f"约定）")
        if op.type == NEW_CLUSTER:
            if pending is None:   # 没有紧邻的 WARD_MERGE 腾位——见下方更正框⑤,
                                    # 这条 NEW_CLUSTER 只能是 K 未满/冷启动分支
                assert epoch.get(op.cluster, -1) == -1, (
                    f"scan_op_log_for_ward_events: NEW_CLUSTER 写入的槽 "
                    f"{op.cluster} 之前已经被用过（scanner 记录的当前 epoch="
                    f"{epoch.get(op.cluster, -1)}），但这条 NEW_CLUSTER 前面"
                    f"没有 pending 的 WARD_MERGE 为它腾位。按 §5.6 的 K 未满/"
                    f"K 已满两分支表：K 未满/冷启动的 NEW_CLUSTER 只能落在"
                    f"从未 alive 过的槽上（`epoch` 恒为 -1）；已经被用过的槽"
                    f"要复用必须先经过 WARD_MERGE（会在这里留下 pending 记录）"
                    f"——这条 op 两者都不满足，说明 K 未满分支选中了一个不该"
                    f"选的、已经在用的槽")
            epoch[op.cluster] = epoch.get(op.cluster, -1) + 1
            ident = (op.cluster, epoch[op.cluster])
            sketches[ident] = minhash_of_token(op.token_idx)
            sizes[ident] = 1
            if pending is not None:                      # 顺序契约保证：紧邻
                ward_events.append(pending._replace(trigger_token_idx=op.token_idx))
                pending = None
        elif op.type in (JOIN, NEW_SEGMENT):
            ident = (op.cluster, epoch.get(op.cluster, 0))
            sketches[ident] = np.minimum(_require(sketches, ident, "sketches"),
                                          minhash_of_token(op.token_idx))
            sizes[ident] = _require(sizes, ident, "sizes") + 1
        elif op.type == WARD_MERGE:
            assert pending is None    # 上一个 Ward 事件必须已被消费，见 docstring
            raw_keep = (op.keep_slot, epoch.get(op.keep_slot, 0))
            raw_free = (op.free_slot, epoch.get(op.free_slot, 0))
            assert raw_keep == find(raw_keep), (
                f"scan_op_log_for_ward_events: WARD_MERGE 引用的 keep_slot "
                f"{op.keep_slot}（scanner 当前记录的 epoch={raw_keep[1]}）不是"
                f"当前根——这个身份已经在更早的一次合并里被吸收掉了，op_log 里"
                f"这条 WARD_MERGE 引用了一个 stale/non-root 的身份；若放任 "
                f"find() 把它悄悄改判成祖先身份，等于让并查集『自动修正』一个"
                f"本该报错的上游日志错误，必须在这里就地失败，不能被吸收掉")
            assert raw_free == find(raw_free), (
                f"scan_op_log_for_ward_events: WARD_MERGE 引用的 free_slot "
                f"{op.free_slot}（scanner 当前记录的 epoch={raw_free[1]}）不是"
                f"当前根——同上，stale/non-root 身份必须报错，不能被 find() "
                f"静默吸收")
            keep_v, free_v = raw_keep, raw_free
            assert keep_v != free_v, (
                f"scan_op_log_for_ward_events: self-merge，keep_v == free_v == "
                f"{keep_v}——生产路由的代价矩阵会 mask 对角线（§5.6）排除这种"
                f"情况，但这里扫的是 op_log，不应该假设日志一定合法；不挡住会"
                f"静默损坏状态：下面 sizes[keep_v]=keep_size+free_size 会把同一个"
                f"身份的计数翻倍，随后 sketches.pop(free_v)/sizes.pop(free_v) 又"
                f"会把刚更新的条目整个删掉，把一次自合并变成一次静默的状态丢失")
            keep_size, free_size = _require(sizes, keep_v, "sizes"), _require(sizes, free_v, "sizes")
            assert keep_size > 0 and free_size > 0, (
                "scan_op_log_for_ward_events: keep/free 的精确成员数不能是 0"
                "——size 只会在 NEW_CLUSTER 时置 1、此后单调不减，精确为 0 只"
                "可能是状态维护本身有 bug，不是合法输入")
            pending = WardEvent(
                trigger_token_idx=None,   # 下一条 NEW_CLUSTER 补上
                keep_identity=keep_v, free_identity=free_v,
                keep_sketch_before=_require(sketches, keep_v, "sketches").copy(),  # O(K) 拷贝，
                free_sketch_before=_require(sketches, free_v, "sketches").copy(),  # 与真实成员数无关
                keep_size_before=keep_size,
                free_size_before=free_size,
            )
            sketches[keep_v] = np.minimum(sketches[keep_v], sketches[free_v])
            sizes[keep_v] = keep_size + free_size
            sketches.pop(free_v, None)
            sizes.pop(free_v, None)
            parent[free_v] = keep_v

    return ward_events, epoch, parent, sketches, sizes, pending
```

> **更正（这一轮修的）：五处此前会静默吞掉 bug，必须改成硬失败。**
> ① **`pending`/`NEW_CLUSTER` 的紧邻关系此前只是"等到了就消费"，从不校验
> "等到的是不是它"。** 原循环只在 `op.type == NEW_CLUSTER` 分支内部检查
> `pending is not None`，中间如果插了一条 `JOIN`/`NEW_SEGMENT`/
> `PAD_INSERT`/`CARRY`，会被直接跳过、不产生任何信号——docstring 声称的
> "紧邻"契约实际上从未被真正校验过，只是恰好在正确输入下表现得像是成立。
> 一旦这份契约在别处被打破（比如 op_log 顺序被后续改动不小心破坏），这个
> 函数会把 `trigger_token_idx` 错配给一个不相关的 `NEW_CLUSTER`，产出的
> WardEvent 看起来完全合法，实际已经污染。**修法**：循环体最前面新增
> `if pending is not None: assert op.type == NEW_CLUSTER`——只要上一步
> 产生了 pending，本步不是 `NEW_CLUSTER` 就立刻现场报错，不再有"跳过几条
> op 之后才补上"这条从未被授权过的路径。
> ② **`.get(ident, EMPTY_SKETCH)`/`.get(ident, 0)` 把"缺失状态"伪装成
> "合法的空集合"。** 引用一个从未被这次调用（含种子）见过的身份，本该是
> 调用方的 bug（忘了传 `initial_sketches`/`initial_sizes`，或者 `op_log`
> 切片不合法）——但防御式默认值会让它看起来像一次真实存在、只是恰好是
> 空集的合并/加入，产出 size=0 的"合法"WardEvent，诊断数字被悄悄污染而
> 没有任何报错。**修法**：新增 `_require` helper，所有读取 `sketches`/
> `sizes` 的地方一律换成严格查找；额外在构造 `WardEvent` 前断言
> `keep_size_before > 0 and free_size_before > 0`——一个已知身份的 size
> 精确为 0 是另一类不该发生的状态维护 bug，值得单独一条断言而不是被
> `_require` 顺带盖住（`_require` 抓的是"身份完全未知"，这一条抓的是
> "身份已知但计数被错误清零"，两种失效模式不同）。`EMPTY_SKETCH` 不再
> 用作缺省值，只保留它"min 幺元"这个数学定义本身。
> ③ **只校验了"紧跟着的是不是 `NEW_CLUSTER` 类型"，没校验"这个
> `NEW_CLUSTER` 用的是不是刚释放出来的那个槽"。** §5.4 Phase 2 步骤 4
> 的既有约定是 `slot_idx = free_slot`——释放出来的槽立即交给这个
> orphan（组）建新簇，不是随便一个 alive 槽。若上游（真正执行路由/写
> 日志的代码，不是这个扫描函数本身）因为某个 bug 让 `WARD_MERGE` 之后
> 紧跟的 `NEW_CLUSTER` 落在了另一个槽上，①的类型检查照样通过——它只看
> `op.type`，不看 `op.cluster`——函数会把 `trigger_token_idx` 安到一个
> 和这次合并毫不相关的 token 上，产出的 `WardEvent` 表面完全合法，实际
> 已经张冠李戴。**修法**：紧跟在①的类型断言之后，追加
> `assert op.cluster == pending.free_identity[0]`——`pending.
> free_identity` 是这次 `WARD_MERGE` 构造时就已经记录好的
> `(free_slot, epoch)` 身份（见 `WARD_MERGE` 分支的 `free_v =
> find((op.free_slot, ...))`，构造那一刻它还未被 union，`find` 对它是
> 恒等映射，所以 `free_identity[0]` 精确等于当时的 `op.free_slot`），
> 取其槽号分量与这条 `NEW_CLUSTER` 实际写入的槽号比较，不等就地报错。
> 这个函数存在的意义就是"不假设 op_log 一定合法"（本节其它断言的一贯
> 立场），槽号复用是这条设计里少数几个跨结构操作/主操作的强不变量，
> 值得和类型检查放在同一处、同等严格地校验。
> ④ **`WARD_MERGE` 分支构造 `keep_v`/`free_v` 时直接对
> `(op.keep_slot, epoch.get(op.keep_slot,0))`/`(op.free_slot,
> epoch.get(op.free_slot,0))` 做 `find()`，从未校验这两个"原始操作数"在
> `find()` 之前本身就该是当前根。** 这两个原始 tuple——不是 `find()` 之后
> 的结果——理论上必须恒等于自己的根：`op.keep_slot`/`op.free_slot` 是生产
> 路由的 Ward 代价矩阵从"当前 alive 槽位"里选出来的（§5.6），而
> `epoch.get(...)` 取的正是 scanner 自己维护的、该槽当前存活身份对应的
> epoch，两者拼起来在 op_log 合法的前提下必然是一个从未被 union 过的根。
> 但这只是"op_log 合法"这个假设下才成立的推论，而这个函数存在的全部意义
> 正是不假设 op_log 合法（③ 的立场）——如果上游有 bug 让某个已经被更早一次
> `WARD_MERGE` 吸收掉的 stale 身份又被错误地当成新一次合并的操作数写进日志，
> 直接 `find()` 会**静默把它解析到正确的祖先身份**，日志层面的错误被并查集
> 结构"自动修好"，不会报错，也就永远不会被发现。**修法**：`find()` 之前先
> 分别构造 `raw_keep`/`raw_free`，断言 `raw_keep == find(raw_keep)` 与
> `raw_free == find(raw_free)`——这就是"是否为当前根"的判据（`find` 对根
> 恒等，对非根一定会走到别处），断言通过后再把 `keep_v`/`free_v` 直接赋值
> 为 `raw_keep`/`raw_free`（此时已经证明两者相等，不需要再调用一次
> `find()`）。这条检查独立于③——③ 抓的是"合并之后新簇建在了错误的槽上"，
> ④ 抓的是"合并本身的输入操作数就已经引用了一个不该存在的身份"，两种输入
> 都会在功能上"看起来能跑"，只有分别加断言才能都拦住。
> ⑤ **`NEW_CLUSTER` 分支只在"紧邻 pending 的 WARD_MERGE"这条路径上校验过
> （③），"没有 pending 时凭空冒出一个 NEW_CLUSTER"这条路径完全没有校验。**
> 按 §5.6 的 K 未满/K 已满两分支表：K 已满时复用一个槽必须先经过 Ward 合并
> （会在这里留下 `pending`）；K 未满/冷启动时选的必须是"从未 `alive` 过"的
> 槽（`(~alive).float().argmax(-1)`，第一个空位）。这意味着一条没有
> `pending` 的 `NEW_CLUSTER`，它写入的槽在 scanner 的 `epoch` 记录里必须
> 是"从未见过"（`epoch.get(op.cluster,-1) == -1`）——上游若有 bug 让 K 未满
> 分支错误地选中了一个其实已经在用的槽（比如"空位"判定逻辑本身有 bug），
> 这个函数此前不会发现，会直接把它当成一次合法的新增身份处理，静默地让
> 两个不同的逻辑簇共享同一个物理槽而不触发任何冲突信号。**修法**：
> `pending is None` 分支下追加 `assert epoch.get(op.cluster,-1) == -1`。
> 和①②③④一样，这条检查独立捕捉一种③④都覆盖不到的输入：③④校验的是
> "跟在 WARD_MERGE 后面的 NEW_CLUSTER 对不对"，这条校验的是"不跟在
> WARD_MERGE 后面的 NEW_CLUSTER 对不对"，两类路径互斥、合起来才是
> `NEW_CLUSTER` 的完整校验面。
>
> **这五处校验目前只加在 `scan_op_log_for_ward_events` 里，姊妹函数
> `scan_op_log`（S0.8 cluster assignment divergence/co-assignment 用的
> 主解析器，见 `experiments.md` S0.8 第①项）完全没有——这是刻意的，不是
> 遗漏：`scan_op_log` 的存在理由就是"足够简单、足够小，容易独立确信正确"
> （§5.4"不改 `scan_op_log` 本身"一节），继续往里堆断言会削弱这个理由。
> 需要这五类校验时，调用方应该先对同一段 `op_log` 跑一遍
> `scan_op_log_for_ward_events`（哪怕丢弃它的 `WardEvent`/sketch 输出，
> 只要它的断言全部通过），把它当一次独立的合法性校验，再放心调用
> `scan_op_log`/`resolve_final_slots` 去算真正要的东西——`experiments.md`
> S0.8 一节已经据此更新为显式要求这个先后顺序。**

**典型用法（分段串联，末尾断言 `pending` 已被消费干净）**：

```python
all_events = []
epoch = parent = sketches = sizes = pending = None
for segment in op_log_segments:            # 任意边界的切片都可以，不要求
                                              # 和 flush 批边界对齐
    events, epoch, parent, sketches, sizes, pending = scan_op_log_for_ward_events(
        segment, epoch, parent, sketches, sizes, pending
    )
    all_events.extend(events)

assert pending is None   # 扫完全部段之后才检查；不为 None 说明切片/拼接
                           # 逻辑本身有 bug（漏段、段序错），不是这个函数
                           # 的正常行为，见上方 docstring
```

**"触发这次合并的 orphan 的绝对 token 位置"精确定义为 `tok0`——不是新
概念，是 §5.4 Phase 2 早就命名过的量，不需要另外发明。** Phase 2 的合并
过程里，"K 已满"分支腾出槽位后，`NEW_CLUSTER(slot_idx, 0, tok0)` 写的
`tok0` 就是"这个 orphan（组）里到达顺序最早的那个 token"（见 §5.4"K 未满/
冷启动必须复用同一个 `allocate_new_cluster`"一节）——`WARD_MERGE` 服务的
正是这同一个 orphan（组）的建簇请求，两者必然紧邻（顺序契约第 2 条），所以
`trigger_token_idx` 就是紧随其后那条 `NEW_CLUSTER` 的 `token_idx`，上面的
函数正是这么取的。**但 `trigger_token_idx` 只是一个 path-local 的对齐键，
"不受路由近似影响"这个说法过强，必须收回。** 它在单条路径内部良定义、不
依赖任何实现细节，这一点确实成立："哪个 token 是这个 orphan 组里到达顺序
最早的那个"只依赖这条路径自己的输入数据和路由决策，不需要另外发明"哪个
位置算触发点"这类规则。但两条路径的路由决策本身可以不同：批量路径的
Phase 1 用批前冻结的 centroid 快照做直接/orphan 判定，严格串行参考逐 token
重算——若同一段输入因此在两条路径下分出了不同的 orphan 分组（比如某个
token 在一条路径下因为快照尚未反映最新 centroid 而被判定为 orphan，在另
一条路径下因为重算后的 centroid 已经足够近而被直接并入既有簇），"这个
orphan 组里到达顺序最早的那个 token"这件事本身在两条路径下可以有不同的
答案，即同一次语义上的合并事件在两条路径下产出不同的 `trigger_token_idx`。
**这不是需要修补的 bug，是这个对齐机制必须承受、也已经承受了的正常情形**：
`experiments.md` 按 `trigger_token_idx` 精确匹配两条路径的 `WardEvent`
列表，值相同代表强对齐（两条路径连"谁是最早成员"都一致）；只在一条路径
出现的 `trigger_token_idx` 不强行配对，计入 `ward_event_inserted`/
`ward_event_deleted`——这正是为"两条路径在这一点分歧"准备的处理路径，不是
一个理论上不会触发的兜底分支。

**本地缓冲 vs 持久 `op_log`**：本节说的"本地 op 缓冲"是 Phase 1/2 处理**当前
这一个 flush 批**期间用的临时张量，**不是**跨批持久存在的 `op_log`
（`(B,G,OP_max,4)`，见 §5.13）。批内 Phase 1（一次性向量化写入）和 Phase 2
（串行循环，逐个处理 orphan，含内联的 Phase 3b）都往这个本地缓冲里追加，
整个批处理完（Phase 1 → Phase 3a → Phase 2 全部结束后）才**一次性**把本地
缓冲追加进持久 `op_log`。这个两级结构本身不是新设计
——批量化路由本就需要一个地方暂存"这一批算出来的 op"；**它纯粹是一个提交
粒度问题，和上面推翻的重定向无关**——本地缓冲和持久 `op_log` 一样，条目一旦
追加就不再改写，两者遵守同一条不变量，只是提交时机不同。

> **本地缓冲的构建和消费不受 `record_op_log` 影响，训练/推理两条路径都会
> 执行——不要把它和"持久 `op_log`"的 gating 混为一谈。** `record_op_log`
> 精确的门控范围见 §5.21"`op_log` 只能在训练路径分配"一节：它唯一决定的是
> "这一批（Phase 1 → Phase 3a → Phase 2 全部结束、本地缓冲已经写满）之后，
> 要不要把本地缓冲整体拷贝进跨批持久的 `op_log`"，仅此一步。本节从头到尾
> 描述的 Phase 1/2/3 机制本身——DP-means 路由、Ward 合并候选选择与执行、
> segment id/`PAD_INSERT` count 的向量化 scan、Phase 3a/3b 的 metadata
> 更新、ladder 的物理写入——是让 cache 在两条路径上行为一致的核心机制
> （CLAUDE.md §10 死因 1"训推走同一条路径"），本地缓冲只是这套机制内部用来
> 在一次 flush 批处理期间传递"这一步产生了什么 op"的临时载体，不是
> `op_log` 持久化功能的附属品。**`record_op_log=False` 时它照常被构建、
> 照常驱动 Phase 3a/3b 和 ladder 写入，只是这一批处理完之后不再被拷贝进
> 任何持久结构，随即被丢弃或被下一批覆盖复用**。本地缓冲自身的容量
> （`local_op_cap = 4×flush_granularity`，默认配置下 512 行 int32，见
> 下方）是**每个 `(batch 元素, KV group)` 独立的一份**——和 §5.13 buffer
> 表里其它张量前两维都是 `(B,G)` 的约定一致，本地缓冲的完整形状是
> `(B,G,local_op_cap,4)`，不是全局或整层共享一份：`512×4×4B ≈ 8KB` 只是
> 单个 `(b,g)` 切片的大小，`B=1,G=8`（本文档惯用的展示口径，和持久
> `op_log` 的"(1,8,4·32768,4) int32 ≈16MB/层"算法一致）下一层是
> `8KB×8≈64KB`，28 层约 **1.8MB**。仍然远小于持久 `op_log` 的
> `OP_max=4·T_max`（32k 下约 448MB），两条路径都构建它的开销可以忽略
> 不计，不会重新引入那笔训练专属的内存账目——但这笔小账本身要按
> `(B,G,层数)` 完整展开才是"约 1.8MB"，不能只留单个 `(b,g)` 切片的
> "约 8KB"，否则又会重演之前"一笔小账被误读成全局账"的错误。

> **本地缓冲的容量不是"这一批的 token 数"，是"这一批的 token 数 × 每 token 的
> op 倍数"**——早期版本把本地缓冲的大小写成"就是这一批的 token 数，`≤128`"，
> 只数了主操作。但本地缓冲既然要承载**最终会提交进持久 `op_log` 的完整内容**，
> 就必须同时容纳这一批可能产生的结构操作——`WARD_MERGE`、`PAD_INSERT`，调试
> build 下还有 `CARRY`。全局 `OP_max = c·T_max`（`c=4`，§5.21-2 的推导）已经是
> 按"每个 token 最坏情况对应 1 主操作 + 1 WARD_MERGE + 1 PAD_INSERT + CARRY
> 余量"算出来的，本地缓冲的容量要用**同一个倍数**：`local_op_cap = 4 ×
> flush_granularity`（默认 flush 粒度 128 时是 512），不能只按 token 数分配，
> 否则本地缓冲会在结构操作密集的批次里溢出。
>
> **生产默认不写 `CARRY`（§5.21-2）时，`4×flush_granularity` 有安全余量，不需要
> 额外处理。但 debug/对拍 build 打开 `CARRY` 后，这条余量是不是仍然够用没有被
> 证明过**——§5.21-2 的 `OP_max` 推导把 `CARRY` 归为"低一个量级、可并入余量"，
> 那是对整条序列 `T_max` 的摊还论证（O(T/B′)），不等于对**单个 flush 批**内的
> 最坏 burst 也成立（一个批内如果恰好触发多层级联进位，`CARRY` 条数可能短时
> 集中）。**必须在追加前显式检查是否会超过 `local_op_cap`，超出则硬失败**（沿用
> §5.21-2"矩形预分配 + 硬失败、不做动态扩容、不做静默截断"的一贯原则），不能让
> 对拍场景里的本地缓冲静默溢出或截断——截断的本地缓冲提交进持久 `op_log` 后，
> 会产出一份看似完整、实则丢了尾部结构操作的日志，比直接崩溃更危险。

**`op_log`（本地缓冲和持久存储一样）里没有"重定向目标"这回事——任何 op 类型的
任何字段，一旦追加，永不被后续操作改写。** 六类 op 的字段各自的语义（`arg0`
是槽号还是 count）仍然不同，但这只影响"写入时填什么值"，不影响"写入后会不会被
改"——后者对全部六类 op 统一是"不会"。这比按类型区分"是否重定向目标"更简单，
也不会再有第五类、第六类 op 需要重新判断"它算不算重定向目标"这种问题。

```
Phase 2 处理每个 orphan（或一小簇互相接近的 orphan，由批内 mini DP-means 决定，
见下方"同批多个 orphan 要不要互相 join"）时，若 K_max 已满、需要 Ward 合并腾位：

    1. Ward 代价矩阵不做任何本批专属的额外屏蔽——就用 §5.6 已有的"屏蔽对角线
       与 dead 槽位"，取 (keep_slot, free_slot) = argmin Δ。**候选池不排除
       本批内刚建立的簇**——这正是上面反例的前提，保留它是故意的，不是疏漏
    2. ward_merge_only(keep_slot, free_slot)   # §5.6 的 1-2 步（更正后的编号，
                                                 # 见 §5.6），纯状态 mutation，
                                                 # 不写 op_log。物理内容此刻已经
                                                 # 正确合并、free_slot 已
                                                 # alive=false，但还没有任何
                                                 # 新簇写进去
    3. append WARD_MERGE(keep_slot, free_slot, -1) 到本地缓冲   # 日志写入是
       # 调用方（这里）的职责，不是 ward_merge_only 自己的职责，见 §5.6 的更正
       # （WARD_MERGE 是结构操作，arg2 保留 -1，不需要 token_idx）
    4. slot_idx = free_slot   # 释放出来的槽立即交给这个 orphan（组）的新簇使用
       —— 这一步开始就是 allocate_new_cluster(slot_idx, group) 这个共享原语
       的内容（见本小节末尾"K 未满/冷启动复用同一个原语"一段），**只**初始化
       结构性状态：alive[slot_idx]=true，
          centroid/n_eff/n_total/p_hi_c/current_segment 全部清零（"白纸"状态，
          不沿用旧簇残留值）。
          **数值元数据不在这里赋值**——统一交给 Phase 3b（内联，见下方
          "Phase 3 拆成 3a/3b"一节）从这个 orphan（组）的主操作重新构造，
          避免和 Phase 2 这一步的初始化重复计入同一批 token
       append NEW_CLUSTER(slot_idx, 0, tok0) 到本地缓冲，对应这个 orphan（组）
       里**到达顺序最早**的那个 token（绝对位置记为 p0，**绝对 token_idx 记为
       tok0**——位置和 token_idx 是两个不同的量，见"op_log 跨 Phase 的顺序
       契约"一节，orphan 自己在 Phase 2 里既知道它在序列中的绝对位置 `p_t`
       用于 g_max 判据，也知道它的绝对 `token_idx` 用于喂给 op 的 arg2，两者
       都要显式带着，不能只留一个）
       local_p_hi[slot_idx]    = p0   # Phase 2 私有的临时状态，只活在本次
       local_segment[slot_idx] = 0    # Phase 2 调用期间——不是 p_hi_c/全局
                                        # segment 计数，见下方说明

    若这个 orphan 组里还有其它成员（mini DP-means 分进同一临时簇的其余
    token，按到达顺序逐个处理）：**它们不能再产生 `NEW_CLUSTER`**——每个 token
    恰好对应一个主操作，`NEW_CLUSTER` 已经被组内最早的成员用掉了。**它们的
    `JOIN`/`NEW_SEGMENT` 判据必须读 `local_p_hi[slot_idx]`，不能读全局
    `p_hi_c[slot_idx]`**——全局 `p_hi_c` 此刻仍是第 4 步刚清零的占位值，如果
    这里读全局值，
    `p_t − p_hi_c` 恒等于 `p_t − 0`，对任何有意义长度的文档都会远超 `g_max`，
    组内除最早成员外的所有成员都会被错误地判成"时序打断"，被迫各自开一个新
    segment——而它们本来就是同一次 mini DP-means 判定为彼此接近、大概率也在
    时间上紧挨着到达的一组 token。对每个后续成员（绝对位置 `p_t`，绝对
    token_idx 记为 `tok_t`），按组内到达顺序：

        if p_t − local_p_hi[slot_idx] > g_max:
            local_segment[slot_idx] += 1
            append NEW_SEGMENT(slot_idx, local_segment[slot_idx], tok_t) 到本地缓冲
            （照常触发 §5.11 的 PAD_INSERT，插在这条 NEW_SEGMENT 之前）
        else:
            append JOIN(slot_idx, local_segment[slot_idx], tok_t) 到本地缓冲
        local_p_hi[slot_idx] = p_t   # 不论 JOIN 还是 NEW_SEGMENT，p_hi 都要推进
                                       # 到"最近一次收到成员的位置"——这是 p_hi
                                       # 的定义（§5.13），和是否开新段无关

    这和"一个已存在的簇后续收到新成员"用的是同一套判据（§5.3 第二个判据），
    唯一的特殊之处是这个簇是本批刚建的，所以判据读的是 Phase 2 自己维护的
    局部状态而不是全局 buffer。
```

> **`local_p_hi`/`local_segment` 只是 Phase 2 处理单个 orphan 组时的临时脚本
> 状态，不是新增的持久 buffer，不进 §4/§5.13 的内存账目，Phase 3a/3b 也不
> 需要读它。** 它按 `slot_idx` 存在一个小 dict/scratch 数组里，某个 slot 被
> 第 4 步重新 `NEW_CLUSTER` 时（不论是首次建立还是本批内被 Ward 合并释放后
> 再次复用）直接覆盖重置，不需要跨 orphan 组保留，处理完当前 orphan 组即可
> 丢弃。Phase 3a/3b 判断"这是不是一次新 segment"直接读 op 类型本身
> （`NEW_SEGMENT` vs `JOIN`），
> §5.5 的 γ 衰减规则只依赖"这条 op 是不是 `NEW_SEGMENT`"，不需要重新计算时序
> 判据，因此也不需要知道 `local_p_hi` 的具体数值——两者是完全独立的状态，互不
> 依赖。
>
> **为什么这个问题只出现在 Phase 2 的 orphan 组，Phase 1 不需要类似修复**：
> Phase 1 对整批 token 使用同一份**批前冻结**的 `p_hi_c` 快照做时序判据——这是
> 一个已经承认、已经在 S0.8 测的近似（§5.4 开头"这是一个近似"那段）。它读到的
> 值虽然不是全批最新的，但**始终是一个真实存在过的历史值**（这个簇在本批开始
> 前最后一次收到成员的位置），只是没有随批内进展更新，误差方向明确、幅度
> 有界（顶多让本该 `JOIN` 的 token 误判成 `NEW_SEGMENT`，不会反过来）。Phase 2
> orphan 组的问题性质不同：第 4 步把 `p_hi_c` **清零**到一个不代表任何真实
> 历史的占位值，如果后续成员直接读这个占位值，得到的不是"稍微过时的近似"，
> 而是"用一个哨兵值参与运算"——这是一个正确性 bug，不是近似误差，必须用
> `local_p_hi` 修掉，不能归入 Phase 1 那类"已知、可接受、留给 S0.8 测量"的
> 近似里一并放过。

**`K` 未满/冷启动必须复用同一个 `allocate_new_cluster(slot_idx, group)` 原语，
不能各写一套**：上面第 4 步到"这和『一个已存在的簇后续收到新成员』用的是
同一套判据"结束的全部内容——`alive`/元数据清零、`append NEW_CLUSTER`、
`local_p_hi`/`local_segment` 初始化、组内后续成员的 `JOIN`/`NEW_SEGMENT`
判据循环——**只依赖一个已经确定是空槽的 `slot_idx`，和这个 `slot_idx` 是
"Ward 合并腾出来的"还是"本来就空着的"完全无关**。§5.6 的"新簇形成的条件"表把
`K 未满`写成一句"占用 `alive` 为 false 的第一个槽位"，`冷启动`写成"第一个
flush 的 token 必然建簇 0"——这两行只回答了"`slot_idx` 从哪来"，没有回答
"拿到 `slot_idx` 之后怎么初始化"，容易让实现者误以为这两条路径不需要
`current_segment` 清零、不需要 `local_p_hi`/`local_segment`，因为详细的
初始化步骤只在"K 已满"分支里写全了。**这是错的：三条路径（冷启动、K 未满、
K 已满）在"如何初始化一个新簇"这一步必须是同一份代码，唯一的差别在前一步
"如何拿到一个空槽"**：

```
def new_cluster_slot() -> int:
    if K_max_slots_full():
        keep_slot, free_slot = argmin_ward_cost(...)   # K 已满：先合并腾位
        ward_merge_only(keep_slot, free_slot)
        append WARD_MERGE(keep_slot, free_slot, -1) 到本地缓冲
        return free_slot
    else:
        return (~alive).float().argmax(-1)   # K 未满/冷启动：直接找第一个空槽，
                                                # 冷启动时"全部 alive=False"是
                                                # 这条路径的一个特例，不需要
                                                # 单独分支

slot_idx = new_cluster_slot()
allocate_new_cluster(slot_idx, group)   # 上面第 4 步开始的全部内容，三条路径
                                          # 共用同一次调用，不分叉
```

**`K` 未满/冷启动分支里，`alive[slot_idx]` 进入 `allocate_new_cluster` 之前
为什么必然是"白纸"，还有一个隐含前提必须显式点出**：Ward 分支的"白纸"是
`ward_merge_only`/第 4 步显式 `zero_()` 出来的，`K` 未满分支的"白纸"则依赖
"一个从未被 `alive` 过的槽，它的 `centroid`/`n_eff`/`n_total`/`p_hi_c`/
`current_segment` 本来就是全零"——**这个前提不是自动成立的，必须由
`reset_parameters()` 在整个 cache 生命周期开始时显式 `torch.zeros(...)`
（或等价的显式清零）保证，不能依赖 `torch.empty` 之类不保证清零的分配**，
否则一个"从未使用过"的槽可能带着未初始化的垃圾内存，`allocate_new_cluster`
里"数值元数据不在这里赋值，统一交给 Phase 3a/3b 重新构造"这条设计（依赖
`n_eff_pre==0` 触发 §5.5 的直接替换分支）会在这类槽上失效——`n_eff` 如果不是
真正的 0 而是垃圾值，`n_eff_pre==0` 这个判据就不成立，代码会走进错误的加权
混合分支。这条前提和 §5.11"填充槽必须显式清零，不能依赖默认值"是同一类
纪律，放在这里一并点出，不是重复。

**Ward 合并候选池不受限（不排除本批内新建的簇）不再需要任何补偿机制**：上一轮
需要重定向，是因为担心"Phase 1 已经写下的、引用了后来被合并掉的槽的 op"会在
重放时产生歧义。**这个担心本身不成立**——只要不改写历史，`op_log` 里每一条已
写入的 op，在它被写入的那一刻引用的槽号，就精确对应"重放执行到这一步时，那个
槽号真正代表的簇"，因为重放是按同一条时间线顺序执行的，不存在"当前槽映射"和
"历史槽引用"对不上的问题——这类"对不上"只有在有人事后改写历史记录时才会出现，
而现在没有人这么做了。批量化（Phase 1 一次性向量化 scatter、Phase 2 串行处理
orphan）本身也完全不受影响——它们只往本地缓冲**追加**，从不回头改写已经写入的
条目，"改写本身能不能整批向量化完成"这个问题因此根本不需要被回答，因为已经没有
改写这一步。

**同批多个 orphan 要不要互相 join**：这个问题和上面的槽位冲突是两回事，**已经
由 §5.4 Phase 2 自身的设计回答了**——"在这批 orphan 内部跑一个小 DP-means"意味着
互相接近的 orphan 在**分配槽位之前**就已经被分进同一个临时簇。但"共享同一个
新簇"和"共享同一条 `NEW_CLUSTER` op"是两件事——见上面伪代码最后一段：组内只有
**到达顺序最早**的成员产生 `NEW_CLUSTER`，其余成员对同一个 `slot_idx` 产生
`JOIN`/`NEW_SEGMENT`，这是被"每个 token 恰好一个主操作"这条 §5.21-2 的既有约束
逼出来的，不是新规则；批内 join 关系（谁和谁分进同一个临时簇）在 Phase 2 开始时
已经通过 mini DP-means 确定好了，不依赖任何"已建立槽位是否可见"的运行时状态。

**Phase 3（拆成 3a/3b，见下方更正框）是本批"主操作 token 内容贡献"唯一的写入
点——对 Phase 1 分配到的既有簇和 Phase 2 新建的簇一视同仁，不再有第二个入口。**
这是相对更早一轮的一处简化，直接消掉了一类此前存在的 bug：更早一轮让 Phase 2 的
"新簇写进去"那一步（旧编号第 5 步）**同时**给新簇的 centroid/`n_eff`/`n_total`/
`p_hi_c` 按 orphan（组）内容设初值，随后 Phase 3 又对**整个**本地缓冲统一跑一遍
同样的更新——这会把新建簇的 orphan token 计入两次：一次在 Phase 2 初始化时，
一次在 Phase 3 统一更新时。上面第 4 步已经改为"只置 `alive=true`，数值元数据
全部清零"，原因就在这里：**Phase 2 不再对任何 token 的内容做数值更新，Phase 3
是唯一做这件事的地方，天然不会有双计数**。

> **这句话必须精确到"token 内容贡献"，不能笼统说成"Phase 3 是这些字段唯一的
> 写入点"——后者是过强表述，会和 §5.6 的 `ward_merge_only` 直接冲突。**
> `ward_merge_only` 的步骤 1（"合并簇级元数据"）本来就会写 `μ_new`/`n_eff_new`/
> `n_total_new`/`p_hi_new`——这是必须存在的第二个写入来源，不是需要消灭的
> 冗余：它写的是**把两个既有簇已经积累的历史值合并成一个**，不处理本批任何
> 一个 token 的原始内容，和 Phase 3 处理的"这批新到达的 token 该怎么在线更新
> centroid"是两类不重叠的写入。
>
> **但"两者写入范围不重叠"不等于"顺序上互不影响"——这里有一个此前被漏掉的
> 真实冲突，必须先解决，下面的"Phase 3 具体做法"才立得住。**

#### 未闭合的漏洞：Phase 3 若推迟到批末，Ward 合并会读到本批 Phase 1 的 stale metadata

**具体反例**（沿用本节前面"Phase 2 的 Ward 合并会让 Phase 1 已经写下的 op 指向
错误的槽"那节已经用过的场景，这次盯住 metadata 而不是 ladder）：Phase 1 用批前
快照把某个 direct token `tok0` 路由到既有簇 X，写下 `JOIN(X, seg, tok0)`；紧
接着 Phase 2 处理某个 orphan，`K_max` 已满，Ward 代价矩阵选中
`(keep_slot=C, free_slot=X)`——X 恰好是 Phase 1 这一批刚写过 `tok0` 的那个簇
（前面那节已经论证过，这不是边界情形，是 Ward 尺寸加权在"批内刚被触碰、暂时
看起来还小"的簇上的常规偏好）。

如果 Phase 3（`centroid`/`n_eff`/`n_total`/`p_hi_c`/`current_segment` 的 §5.5
在线更新）严格等到 Phase 1 **和** Phase 2 全部跑完才统一运行，`ward_merge_only
(C, X)` 执行的那一刻，X 的 metadata 仍然是**批前**的快照——`tok0` 加入 X 这件
事只体现在 op_log 和（已经按前面"Phase 1 整体"原则物理写好的）X 的 ladder 里，
还没有体现在 X 的 `n_total`/`μ`/`p_hi_c` 上。于是：

1. **`tok0` 对 X 的内容贡献在 Ward 合并这一步被悄悄丢弃**：`ward_merge_only`
   把 X 的（批前、不含 `tok0`）`n_total`/`μ`/`p_hi_c` 合并进 C，`tok0` 虽然
   物理上已经躺在被并入 C 的 ladder 里，但它对**元数据**的贡献从未进入 C 的
   `n_total`/`μ`——C 的 `n_total` 会比它真实持有的 entry 数少 1，C 的 `μ` 也
   没有真的混入 `tok0` 的 key。
2. **紧接着 X 被复用建立一个全新的簇（orphan 的新簇），若 Phase 3 仍按"批末
   统一 walk 本批 ops，用 `op.cluster` 分组"这个规则处理，会把 `JOIN(X, seg,
   tok0)` 这条 op 错误地喂给复用后的新簇**——`op.cluster` 只是一个裸槽号，
   这一步（不同于离线的 `derive_final_cluster`）从未引入过 `(slot, epoch)`
   版本化身份，无法区分"合并前的旧 X"和"合并后复用的新 X"。`tok0` 的 key
   内容因此被错误地混进一个和它毫无关系的簇的 centroid，而它真正所属的簇
   （现在活在 `keep_slot=C`）永远收不到这份贡献。

**这正是"op_log 跨 Phase 的顺序契约"和"Phase 1 整体先于 Phase 2 整体"这两条
已经确立的原则本该覆盖、却被漏掉的一个推论**：那两条原则解决的是 **ladder**
（entry 物理内容）的时间线问题——要求 Phase 1 的向量化 ladder 写入必须在
Phase 2 开始前完整落地，这样 Ward 合并才能看到真实、最新的 ladder。但
**metadata 是一条独立于 ladder 的时间线**，"Phase 3 批末运行"这个设计从未被
拿去对照同一条原则重新检查，于是 metadata 时间线落后 ladder 时间线整整一个
Phase——Ward 合并看到的 ladder 是对的，看到的 metadata 却是错的，两条本该
同步的时间线出现了一个此前没人注意到的裂缝。

#### 修法：Phase 3 拆成 3a（紧跟 Phase 1，向量化）和 3b（内联进 Phase 2，逐 token），不再有一个独立的"批末"步骤

**决定**：把"Phase 3"从"Phase 1、Phase 2 都结束后才运行的第三个批处理步骤"，
改成按**内容来源**拆开、各自在能拆的最早时刻执行，不再存在任何一个统一的、
推迟到批末的入口：

- **Phase 3a**（向量化，紧跟 Phase 1 完成之后、Phase 2 开始之前）：只处理
  Phase 1 产出的主操作（direct token 的 `JOIN`/`NEW_SEGMENT`——direct token
  不会产生 `NEW_CLUSTER`，那是 orphan 专属），用 §5.5 的公式批量更新它们各自
  目标簇的 `centroid`/`n_eff`/`n_total`/`p_hi_c`/`current_segment`。这正是
  前面"这个技巧的适用范围不止这里"那个指针指向的向量化任务——但含
  `NEW_SEGMENT` 触发的 `γ` 衰减重启时，`n_eff`/`centroid` 需要的是一个
  仿射变换复合的并行扫描，**不是**和 segment id/pad count 同构的
  `steps_since` 式重置扫描（上方"更正"框已说明两者的"重置"不是同一类
  操作），完整推导见下方"Phase 3a 的具体做法"——这里第一次把它从"留给
  以后"变成"必须在这里执行"，因为现在明确了它必须在 Phase 2 开始前完成，
  不能再推迟。
- **Phase 3b**（内联，逐 token，不是批处理）：Phase 2 本身已经是串行循环
  （§11-B 的摊还论证只要求它的**总执行次数**被 `E[K]` 界住，从未要求它的
  metadata 更新也批量化）。每处理完一个 orphan（组）的主操作、把它 append
  进本地缓冲之后，**立即**（不等其它 orphan、不等 Phase 2 循环结束）用同一套
  §5.5 公式更新它目标槽的 metadata。这不需要任何新的向量化技巧——Phase 2
  本来就在做"处理一个 orphan、写它的 op"这件事，Phase 3b 只是把"写完 op
  之后顺手更新这个槽的 metadata"接到同一次循环迭代里，没有增加一次额外的
  遍历。

**这个拆分让"Ward 合并读到 stale metadata"这件事结构性不可能发生**：Ward
合并只会在 Phase 2 内部触发。到 Phase 2 开始时，Phase 3a 已经让**所有**簇
（不论本批是否被 Phase 1 触碰过）的 metadata 反映了 Phase 1 的全部贡献；
Phase 2 内部，每个 orphan（组）处理完自己的主操作后立即执行 Phase 3b，所以
**当序列中稍晚的一次 Ward 合并触发时**，任何可能成为其候选（`keep_slot` 或
`free_slot`）的槽——无论是 Phase 1 触碰过的（Phase 3a 已更新）、还是本次
Phase 2 循环中更早被某个 orphan 触碰过的（那次迭代的 Phase 3b 已更新）——它
的 `n_total`/`μ`/`p_hi_c`/`current_segment` 都已经是本批目前为止的真实值，
不存在"合并腾位时还没被这批的贡献更新过"的槽。反过来，`free_slot` 被释放
复用给新簇之后，新簇自己的主操作也在同一次内联步骤里被立刻计入自己（新的、
白纸状态的）metadata，不会和已经被合并走的旧内容混在一起——因为**混淆的
根源（用裸 `op.cluster` 在批末回头分组）已经不存在了：每条主操作在它被追加
的那一刻就被消费掉，不会留到"以后"用一个可能已经改变含义的槽号去重新解释**。

**这不是重新引入"批内重定向"**——本节前面推翻重定向的理由是"改写已经写入的
op 字段"会破坏 op_log 的执行日志性质；Phase 3a/3b 不改写任何 op 字段，
`op_log` 依然是纯追加、永不改写的日志，改变的只是**消费这些 op 去更新
metadata 的时机**，这是一个纯粹的调度问题，不触及本节已经证明成立的"op_log
顺序契约"三条规则。

**对 §11-B 摊还论证没有影响**：Phase 3a 是向量化的（和 segment/pad 的向量化
同一量级开销）；Phase 3b 内联进 Phase 2，Phase 2 的总执行次数仍然被 `E[K]`
界住，加一次 metadata 更新不改变这个界，只是把常数因子略微调大。

**对 §5.4 开头"这是一个近似"的 S0.8 待验证近似没有影响**：Phase 1 判断"谁去
哪个簇"仍然使用批前冻结的 centroid/`p_hi_c`——这个近似的性质完全不变，S0.8
该测的还是测（拆成三项，见 `experiments.md` §6 的 S0.8 决策门）。这次修的是
**判断做出之后，结果什么时候被写回 metadata**，是一个正确性问题（会不会读到
stale 值），不是近似精度问题（近似应该多准），两者正交。

> **上面"顺序上 Ward 合并总是先于 Phase 3（它发生在 Phase 2 内部），所以
> Phase 3 读到的永远是『本批全部结构变动都已经落地之后』的起点"这句话是错
> 的，必须收回。** 它假设的前提是"Phase 3 只会在 Ward 合并全部结束后才运行
> 一次"——这正是上面刚刚推翻的旧设计。**正确的表述**：`ward_merge_only` 步骤
> 1（结构性合并，处理两个既有簇已积累的历史值）和 Phase 3a/3b（内容性更新，
> 处理本批新到达 token 的贡献）写入范围确实不重叠，但让它们不冲突的不是
> "Ward 总在 Phase 3 之前"，而是**Phase 3a/3b 的调度保证了：任何一次
> `ward_merge_only` 执行时，它将要读的两个槽的 metadata 都已经是本批目前
> 为止的最终值**——这是一个关于*调度*的保证，不是关于*两次事件谁先谁后*的
> 巧合。

**Phase 3a 的具体做法**：Phase 1 结束、Phase 2 开始之前，对本地缓冲里刚刚由
Phase 1 写入的主操作（只有 `JOIN`/`NEW_SEGMENT`），逐簇更新
`centroid`/`n_eff`/`n_total`/`p_hi_c`/`current_segment`（`current_segment`
只在遇到 `NEW_SEGMENT` 时被那条 op 的 `segment` 字段覆写，`JOIN` 不改它）。
**这里同样要用 `op.token_idx` 去取这条 op 对应的 `k_raw`，不能假设"本地
缓冲里的第几条 op 就对应本批第几个 token"**——原因和 backward 重放的
`token_ptr` 问题完全一样（§5.4"op_log 跨 Phase 的顺序契约"一节）。

**五个字段里，三个是平凡的分组归约（reduce），不需要扫描（scan）**，直接
复用 §5.4"Segment id"/"PAD_INSERT 的 count"两节已经定义好的
`c*[t]`（`t=0..m_direct-1`，direct 子序列下标）、`rank[t]`（组内到达序，
0-indexed）、`nsg_incl[t]`（组内 `new_seg` 的包含性前缀和）：

```
n_total_new[c] = n_total_old[c] + count(t : c*[t] == c)      # 分组计数，
                                                                # 不依赖顺序
last[c]        = argmax_{t : c*[t]==c} rank[t]                 # 组内到达
                                                                # 最晚的 t
p_hi_new[c]        = p_{last[c]}                                # p_hi 的定义
                                                                 # 就是"最近
                                                                 # 一次收到
                                                                 # 成员的位置"
current_segment_new[c] = op.segment[last[c]]                    # 同一个 t，
                                                                 # 直接读它已经
                                                                 # 算好的
                                                                 # segment 字段
```

三者都只需要"这一组最后一个成员是谁"，`argmax`/`gather` 是标准原语，不需要
中间任何一步的值。

**`n_eff`/`centroid` 是唯一需要真正扫描（scan）的两个字段——这是 Phase 3a
里唯一困难的部分**，值得把 §5.5 的逐成员递推摊开重新过一遍，给出闭式的
向量化方案（上方"更正"框已经说明了为什么不能照搬 `PAD_INSERT` 的
`steps_since` 技巧）。记组内到达序为 `r = rank[t]`，`n_eff_{-1} :=
n_eff_old[c]`、`μ_{-1} := centroid_old[c]` 是批前持久值，§5.5 的递推是：

```
d[t]    = γ  若 new_seg[t]，否则 1                # 衰减因子，标量，∈[0,1]——
                                                    # γ=0 是 §5.5 明确支持的
                                                    # 合法配置（"归零"分支），
                                                    # 不是要排除的边界
n_eff_r = d[t]·n_eff_{r-1} + 1
μ_r     = (d[t]·n_eff_{r-1}·μ_{r-1} + k_t) / n_eff_r
```

**这是一串仿射变换（affine map）的复合**：每个成员把 `(n_eff, μ)` 映射成
`x ↦ d[t]·x + 常数项` 的形式，仿射变换在复合下满足结合律——这正是它可以
被压成一次并行扫描的原因，和 `_binary_carry`（§5.21-3）能被压成固定轮数的
掩码 carry 是同一类"用结合律换并行"，只是这里的幺半群（monoid）是仿射
复合，不是二进制进位。

**做法**：给每个成员一个仿射状态三元组 `(A, Bn, S)`（`A`、`Bn` 是标量，`S`
和 `k`/`μ` 同维），代表"从这一段开始到当前位置为止，`n_eff` 和
`n_eff·μ` 各自被复合成什么仿射变换"：

```
成员 t 自己的状态（scan 的叶子/base case）：
    A_t  = d[t]
    Bn_t = 1
    S_t  = k_t.float()   # 见下方 dtype 说明：显式转 fp32，不沿用 k_raw 的
                           # activation dtype

合并两段状态（"先" ⊕ "后"，把"后"接在"先"之后，标准仿射复合，结合律成立）：
    A   = A_后 · A_先
    Bn  = A_后 · Bn_先 + Bn_后
    S   = A_后 · S_先  + S_后
```

> **`(A, Bn, S)` 全程必须是 fp32，`S_t` 的输入 `k_t` 必须显式转型，不能
> 沿用 `k_raw` 的 activation dtype。** `k_raw` 在 dump（`experiments.md`
> 的 dump 表）和生产路径里都以 fp16/bf16 存储/传递，但 §5.13 的
> `centroid`/`n_eff` buffer 明确声明 fp32——如果实现者顺手让 `S_t = k_t`
> 直接沿用 `k_t` 自己的低精度 dtype（这是最容易踩的坑：`S` 是逐元素对
> `k` 做的运算，"跟着输入 dtype 走"是最不需要多想的默认写法），累加器
> `S`/最终的 `centroid_new` 就会被低精度污染。这比一次性的 fp16 计算更
> 危险：`centroid` 是**跨整条序列在线更新**的量（每个 flush 批都要把
> 这批的贡献混合进去，一条 32k 序列有几百次这样的批次），每次都在 fp16
> 精度上算再写回 fp32 buffer，量化噪声会跨批次反复注入、累积，不是
> 单次可忽略的舍入——这也是为什么它需要在这里单独强调，而不是只在
> §5.13 的 `w` dtype 警告（`log_kv_cache.py` 现有 `level_w` 沿用
> activation dtype 那个坑）旁边顺带一提就够了，两处坑的后果都是"精度
> 静默劣化"，但这里是**持续累积**的版本，更隐蔽。`A`/`Bn` 本身不直接
> 参与和 `k`/`μ` 的运算（`Bn` 只是标量计数，`A` 只是 `γ`/`1` 的乘积），
> 但既然要和 `S`/`centroid`/`n_eff` 做加乘，也一并声明成 fp32，不留
> 隐式类型提升的歧义。最终写回 `centroid`/`n_eff` buffer 时两者 dtype
> 已经一致，不需要额外转换。

按 `c*[t]` 分组做一次**包含性（inclusive）前缀扫描**（分组/排序用的是和
`nsg_incl` 完全相同的 `(c*[t], rank[t])`，只是归约算子换成上面的仿射
复合，不是求和或求最大——**这不是现成的库原语**，不同于前面 segment
id/PAD count 依赖的原生 segmented cumsum/cummax，这里需要手写一个
Hillis-Steele 式的倍增扫描：`⌈log₂ flush_granularity⌉` 轮，**静态轮数，
和 §5.21-3 同一个纪律**——第 `s` 轮里每个位置和"组内比它早 `2^s` 步的
位置"合并，跨组或越界的位置用幺元 `(1, 0, 向量 0)` 代替）。扫描结束后，
每个成员 `t` 手里的 `(A_r, Bn_r, S_r)` 就是"从组内 rank 0 到 `rank[t]`
（含）"的复合仿射变换，取组内最后一个成员（上面已经算出的 `last[c]`）
即得整簇这一批的最终变换：

```
n_eff_new[c]    = A_{last[c]} · n_eff_old[c] + Bn_{last[c]}
centroid_new[c] = (A_{last[c]} · n_eff_old[c] · centroid_old[c] + S_{last[c]})
                  / n_eff_new[c]
```

（可以对一个 2 成员的组手动展开验证：`n_eff_r1 = d1·(d0·n_eff_{-1}+1)+1
= (d1·d0)·n_eff_{-1} + (d1·1+1)`，和 `A=d1·d0`、`Bn=A_后·Bn_先+Bn_后
=d1·1+1` 逐项相符；`S` 同理展开也一致。）

**为什么不直接写闭式解，要用扫描**：把上面的递推展开成闭式确实存在
（`A_r=γ^{nsg_incl[t]}`，`Bn_r`/`S_r` 展开后会出现形如 `γ^{-nsg_incl[j]}`
的负指数项），但**负指数在这里是真实的数值风险，不是理论洁癖**——一个
病态批次（`flush_granularity=128`，同一个簇的全部 128 个 direct token
每个都触发 `NEW_SEGMENT`，即 `g_max` 设得极小）在默认 `γ=0.5` 下会要求
算出 `γ^{-128} = 2^128 ≈ 3.4×10^38`，逼近 fp32 的溢出边界；`γ=0`（§5.5
明确支持的合法配置，"归零"分支）下这个闭式解直接是 `0^{-k}`，未定义，
连"数值上有风险"都算不上，是彻底不能用。上面的扫描完全避开这两个问题——
`A` 全程是若干个 `∈[0,1]` 的数相乘（**包含 0**：`γ=0` 时只要这一组内出现
过一次 `NEW_SEGMENT`，`A` 就会精确变成 `0`，这是符合语义的正确结果——
表示这次衰减彻底遗忘了衰减之前的全部历史，`n_eff_new`/`centroid_new` 的
公式在 `A=0` 时分别精确退化成 `Bn_last`/`S_last/n_eff_new`，即"只看
衰减之后的成员"，和 §5.5"`γ=0` 或冷启动：没有历史可混，直接替换"这条
分支的语义完全一致，**不是需要用 `assert A>0`、log 域运算或取倒数之类
写法规避的数值边界**），只会变小或不变，不会变大；`Bn`/`S` 每一步合并
都只用**当前已经算出的、有界的**中间结果做加乘，从不出现负指数或除以
一个可能趋近于 0 的量，是数值稳定的标准做法（需要算"衰减累加"的
实现——比如强化学习里的 GAE——回避闭式解、改用扫描，是同一个理由）。
**必须覆盖 `γ=0` 且本批内同一簇连续多次 `NEW_SEGMENT` 的合成场景**，
作为 §5.4"必须补的单测"第 3 条（Phase 3a/3b 元数据更新的正确性）对拍
数据的一部分——这是最容易让"`A` 恒为正"这类不成立的隐含假设（例如实现
里悄悄写了 `assert A > 0` 或改用 `exp(cumsum(log(d)))` 这类在 `A=0` 处
未定义的写法）现出原形的输入。

**顺序要求**：不同簇之间彼此独立，可以任意顺序或并行处理；同一个簇的扫描
必须尊重组内到达顺序（`rank[t]`），这靠分组排序后再扫描保证——和 §5.21-2
对批量 ladder 写入提的"保序"要求是同一类约束，同一个理由（"结果只依赖同簇
成员的相对到达顺序，不依赖处理粒度"）。

**正确性验证复用已有的测试，不需要新增一条**：§5.4"必须补的单测"第 3 条
（Phase 3a/3b 元数据更新的正确性）已经要求生产路径的向量化结果和一个逐
token 串行、按 §5.5 未向量化原始递推逐步执行的朴素参考实现对拍——**按
该条给出的比较口径**（整数字段 `n_total`/`p_hi_c`/`current_segment` 逐位
精确，浮点字段 `centroid`/`n_eff` 数值容差内一致，不是笼统的"逐位"，理由
和具体容差见该条），上面给出的三个平凡归约和这条仿射扫描，就是"生产路径"
这一侧具体做的事，这条既有的对拍天然覆盖它们对不对，不需要单独再定义
一套断言。

**Phase 3b 的具体做法**：Phase 2 每处理完一个 orphan（组）的一条主操作
（`NEW_CLUSTER`/`JOIN`/`NEW_SEGMENT`）、把它 append 进本地缓冲之后，立即用
同一条 §5.5 公式更新 `op.cluster` 这个槽此刻的 metadata。**新建簇的第一个
成员不需要特判**——`allocate_new_cluster` 已经把它的 `n_eff` 清零，§5.5 公式
里 `n_eff_pre == 0 时 μ_c ← k` 这条分支（本就是为"冷启动或 γ=0 归零"设计的）
会自动接住"这是一个刚建立、从未有过成员的簇"这个情形，和"γ=0 导致的段内归零
重启"是同一段代码，不需要为"这个簇是不是本批新建的"专门分叉。因为是逐 token
立即执行，"顺序要求"在这里自动满足，不需要像 3a 那样借助向量化 scan 技巧。

**执行时机（更正：不再是"Phase 1、Phase 2 都结束后才运行"）**：Phase 3a 在
Phase 1 之后、Phase 2 之前运行一次（向量化）；Phase 3b 内联在 Phase 2 循环
内部，随每个 orphan 主操作产生而立即执行，不是一个独立的、推迟到批末的步骤。
**更早一轮"Phase 3 必须在本批 Phase 1 和 Phase 2 全部处理完之后才运行"这句话
是错的，必须收回**——那句话隐含假设"Phase 3 只处理内容更新、不会被 Phase 2
内部的 Ward 合并需要"，但 Ward 合并的代价矩阵和结构性合并本身就需要读当时
最新的 `n_total`/`μ`/`p_hi_c`/`current_segment`，推迟到批末运行恰恰是本节
这一轮要修的 bug 的根源。

**必须补的单测，拆成三条不同性质的断言，不能合并**：

1. **重放正确性（S0.1，必须逐位精确，纯 CPU 可测，只覆盖 attention 需要的
   张量，不覆盖 centroid 类元数据——理由见下）**：构造一个批内既有 Phase 1
   大量分配、又有 Phase 2 orphan 触发 Ward 合并、且合并候选恰好选中**本批内
   刚建立的簇**（就是上面反例描述的场景，S0.1 必须专门覆盖这一类输入，不能
   只测"合并候选是批前就存在的旧簇"这种更简单的情形）的合成场景，断言
   （a）`op_log`（本地缓冲与提交进持久存储后）里任何已写入条目的任何字段都
   没有被后续操作改写——这条本身就是回归测试，直接对拍"写入前"和"整批处理
   完之后"的逐字节快照；（b）同一个新槽如果被组内多个 orphan 共享，恰好一条
   `NEW_CLUSTER` + 若干条 `JOIN`/`NEW_SEGMENT`，不多不少；（c）**按标准顺序
   重放算法执行整批提交进持久 `op_log` 的条目，产出的 ladder 张量
   （`k̄/v̄/w/p_lo/p_hi/sum_wp/σu/σ2/γa/γb/γ`）、`level_count`、`alive`、
   `pad_mask` 与"直接执行本批的批量向量化写入"这两条路径逐位一致**——这条
   测的是"`op_log` 忠实记录了批量前向实际做了什么"，不是"批量前向的路由
   决策对不对"。**`centroid`/`n_eff`/`n_total`/`p_hi_c`/`current_segment` 不在
   这条断言的范围内**——§5.21-2 已经明确"重放完全不需要 centroid"，上面这个
   标准重放算法
   从不触碰、也不重建这些字段，把它们纳入"重放正确性"断言等于要求一个按
   设计就不做这件事的算法去做这件事，断言本身就是错的。如果需要验证
   Phase 3 的元数据更新对不对，那是下面第 3 条的职责，两条不能合并。
2. **批量近似质量（S0.8，允许有分歧率，不要求逐位一致）**：本场景（Ward
   合并选中批内刚建立的簇）作为 S0.8 分歧率统计里**必须覆盖的一类特殊
   输入**，衡量"批量路由的最终路由决策"和"严格串行、每处理一个 token 就
   重算一次 centroid/`p_hi_c`/`alive` 的基准路由"之间差多少。**这条不要求、
   也不应该要求逐位一致**——Phase 1 冻结 centroid 和 `p_hi_c` 本来就是一个
   已经承认、已经在测的近似（§5.4 开头"这是一个近似"那段），Ward 合并只是
   这个近似在"K 已满、批内密集触发新簇"这种极端情形下的一个特例，不应该单独
   拔高到"必须和严格串行版逐位相同"这个不属于它的正确性等级——那样会把
   "重放准不准"和"批量决策的近似质量"这两个不同维度的问题混成一个。
3. **Phase 3a/3b 元数据更新的正确性（S0.1，整数字段逐位精确、浮点字段
   数值容差内一致——理由见下方，与第 1 条完全独立的一条断言，不要和
   "重放正确性"共用同一个测试）**：用同一批合成数据，分别用
   (a) 生产路径（Phase 1 → Phase 3a → Phase 2-含内联 Phase 3b 完整跑一遍）
   得到的最终 `centroid`/`n_eff`/`n_total`/`p_hi_c`/`current_segment`，和
   (b) 一个独立的、逐条重放的参考实现，比较两者（比较口径见下方，不能
   笼统要求全部字段逐位一致）。

   > **更正（这一轮修的）：参考实现只喂"主操作序列"是错的，必须喂完整 op
   > 序列（含 `WARD_MERGE`）。** 上一版让参考实现只吃"本批最终本地缓冲里
   > 原样的主操作序列"——但 `WARD_MERGE` 是结构操作，不在"主操作"之列，
   > 如果参考实现真的看不到它，下面 (i)/(ii) 两条断言根本无法成立：参考
   > 实现既无法把被合并簇（`free_slot`）批内已经收到的贡献并入
   > `keep_slot`（(i) 断言的正是这份贡献有没有被正确合并），也无法在
   > 复用槽位建新簇之前清空旧内容（(ii) 断言的正是新簇有没有干净地不
   > 包含旧内容）——而验证这两条正是这个测试项存在的全部意义（见上面
   > "未闭合的漏洞"一节的反例）。**正确做法**：参考实现按物理顺序走
   > **完整**的本地缓冲（主操作 + `WARD_MERGE`，`PAD_INSERT`/`CARRY`
   > 对这五个字段没有任何影响，原样跳过），处理模式和 backward 重放循环
   > （§5.21-2）完全同构，只是把 `append_to_ladder` 换成对 metadata 的
   > 操作：遇到主操作（`NEW_CLUSTER`/`JOIN`/`NEW_SEGMENT`）按 §5.5 公式
   > 更新 `op.cluster` 这个槽的 scratch 元数据（`NEW_CLUSTER` 先把该槽
   > scratch 清零，是"白纸"状态，再应用 §5.5，和 `allocate_new_cluster`
   > 的语义一致）；遇到 `WARD_MERGE(keep,free)` 对 scratch 状态套用和
   > §5.6"K_max 满时的完整合并过程"步骤 1 完全相同的合并公式
   > （`μ_new=(n_eff_a·μ_a+n_eff_b·μ_b)/(n_eff_a+n_eff_b)`、
   > `n_total_new=n_total_a+n_total_b`、`p_hi_new=max(p_hi_a,p_hi_b)`、
   > `current_segment_new=max(...)`），再把 `free` 的 scratch 清零。
   > **不需要引入 `derive_final_cluster` 那套 `(slot,epoch)` 版本化
   > 身份**——那套机制是为了让一个跳过时间顺序、只看 `WARD_MERGE` 边的
   > 离线并查集派生保持正确；这里是一次真正按时间顺序逐条执行的模拟，
   > "遇到 `NEW_CLUSTER` 就清零 scratch"已经天然处理了槽位复用，不会
   > 把"旧 X"和复用同一物理槽的"新 X"混成一个身份。

   > **更正（这一轮修的）：上面这套参考实现隐含假设了一个从空白开始的
   > 单批场景，对"alive 但本批未被任何主操作命中的槽"没有定义该从哪里
   > 起算，这个空白本身就是一类 bug 的来源，必须补上。** 真实的 forward
   > 是很多个 flush 批连续处理同一条序列，第 `N` 批开始时，绝大多数已经
   > `alive` 的簇既不是"这一批刚 `NEW_CLUSTER`"，也不是"这一批被
   > `JOIN`/`NEW_SEGMENT` 命中"——它们只是在更早的批次里建立、此后一直
   > 存在，这一批可能完全没有 token 路由到它们。这类槽的正确行为是
   > **metadata 原样不变，等于进入本批之前的值**（该值一般不是 0——它是
   > 这个簇迄今为止累积的真实内容），不是"0"，也不是"未定义"。上面的
   > 参考实现描述本身没有错（`NEW_CLUSTER` 清零、`JOIN`/`NEW_SEGMENT` 走
   > §5.5 公式），但它默认 scratch 状态从哪里起算从未交代——如果测试只
   > 构造单个孤立批次、且隐式让 scratch 起点全局为空，这套参考实现就只能
   > 覆盖"整条序列的第一批"，第二批及以后任何"alive 但本批未被命中"或
   > "本批 `JOIN` 进一个更早批次建立的簇"的情形都测不到，而这恰恰是长
   > 序列下最常见的情形，不是边界情形。
   >
   > **修法**：参考实现（以及和它对拍的生产路径调用）都必须接受一个
   > 显式的 `initial_scratch: dict[int, Metadata]` 种子参数（`Metadata`
   > 打包 `centroid`/`n_eff`/`n_total`/`p_hi_c`/`current_segment`），代表
   > "进入本批之前"的完整状态，由测试构造者显式提供（可以全零，代表冷
   > 启动；也可以非零，代表"已经跑过若干批"）。参考实现里，任何主操作
   > 引用的槽号若不在 `initial_scratch` 里、也没有在**这次调用看到的 op
   > 序列内**先出现过一次 `NEW_CLUSTER`，是测试构造本身的 bug（引用了一个
   > 既非新建、又未声明历史状态的槽），应该直接断言失败，不是需要静默
   > 兜底的情形。**生产路径这一侧不需要任何对应改动**——它读写的
   > `centroid`/`n_eff`/`n_total`/`p_hi_c`/`current_segment` 本来就是跨批
   > 持久的 buffer（§5.13），Phase 3a/3b 天然只更新本批 op 引用到的槽，
   > 未被引用的槽内容自动原样保留；这条更正纯粹是补全**参考实现和测试
   > 构造方法**的规格，不涉及任何生产代码行为的改变。

   **比较口径必须按字段类型拆开，不能笼统要求"逐位一致"**：`n_total`/
   `p_hi_c`/`current_segment` 是纯整数字段（分组计数、gather、取 `max`，
   过程里不出现任何浮点运算），(a)(b) 两条路径必须**逐位精确相同**——
   不同就是逻辑 bug，没有"数值噪声"这个借口。`centroid`/`n_eff` 是浮点
   字段（fp32），(a) 走的是 §5.5 递推的一个**并行仿射扫描**（"Phase 3a
   的具体做法"一节），(b) 走的是同一个递推的**朴素顺序执行**——两者数学
   上算的是同一个量，但浮点加法/乘法不满足结合律，并行扫描重新组合运算
   顺序之后，IEEE754 舍入误差可能在最后几位上和顺序执行不同，这不是
   bug，是任何"并行 scan vs 顺序循环"的浮点比较都会遇到的标准情形（GPU
   并行规约和 CPU 顺序求和即使数学等价也不逐位相同，是同一个原因）。
   **要求 `centroid`/`n_eff` 逐位相同，等价于要求参考实现去复刻生产路径
   完全相同的运算结合顺序**——真这么做的话，两个实现会变成同一段扫描
   逻辑的两份拷贝，一份的 bug（比如某一轮掩码写错）会在另一份里被逐位
   复现，参考实现"独立交叉验证"这件事就名存实亡，比不设容差更糟。
   **决定**：`centroid`/`n_eff` 改用数值容差，且必须是"绝对+相对"的
   组合公式（`torch.testing.assert_close(actual, expected, rtol=1e-4,
   atol=1e-5)`，或等价的 `|a−b| ≤ atol + rtol·|b|`），**不能只给
   `rtol`**——纯相对误差在参考值 `b=0` 处除零/未定义，而 `n_eff`/
   `centroid` 在**从未被分配过的死槽**上恰好精确是 0，原因不变
   （`reset_parameters()` 的显式 `zeros(...)`）。**但"alive 且本批未被
   任何主操作命中的槽也是 0"这个说法是错的，必须收回**——这类槽（上一批
   甚至更早批次里建立、此后一直存活但这一批没有 token 路由到它）的正确
   值是**进入本批之前的旧值**，一般不是 0；`atol` 保护的是"参考值恰好
   是 0（或接近 0）"这类条目在纯相对误差公式下的除零/未定义，死槽是这类
   条目里唯一能保证恰好是 0 的一类，alive-but-untouched 槽的参考值可以是
   任意非零数，不属于这个理由覆盖的范围，但下面的组合公式对它们同样
   成立。`rtol=1e-4` 量级上远大于 fp32 单精度在 `flush_granularity≤128`、
   `⌈log₂128⌉=7` 轮扫描下能积累的舍入误差，两者组合后足够收紧到能抓住
   真正的逻辑 bug：双计数、遗漏贡献这类问题造成的偏差通常是量级上的，
   不是最后几位的舍入噪声或一个趋近于零的参考值造成的虚假告警。
   **dead/未被触碰的槽不需要从比较范围里单独摘除，但理由不是"两边都是
   0"，而是"两边都源自同一份 clone"**：只要 (a) 生产路径和 (b) 参考
   实现都用上面更正框要求的 `initial_scratch` 种子做起点，一个未被本批
   任何 op 触碰的槽在两条路径下的值**逐位等于同一个 `initial_scratch`
   条目**——死槽的 `initial_scratch` 条目恰好是 0（由 `reset_parameters()`
   保证），alive-but-untouched 槽的 `initial_scratch` 条目是那个簇的真实
   历史值，不必是 0——两种情况下 (a)(b) 两侧都相等，用上面"绝对+相对"的
   组合公式比较整个 `(K_max, d)` buffer 依然天然通过（值相等时
   `|a−b|=0 ≤ atol` 对任意 `atol>0` 恒成立），不需要按"是不是被触碰过"
   分情况摘除或另外处理；`n_total`/`p_hi_c`/`current_segment` 继续要求
   逐位精确（这三个字段不存在浮点误差，不存在"参考值为 0 时公式退化"
   这个问题，对 alive-but-untouched 槽的旧值同样逐位精确相等）。

   断言：整数字段逐位精确、`centroid`/`n_eff` 在上述容差内一致，且
   **新建簇（本批内 Ward 合并腾出槽位后建立的那些）的最终 `n_total`
   精确等于它实际收到的 token 数，不多不少**——这条直接抓住
   "Phase 2 初始化和 Phase 3 更新双计数"这类 bug：如果双计数复现，这里的
   `n_total` 会系统性偏大。**这批合成数据必须专门覆盖上面"未闭合的漏洞"那节
   的反例场景**（Phase 1 已经对某个既有簇写下主操作、Phase 2 随后的 Ward
   合并恰好选中这个簇作为 `free_slot`）：断言 (i) 被合并进 `keep_slot` 的
   `n_total`（逐位精确）/`μ`（容差内）包含了 Phase 1 那次主操作的贡献
   （不是批前快照）；
   (ii) 复用该槽位建立的新簇的 `n_total`/`μ` **不包含**已经被合并走的旧内容
   的任何贡献——这两条合起来直接抓住"Ward 合并读到 stale metadata"这类 bug，
   如果时序修复回归，(i) 会因为漏掉一份贡献而偏小，(ii) 会因为被错误混入而
   偏大或偏离预期 centroid，两个方向都要测，只测一个方向可能被另一个方向的
   巧合掩盖。**这批合成数据还必须专门覆盖上面"参考实现隐含从空白开始"那节
   的反例场景**：至少构造两个连续批次，第一批让某个簇 `X` 建立并累积真实、
   非零的 `centroid`/`n_eff`/`n_total`，第二批完全不包含任何引用 `X` 的主
   操作；断言 (iii) 处理完第二批之后，`X` 的全部五个字段在 (a)(b) 两条路径
   下都精确等于（浮点字段：在同一组合容差内等于）它在第一批结束时的值，
   一字不差——这条直接抓住"实现把 alive-but-untouched 的槽误当成需要清零/
   重新初始化"这类 bug，只用单批次合成数据结构性地测不出它，因为单批次里
   "这一批建立"和"更早批次建立、这一批只是恰好没被命中"这两类槽根本无法
   区分。**这条断言存在的意义是把"重放/回放需要什么"和"调试工具/S0.8
   对拍想知道什么"彻底分开成两个独立契约**——前者只服务 backward 的梯度
   正确性，后者只服务分析工具，任何时候都不应该被混进同一个正确性等级里。

### 5.5 分簇：centroid 更新

段内话题稳定 ⇒ 计数均值是对的；跨段可能已漂移 ⇒ 新成员该占更大权重：

```
开新段时（在这一段第一个成员到达之前执行一次）:
    n_eff_pre ← γ · n_eff                     # γ=1 保留全部历史权重，γ=0 归零

收到成员 k（段内的每一个成员，包括刚开新段后的第一个）:
    n_eff_pre ← n_eff                          # 沿用当前值（新段的第一次是上面刚写的衰减值）
    n_eff     ← n_eff_pre + 1
    if n_eff_pre == 0:
        μ_c ← k                                 # γ=0 或冷启动：没有历史可混，直接替换
    else:
        μ_c ← (n_eff_pre · μ_c + k) / n_eff      # 等价于 μ_c + (k − μ_c) / n_eff，用新计数做分母
```

这是 §2.4 里说的"零副作用的门控更新"——centroid 只是路由元数据，不参与 attention
读出，擦它不丢任何信息。注意 **`n_eff` 因此必须是浮点而非整数**。

> **一处曾经写错的公式**：早期版本写的是"`μ_c ← μ_c + (k − μ_c) / n_eff`，
> 然后 `n_eff ← n_eff + 1`"——字面实现有两个问题。**Off-by-one**：标准在线均值第
> `n` 个样本要除以**新**计数 `n`，不是旧计数 `n−1`；把"先用旧 `n_eff` 算 μ、再
> 递增"读成两条独立语句，除数错了一位，`n_eff` 越界越大越不明显（相对误差
> `O(1/n)`），小簇（正是 needle 最关心的那类）反而误差最大。**除零**：`γ=0` 时
> `n_eff_pre = 0`，新段第一个成员按原公式要除以"递增前的 `n_eff`"即 `0`，
> `(k−μ_c)/0` 直接炸。上面改写后 `n_eff_pre==0` 时走 `μ_c ← k` 这条专门分支，
> 既避免除零，又精确对应 `γ=0`"直接替换"这个已经写在注释里但从未在公式里兑现的
> 语义。`n_eff_pre==0` 只可能在 `γ=0` 或该簇刚建立、从未有过成员时出现，两种情形
> 都应该是"没有历史可混"，直接替换是唯一自洽的选择，不是特判，是补上被遗漏的
> 边界情况。

> **更正（曾经写错）**：早期版本只维护一个 `n_c`，同时拿它喂 centroid 的在线均值
> **和** §5.6 Ward 合并代价里的 `n_a, n_b`。**这是两件不同的事,用同一个量会互相
> 污染**：centroid 更新要的是"新成员该多大程度上拉动 centroid"，`γ` 衰减故意让
> 历史久远的段贡献变小,这是对的;但 Ward 代价 `(n_a n_b)/(n_a+n_b)·‖μ_a-μ_b‖²`
> 要近似的是"合并这两个簇会让簇内平方和增加多少",这个量该反映**真实物理规模**
> （簇里实际存了多少 token/entry），不该被 `γ` 衰减。用被衰减过的 `n_c` 算 Ward
> 代价，会让一个**跨越多个段、历史很长、真实内容很多**的簇因为 `γ` 反复衰减而显得
> "小"，被 Ward 误判成廉价可合并——恰恰挑中了最不该被廉价合并的那类簇。
>
> **拆成两个独立量**：
> - **`n_eff`**：上面这条公式,只喂 centroid 更新,`γ` 衰减。
> - **`n_total`**：单调递增,**从不衰减**,每次该簇收到一个 primary 操作
>   （`NEW_CLUSTER`/`JOIN`/`NEW_SEGMENT`，见 §5.21-2）就 `+1`——它精确等于这个簇
>   从建立以来收到过的**真实 token 数**（等价于对该簇 ladder 里所有 entry 的
>   `w` 求和，但维护一个运行计数器比每次现场求和便宜）。**Ward 代价改用
>   `n_total_a, n_total_b`**，§5.6 的合并过程相应更新：`n_total_new = n_total_a
>   + n_total_b`（简单加法，单调不减，和 `n_eff_new = n_eff_a + n_eff_b`
>   分开维护——后者继续喂 centroid 混合，前者只喂 Ward 代价）。
>
> **`n_total` 还有一处不能兼任**：§5.11 的 `PAD_INSERT` 对齐计算也不能用
> `n_total_c`（会在第一次填充后算错，反例见 §5.11 那条更正框），但那处的修法
> **不是**再拆一个新计数器，而是直接复用已经在维护的 `level_count[cluster,0]`
> ——细节见 §5.11，这里只提醒"`n_total` 不能哪里都塞"这个教训又出现了一次，
> 不代表设计里计数器数量还会继续增长。

### 5.6 分簇：新簇的形成条件、`K_max` 的定尺与自适应

#### 新簇形成的条件

唯一的语义条件：`min_c ‖k_x − μ_c‖² > λ_new`。加上三条结构性路径：

| 路径 | 说明 |
|---|---|
| 冷启动 | 第一个 flush 的 token 必然建簇 0；开头若干 token 会快速填出初始簇集（§11-F 的风险来源）|
| K 未满 | 占用 `alive` 为 false 的第一个槽位，然后调用和"K 已满"分支完全相同的 `allocate_new_cluster(slot_idx, group)` 原语初始化（§5.4"`K` 未满/冷启动必须复用同一个 `allocate_new_cluster` 原语"一段）——这两条路径的差别只在"`slot_idx` 从哪来"，初始化步骤不分叉 |
| K 已满 | 先 Ward 合并腾位，再调用同一个 `allocate_new_cluster`。`K_max ≥ 2` 时 Ward 候选池必然非空（§5.6 已证明），**这一步不会失败，因此不需要定义任何 fallback 行为**——见下方更正框 |

**同样重要的是哪些情况「不」建新簇**：时序被打断 → 开新 **segment**，不开新 cluster
（v3.1 的核心修正）；位置远 → 与簇身份无关，只进 join cost 的排序项和 segment 判定。

> **更正（这一轮修的）：上表"合并失败必须有 fallback：强制并入最近簇"和本文档
> 别处已经证明的不变量互相矛盾，必须删掉，不是留一句模糊的话就算数。** §5.6
> 明确写着"`K_max ≥ 2` 时永远存在可合并的一对，所以这一步不会失败"，§5.19-7
> 也只要求"屏蔽对角线和 dead 槽位，否则 `argmin` 会选到自己或空槽"——这两处
> 说的都是**如何让 `argmin` 不选到无效结果**，不是"如果找不到候选该怎么办"，
> 因为后者在 `K_max ≥ 2` 时根本不会发生。**"合并失败"这个前提本身不成立**，
> 硬要给一个不会发生的事件定义 fallback 行为，只会制造一个从未被测试、语义
> 从未被定义的死代码分支（它该写 `JOIN` 还是 `NEW_SEGMENT`？该消费哪个 token？
> 都没有答案，因为这条路径根本不存在于任何合法状态转移里）。`K_max=1` 那条
> 已经有自己独立、明确定义的分支（下面"`K_max=1` 是退化边界"一节），不依赖
> 也不经过这里。
>
> **实现层面的正确处理方式是断言，不是 fallback**：`ward_merge_only` 调用前
> 对 Ward 代价矩阵的 `argmin` 结果加一条 `assert alive[keep_slot] and
> alive[free_slot] and keep_slot != free_slot`——如果这条断言失败，说明前面
> 某处状态维护出了 bug（比如 `alive` 计数和实际槽位状态不同步、屏蔽逻辑写
> 漏了对角线或 dead 槽位），需要去修那个 bug 本身，而不是在这里"兜底"出一个
> 从未定义过语义的分支去掩盖它。

**一个必须记住的结构性限制**：这个设计**只合并、从不分裂**。簇的 centroid 漂移后，
早期成员可能已经离它很远，但没有机制把它们分出去——而且**分裂在物理上不可能**，
因为那些成员早已被 merge 进 entry，无法反解。所以**早期的路由错误是永久的、
不可恢复的**。这与 §11-F 的冷启动风险叠加，是一个复合风险。

#### `K_max` 的**默认值**是 `max_seq_length` 的函数（但可被实验覆盖）

`L_alloc` 已经是从 `N` 推导的量，`K_max` 的默认值没有理由不是。固定 16 在 1M 上下文
下显然不成立（所有话题被硬塞进 16 个语义身份）。默认公式：

```
K_max_default = max( 4, ⌈ c · log₂ N ⌉ )        c = 1（占位值，由 S0.2 定）
```

取上取整而非下取整：`K_max` 绑定会改变算法的性质（§5.6 末尾），宁可给松一点。
`max(4, ·)` 是短上下文的下限保护。`N` 不是 2 的幂时（如 Qwen3 的 `block_size=40960`，
`log₂ ≈ 15.32`）取 16。

> **这是默认值，不是禁止覆盖。** `--log_kv_cluster_k_max` 仍然可以显式设定，§7 消融表
> 的 `K_max ∈ {1,4,16,64}` 和 §6 Stage 2 的示例正是在扫它——**消融的目的就是检验这个
> 默认公式对不对**。早期版本写成"不是自由参数"是措辞错误。
>
> **覆盖时必须连带重算 `L_alloc`**（§5.12 的公式里有 `K_max`）。用默认 `K_max` 算出的
> `L_alloc` 配上被覆盖的 `K_max`，预算算术直接失效——比如 `K_max` 从 15 调到 64 而
> `L_alloc` 不动，总 entry 数会从 1320 变成 5632，把内存对齐的公平性论证打穿。

取 `K_max = c·log₂N` 时，总 entry 数是 `Θ(log²N)`：

| 上下文 | `L_max`（单簇兜底）| `K_max` | `L_alloc` | entry 数 | + recent | 相对 32k |
|---|---:|---:|---:|---:|---:|---:|
| 32k | 13 | 15 | 11 | 1320 | 2344 | 1× |
| 128k | 14 | 17 | 12 | 1632 | 2656 | 1.13× |
| 1M | 17 | 20 | 15 | 2400 | 3424 | 1.46× |

**上下文涨 32 倍，内存涨 1.46 倍。**

**必须是 `max_seq_length` 的函数，不能是每条序列实际长度的函数**——否则同 batch 内
不同样本 `K_max` 不同，张量又 ragged 了。

#### `K_eff(n)` 的三种可能形状，S0.2 判定

32k 下三者数值接近、分不出来；到 1M 差几个数量级：

| 假设 | 出处 | 1M 下的 `K_eff` |
|---|---|---:|
| **饱和到常数** | DP-means 在有界分布上的**填充数**（packing number）| 与 32k 同量级 |
| `α·log n` | CRP 先验 | ~20 |
| `n^d`（d≈0.5）| Heaps 定律 / Pitman-Yor | ~1024 |

第一个假设理论上最站得住：DP-means 用固定半径 `λ_new` 分簇，簇数上界就是 key 空间
在该半径下的填充数——而 key 是有界网络的输出，落在 `R^128` 的有界区域里，所以填充数
是**与 n 无关的常数**，只取决于 `λ_new` 和 key 分布的内在维度。CRP 的 `α log n`
增长来自**先验**（它永远允许开新桌），不是来自几何。真实文本非平稳会推迟饱和，
但不会阻止它。

**方法论后果**：S0.2 必须测 `K_eff(n)` 的**整条曲线**（n 从 1k 扫到 32k），而不是
在 32k 取一个点——要外推到 1M 需要的是增长指数，单点给不了。

**若实测是幂律**，1M 下 `K_max ≈ 1024`，总 entry 数约 13.9 万（仍是 1M 稠密的 13%），
但"对数空间"的卖点没了，论文定位要改成 `O(n^d log n)`，且内存对齐的公平性论证要
重新评估。**这是 S0.2 的决策门。**

#### 从 `K_eff` 到 `c` 的决策规则（此前是空白）

`K_max_default = max(4, ⌈c·log₂N⌉)` 里的 `c=1` 只是占位——S0.2 会测出
**unclipped** `K_eff(n)` 的整条曲线，但"测到了曲线之后怎么定 `c`、什么时候该往上
调、什么时候该整个放弃这条线"此前从未落成规则。分三步：

**第 0 步，先定死 S0.2 测的是哪个 `K_eff`——这不是同一个量的两种叫法，是两个
不同的实验**：

- **纯 DP-means 口径**（不含 `η`/`g_max`/`γ`，只用"`min_c ‖k−μ_c‖² > λ_new` 就
  开新簇"这一条规则，`K_max` 不设上限）：测的是**内容本身的语义多样性**，回答
  "填充数假设站不站得住"这个理论问题（§5.6"第一个假设理论上最站得住"那段），不
  受批量路由的时序 tie-break 影响，是拟合"对数 / 幂律 / 饱和常数"三种曲线形状
  时应该用的口径。
- **生产三路路由口径**（`η`/`g_max`/`γ` 全部按生产默认值打开，只是把 `K_max` 设
  成一个不会绑定的大数，其余和 §5.3/§5.4 的真实路由逻辑完全一致）：测的是**这套
  路由算法实际会产生多少簇**——`η` 的平局裁决会让极少数边界 token 的归属偏离纯
  语义最近簇，进而可能多开或少开簇，这个偏差只有跑真实路由才能测到。

**下面第 1 步的 `c` 必须用生产三路路由口径的 `K_eff` 算，不能用纯 DP-means 口径**
——`K_max` 要size 的是"这套算法实际会尝试开多少簇"，不是"内容理论上有多少语义
多样性"；两者数值接近但不保证相等，混用会让 `c` 定得系统性偏松或偏紧而不自知。
纯 DP-means 口径只喂上面的曲线形状判定（§5.6 决策门），不喂 `c`。

1. **若 S0.2 确认对数形状成立**（曲线在 `α·log n` 拟合下残差显著小于另外两种
   假设）：**不要只用拟合斜率**——如果截距 `β` 较大，"`c = α·ln(2)`"这个只看
   渐近斜率的算法会在 32k/128k 这种曲线还没跑远、截距项仍占相当比重的区间**系统性
   低估** `K_max`，而这恰恰是本方案实际部署的主战场（1M 才是渐近区间，32k/128k
   是现在就要跑分的地方）。改用**观测区间上界法**：

   ```
   c = max_{n ∈ 已测区间} ⌈ (K_eff(n) + margin) / log₂(n) ⌉
   ```

   即取 S0.2 实际扫过的每个 `n`（1k 到 32k），用该点的**实测** `K_eff(n)`（不是
   拟合曲线的外推值）算出"这个点至少需要多大的 `c` 才能覆盖它"，再取所有点里的
   最大值——这保证了 `K_max_default` 在**已验证的整个区间**上都不低估，而不是只
   在 `n→∞` 的极限下渐近正确。`margin` 给一个固定安全余量（比如 S0.2 曲线拟合
   的残差标准差的若干倍），对冲测量噪声。**只有当曲线形状在整个已测区间都干净地
   贴合对数曲线、拟合优度很高时**，`c = α·ln(2)` 这个渐近斜率法才和观测区间上界
   法给出接近的数字；只要贴合不完美或截距不可忽略，观测区间上界法更保守也更
   诚实——它不依赖"外推到 1M 时曲线形状不变"这个额外假设，只依赖"已经测过的区间
   确实够用"这个可以直接验证的事实。取 `⌈c⌉`——和"取上取整、宁松勿紧"的既有
   原则一致（§5.6 上面那条 blockquote）。这一步把 S0.2 的曲线测量和生产参数直接
   挂钩，不再需要另外拍一个数。
2. **新增一个必须在 Stage 1/2 测的运行时指标：`K_max` 绑定率**——一次 eval/训练
   跑下来，"因为 K 已满而触发 Ward 强制合并"的事件数 / 总 `NEW_CLUSTER` 尝试数。
   这是 S0.2 的 unclipped `K_eff`（离线、无 `K_max` 上限）和生产路径实际表现之间
   缺的那一环：`K_eff` 大不直接等于内存超预算（已在 experiments.md S0.2 的
   决策门里讲清楚），但绑定率高**确实**直接等于路由质量下降——这才是需要盯着的
   量，且是可以在真实 eval 里量出来的，不需要额外的离线分析。
   - **绑定率 < 10%**：`c` 定得够用，不用动。
   - **绑定率持续偏高（比如 > 30%）**，先按第 1 步用更大的参考上下文重新拟合/
     加倍 `c` 重测——如果加倍 `c` 后绑定率明显下降，说明只是常数没给够，继续加到
     绑定率落回可接受区间即可（`Θ(log²N)` 的预算表已经说明加大 `c` 是线性代价，
     不是灾难性的）。
   - **加大 `c` 之后绑定率没有随之明显下降**：这是信号，说明第 1 步的前提
     （对数形状）本身站不住——真实内容的语义多样性不是 `log n` 能刻画的，回到
     §5.6 上面"若实测是幂律"那条决策门，**这不是继续调 `c` 能解决的问题，是需要
     重新评估整条曲线形状假设的问题**，调 `c` 和换假设是两件事，不要混着做。
3. **停止条件**：如果为了把绑定率压到可接受区间，`c` 必须大到让 `K_max`
   逼近"总 entry 数不再显著小于 vanilla"的地步（§4 那张 1×/1.13×/1.46× 的表
   失去意义），那么"次线性压缩"这个卖点本身对这类工作负载不成立——这时候
   应该停止扩大 `c`，转而如实报告"这条工作负载下语义多样性太高，本方案退化为
   与 vanilla 相当"，而不是无限调大 `c` 直到分数好看。这与 CLAUDE.md §4 的
   退化论证是一致的：退化的终点是现状，不是通过無限加大内存把现状伪装成改进。

#### 不设按序列长度的运行时配额

短序列不限制簇数，正常使用，撞上限才合并。理由：内存已按最坏情况预分配，短序列少用
簇**买不到任何东西**，却要付强制合并的代价，而合并不可逆。

**"只有最长的序列才可能用完配额"是错的**：簇数跟踪的是**语义多样性，不是长度**。
一份 2k token 涵盖 20 个话题的文档，比一份 32k token 通篇讲同一件事的文档用掉更多簇。

**救命的性质：两个最坏情况互斥。** 所有簇的成员数之和恒等于 `n`，所以簇多 ⟹ 每簇
成员少 ⟹ ladder 都很浅；某簇巨大 ⟹ 其它簇必然很小。**不可能同时顶满**。这是 §5.12
预分配能收敛的根本原因。

#### 合并就是预算转移机制本身（注意：不是靠"牺牲 haystack"）

K 满时一个 needle 到来：它离所有 centroid 都远 → 拿到新槽位；代价由**被合并的那两个
簇共享一个 centroid** 来承担。**novel 的内容赢得槽位，已有内容付账**——这正是 §3
想要的行为，自动发生，不需要额外机制。

> **一处曾经写错的理由，务必别再犯**：早期版本说"被牺牲的是最相似的两个簇，多半是
> 两个 haystack 簇"。**这是错的**——Ward 代价带尺寸加权 `(n_a n_b)/(n_a+n_b)`，
> 两个 haystack 大簇（n=1000, d²=0.01）代价是 500×0.01 = 5，两个 needle 单点簇
> （n=1, d²=1.0）只有 0.5×1.0 = 0.5，**Ward 优先合并的是小簇而不是大簇**。
>
> 结论仍然成立，但机制不同，见下面 §5.6-合并过程（`ward_merge_only` 第 2 步）：
> **合并两个簇不必然合并
> 它们的 entry**。两个小簇归并后若该层占用仍 `≤ B′`，entry 原样共存、无池化损失，
> needle 的分数/读出/位置不受影响（§3 的论证只依赖它自己的 entry，不依赖 centroid）。
> 付出的代价是路由质量（混合 centroid 可能误导后续 token），而不是已存内容的分辨率。

> **⚠️ "小簇合并零损失"是有条件的暂态性质，不是不变量。** 它只覆盖**单次**合并、
> 且合并后该层占用 `≤ B′` 的情形。在 `K_max` 持续绑定的场景下占用会累积：
>
> ```
> 8 needle + 2 sink + 5 haystack = 15 簇，K_max 满
> 新 token 到来 → Ward 偏好小簇 → 两个 needle 簇合并（level 0 占 2，无损）
> 再来 → 又一个 needle 并进去（占 3，仍无损）
> …… 累积到 level 0 占满 B′=8 ……
> 再来一次 → 超过 B′ → 进位触发 → needle 之间开始互相平均  ✗
> ```
>
> **multi-needle + sink 恰恰是这个失效路径的最坏情况**：Ward 的尺寸加权偏好小簇，
> 而 needle 簇正是最小的那些，于是反复合并会把 needle **系统性地漏斗到同一个簇**里，
> 其 level 0 在 `B′=8`、needle 数为 8 时正好填满。此时不仅 entry 开始合并，那个簇的
> centroid 也已是 8 个无关 needle 的混合体，会继续误导路由。
>
> 所以宣传口径必须是"**`K_max` 不绑定时**小簇合并无池化损失"，而不是无条件的
> "零信息损失"。操作后果：**multi-needle 实验必须同时报告 `K_max` 的绑定频率**，
> 否则分数差异无法归因（是聚类不行，还是预算被 needle 自己撑爆了）。这与 §12-D
> 记的那条未决问题是同一个根。

反过来的失败模式：如果那个"远"的 token 是噪声而非 needle，它同样拿到槽位，反复的
噪声会让簇集合持续抖动。**鲁棒性完全取决于 `λ_new` 调得让真正的离群点足够罕见。**

#### 真正该自适应的是阈值，不是上限

`K_max` 是**破坏性**的调节手段（超了就销毁已有结构），`λ_new` 是**预防性**的（超了
就变得更挑剔，不动已有结构）。想让簇数跟随某条轨迹，应该拧阈值：

```
维护目标轨迹  K_target(t)
K(t) > K_target  →  调高 λ_new（更挑剔，新簇更难开）
K(t) < K_target  →  调低 λ_new（有富余，允许更细的划分）
```

这样簇数被软性引导到目标形状，**没有任何一次强制合并**，`K_max` 退回纯粹的分配
安全网。代价是引入一个反馈控制器，**必须加迟滞（hysteresis）**防止 `λ_new` 在阈值
附近来回振荡把簇结构抖散。列为 v2 选项，v1 先用固定 `λ_rel`。

#### `K_max` 满时的完整合并过程

```
触发条件: 需要开新簇 且 K_max 个槽位全部 alive

候选选择（由调用方完成，不属于 ward_merge_only 这个 primitive 本身——见下方
更正。replay 时这一步也不会发生：WARD_MERGE op 的 arg0/arg1 已经把选好的
(keep_slot, free_slot) 记下来了，重放直接读，不重新计算 Δ）：
   Ward 代价矩阵（**用 `n_total`，不是 `n_eff`**——理由见 §5.5 的更正框：Ward
   代价要反映真实物理规模，`n_eff` 会被 `γ` 衰减，用它算代价会让历史长、内容
   多的簇显得"小"而被误判成廉价可合并）
   Δ[a,b] = (n_total_a·n_total_b)/(n_total_a+n_total_b) · ‖μ_a − μ_b‖²
   屏蔽对角线与 dead 槽位（§5.19-7），取 (keep_slot, free_slot) = argmin Δ

ward_merge_only(keep_slot, free_slot) 的完整内容（纯状态 mutation，不写
op_log，见下方更正）：

1. 合并簇级元数据（centroid 混合继续用 `n_eff`，物理规模计数继续用 `n_total`。
   **这是 `keep_slot` 元数据的合法写入点，和 §5.4 的 Phase 3a/3b 不冲突**——
   这里写的是"两个既有簇已积累的历史值怎么组合"，Phase 3a/3b 写的是"本批新
   到达 token 的内容贡献"，两类写入不重叠也不需要互相知道对方。**但"不冲突"
   依赖一个前提：调用这一步之前，`keep_slot`/`free_slot` 双方本批目前为止的
   内容贡献必须已经通过 Phase 3a/3b 落地——这不是自动成立的，是 §5.4"Phase 3
   拆成 3a/3b"一节要保证的调度顺序，早期设计把 Phase 3 推迟到批末运行时并不
   成立，见那节的反例）
   μ_new       = (n_eff_a·μ_a + n_eff_b·μ_b) / (n_eff_a + n_eff_b)
   n_eff_new   = n_eff_a + n_eff_b
   n_total_new = n_total_a + n_total_b
   p_hi_new    = max(p_hi_a, p_hi_b)
   current_segment_new = max(current_segment_a, current_segment_b)   # 见 §5.4
                          # 那条更正框：这只保证合并后 keep_slot 未来新开的
                          # segment id 大于 a/b 双方的历史最大值，不保证 a/b
                          # 各自的历史 segment id 互相之间不重复（它们本来就
                          # 是两个独立计数、都从 0 开始，天然靠 op_log 里各自
                          # 原本的 cluster 字段区分，不靠 segment 数值区分）

2. 合并两条 ladder —— 逐层归并 + 级联进位
   for ℓ in 0 .. L_alloc-1:
       把 a、b 第 ℓ 层的 entry 按到达顺序归并（≤ 2B′ 个）
       if 总数 ≤ B′:  直接放进新簇第 ℓ 层，该层结束      ← 关键分支，见下
       else:          保留最新的 B′ 个，其余成对 compact 进位到 ℓ+1
   顶层溢出 → 饱和累加（§5.12）
   释放其中一个槽位（alive=false）
```

调用方在拿到 `ward_merge_only` 的返回后自行 `append WARD_MERGE(keep_slot,
free_slot, -1)`（**合并方向必须记**，§5.21-2）——日志写入不属于这个 primitive。

> **更正（这一轮修的）：早期版本把"1-3+5 步"（含 Ward 代价矩阵候选选择和写
> `op_log`）都算作 `ward_merge_only` 的内容，这和它在别处的实际用法对不上，
> 必须改成上面这样。** 两处不一致：①§5.4 的 Phase 2 在调用完 `ward_merge_only`
> 之后**又**显式 `append WARD_MERGE(...)` 到本地缓冲——如果 `ward_merge_only`
> 自己也写一次，同一次合并会被记两条 `WARD_MERGE`，直接吃掉一倍不必要的
> `OP_max` 预算，且下游任何按"一次合并对应一条 `WARD_MERGE`"假设写的逻辑
> （比如 §5.21-2 的 `OP_max` 上界推导）都会被打穿。②backward 重放遇到
> `WARD_MERGE` op 时调用 `ward_merge_only(op.keep_slot, op.free_slot)`，
> 目的仅仅是复现它的状态 mutation 副作用（重放循环里的注释一直写着"只合并"）
> ——如果 `ward_merge_only` 内部还会写 `op_log`，重放就会一边**读**日志一边
> **写**日志，把一个本该只读的重放循环变成有副作用的写操作，污染
> backward 的 `ctx` 状态和 `OP_max` 计数。**正确定义：`ward_merge_only` 只
> 负责状态 mutation（合并簇级元数据、合并两条 ladder、释放槽位），候选选择
> 在它之前由调用方完成，日志写入在它之后由调用方完成**——§5.4 的 Phase 2
> 一直都是这么用的，这次只是把 §5.6 自己的定义改成和这个既有用法一致，而
> 不是反过来要求 §5.4/replay 去将就一个写错的定义。
>
> 早期版本还把"释放槽位"和"新簇写进去"并成同一步（原编号里的"步骤 4：释放
> 其中一个槽位，新簇写进去"），这在单 orphan、非批量的场景里读起来无害，但
> §5.4 的批量路由需要把"合并"和"在释放出的槽里建什么"拆成两个独立步骤——
> `ward_merge_only` 只负责前者，**"新簇写进去"是调用方的职责，不属于这个
> primitive**，调用方在这里就是"触发合并的那个 novelty token"，它接下来会走
> §5.3 的 `NEW_CLUSTER` 路径，把这个刚释放的槽当成自己的新簇写进去。这个
> 拆分是 §5.4 批量路由小节处理"Ward 合并候选池不受限"这个设计的前提，不是
> 本节自己需要，但放在这里定义，供 §5.4 引用。

**第 2 步的 `if 总数 ≤ B′` 是整个机制的关键**：合并两个簇**不必然合并它们的 entry**。

- 两个**小簇**合并 → 归并后该层占用仍 `≤ B′` → **entry 原样共存，无池化损失**（**注意这是
  暂态而非不变量，反复合并会累积占用直至越过 `B′`，见上面的警告框**），只是 centroid
  变成混合体（影响后续路由，不影响已存内容）。
- 两个**大簇**合并 → 每层都是 2B′ 个 → 全线触发 compact → **真实的分辨率损失**。

Ward 代价的尺寸加权恰好与"entry 会不会被迫合并"同向，所以 **Ward 是正确的准则**——
但正确的理由是这个，不是"它会挑中 haystack"（见上面的更正框）。

代价 `O(K²d) + O(L·B′·d)`，事件罕见（次数被"超出 `K_max` 的新簇事件数"界住）。
**注意合并后两簇的 segment 对齐（§5.11）被破坏**，这是可接受的（罕见、预算逼出来的
事件）。`K_max ≥ 2` 时永远存在可合并的一对，所以这一步不会失败。

**硬性不变量（`ward_merge_only` 必须满足，不是"大概率如此"）**：第 2 步
结束时，对**每一层** `ℓ`，`level_count[keep_slot, ℓ]` 必须精确等于该层归并
之后新 ladder 里真实存活的 entry 数——**这个值必须由第 2 步的构造过程直接
写出，不能事后靠"应该是对的"去推断，也不能沿用 `a`/`b` 任一侧合并前的旧值**。
这条不变量是上面"§5.11 的 `PAD_INSERT` 直接读 `level_count[cluster,0]`"这个
设计能够成立的**前提**，不是自动附带的性质：如果 `ward_merge_only` 的
实现在某条分支忘了同步更新 `level_count`（比如"总数 ≤ B′，直接放进新簇第 ℓ 层"
这个分支——它不触发 compact/carry，容易被误认为"不需要更新计数"，但它同样
改变了该层的占用数，`level_count` 必须一并写），下一次 `PAD_INSERT` 读到的
就是过期值，§5.11 那条修法会立刻重新失准——和这次修正之前一模一样的错误，只是
触发条件从"第一次填充"变成了"第一次 Ward 合并之后的填充"。

**必须补的单测**（和 §5.11 的对齐单测同一档次，S0.1）：构造一个合成的 Ward
合并场景（两个簇各自有已知的 ladder 状态，含至少一层触发 compact、至少一层
`总数 ≤ B′` 直接放入），合并后：（a）断言 `level_count` 在**每一层**都等于用
`pad_mask`/`w>0` 独立统计出的真实占用数——不能只测 level 0；（b）在合并后的
簇上模拟一次后续的 `PAD_INSERT`（用刚更新的 `level_count[cluster,0]` 算
`count`），断言按这个 `count` 填充之后，下一次段边界前的 level 0 逻辑流长度
确实对齐到 `2^ℓ_block` 的倍数——即，不仅测"计数器数值对不对"，还要测"用这个
计数器算出来的对齐填充确实达到了对齐的效果"，这才是这条不变量真正要保证的
最终结果。

#### `K_max = 1` 是退化边界，不是这一步的一个普通实例，必须单独判定

`K_max ≥ 2` 的保证——"永远存在可合并的一对"——**在 `K_max = 1` 时不成立**：只有一个
alive 槽位，Ward 的 `(K,K)` 代价矩阵屏蔽掉对角线后是空的，`argmin` 无定义。这不是
"罕见失败需要 fallback"，是**结构性地没有第二个槽可腾**——"K 已满 → Ward 合并腾位"
这条路径在 `K_max=1` 下根本不适用，必须在进入 Ward 之前就分流掉。

**决定：`K_max = 1` 时彻底跳过语义新簇判定，novelty 检测不改变任何 op 类型。**
具体地：

```
if K_max == 1:
    # 建立/合并的判定被短路：cluster 0 一旦存在，此后所有 primary op
    # 只能是 JOIN 或 NEW_SEGMENT，取决于 §5.3 的时序判据——novelty 距离
    # (‖k − μ_0‖² > λ_new) 仍然照常计算并写进 metadata/日志供分析用，
    # 但**从不触发新簇路径**，因为没有第二个槽可以腾——这里同样不是"强制并入
    # 最近簇"这种 fallback（那种 fallback 已经删掉，见上面"新簇形成的条件"
    # 一节的更正框），而是 K_max=1 这个退化边界下 novelty 检测本来就不产生
    # 任何新 op 类型的直接结果：唯一的簇就是最近的簇，JOIN/NEW_SEGMENT 的
    # 判定逻辑天然覆盖了这个情形，不需要额外分支。
    op = NEW_SEGMENT if (p_t − p_hi_0 > g_max) else JOIN
else:
    # 正常三路判定（§5.3），K 满时按上面的合并过程走 Ward（K_max≥2 时必有
    # 候选，不会失败，见上面"新簇形成的条件"一节的更正框）
    ...
```

这精确回答了"novelty token 到底降级为 JOIN、NEW_SEGMENT，还是特殊 fallback op"：
**是 JOIN 或 NEW_SEGMENT 之一，由时序判据独立决定，不产生新的 op 类型，也不经过
Ward**——`K_max=1` 下 Ward 合并这条路径根本不存在（没有第二个槽可腾），和
`K_max≥2` 时"Ward 候选池必然非空、这一步不会失败"是两条完全不同的代码路径，
彼此不需要共享任何 fallback 逻辑。**`K_max=1` 本质上退化成"关闭语义路由、只
保留 §5.11 的时序分段机制"**——这与 Stage 2 Config A 想要隔离出的"仅簇轴退化"
基线在设计意图上完全一致（`experiments.md` Config A 一行），现在有了精确到
op 类型的实现依据，不再是两处文档"看起来应该兼容"但没人验证过的默认假设。

CPU 参考实现（§5.18 第 2 步）和生产路径必须共享这条 `K_max==1` 分支，否则 Stage 2
的 A 档和 CPU reference 会在这个边界上分叉——这正是本条修正要防止的事。

### 5.7 簇内压缩：ladder 结构

```
level 0:  B′ 个 entry，每个 w = 1        ← 单 token，不是现在的 2-token 合并
level ℓ:  B′ 个 entry，每个 w = 2^ℓ
level ℓ 满（B′ 个）时，成对合并进位到 ℓ+1
```

因为路由发生在 flush 时、按到达顺序，**簇内成员序列就是位置序列**，所以 Fenwick
合并的天然是位置相邻的成员。

**关键事实（已核实）**：`compact()` 的 docstring 明确写着它"concatenates the two
blocks in time order and pairs **ADJACENT** slots: (slot 2i, slot 2i+1) -> slot i.
Every merged slot covers a contiguous span."——配对是**时间序上相邻**的，不是"块 A
的第 i 个配块 B 的第 i 个"。所以"到达相邻 = 位置相邻"这个贯穿全设计的假设成立。

### 5.8 簇内压缩：O(log n) 保证的准确陈述

单簇容纳 `n_total_c` 个 token 需要 `L_c = ⌈log₂(n_total_c/B′ + 1)⌉` 层，占 `B′·L_c`
个 entry（这里的 `n_total_c` 就是 §5.5/§5.6 定义的**未衰减**簇物理规模，不是
`n_eff`——两者在这条空间论证里必须是同一个量，否则界不成立）：

```
Σ_c B′·log₂(n_total_c/B′)  ≤  K·B′·log₂( n / (K·B′) )  =  O( K·B′·log n )
```

不等号来自 log 的**凹性**：给定总 token 数，簇均衡分布时求和最大。所以
**最坏情况是簇均衡**，一个簇吃掉全部反而更省——这个界是安全的。

### 5.9 簇内压缩：小簇是无损的（needle 论证的承重点）

level 0 存单 token 意味着，**成员数 ≤ B′ 的簇，每个成员都以精确形式存着，从不合并**。
这是"预算转移"的具体兑现：

- needle、罕见实体、异常内容 → 小簇 → **精确保留**（分数、读出、位置全部等同稠密）
- haystack、模板文字、高冗余内容 → 大簇 → 对数压缩

压缩率因此是**内容自适应**的，而不是像位置分桶那样按距离一刀切。

### 5.10 簇内压缩：合并算子

```
merge(A, B):                          # B 在到达顺序上更晚
    w    = w_A + w_B
    k̄    = (w_A·k̄_A + w_B·k̄_B) / w     # pre-RoPE 内容空间
    v̄    = (w_A·v̄_A + w_B·v̄_B) / w
    p_lo   = min(A.p_lo, B.p_lo)        # 锚点：精确、幂等（§2.2）
    p_hi   = max(A.p_hi, B.p_hi)
    sum_wp = A.sum_wp + B.sum_wp        # int64 累加器；p_mid 在读出时才由它算出
    Σ, Γ : 现有 Chan-style 二阶矩合并 + rank-1 截断（唯一有损的一步）
```

`O(d)`，与现有 `compact()` 完全同构——**Σ/Γ 那套一行都不用改**，只是统计空间从
post-RoPE 换成 pre-RoPE。**顺带的好处**：Σ 不再混入位置相位方差，簇内同质时 rank-1
近似精度应系统性恢复——这是另一分支 D1 自检里 width≥8 失真的根因之一。

### 5.11 段边界：对齐填充（不是"拒绝合并"）

#### 先纠正一个前提：这里没有"逐对否决"这回事

`compact` 是把 2B′ 个 entry 拼起来后**按固定下标 `(2i, 2i+1)` 一次性全配对**的
向量化操作。没有"这一对不合并、那一对合并"的接口——想给某一对开口子，就得打散整个
向量化结构。**所以不能在"要不要合并这一对"上做文章，只能在"让边界落在哪里"上做。**

#### 可实现的阻断 = 对齐填充，而且现有代码天然支持

不去拒绝跨段的那一对，而是**插入空位把段边界推到"两对之间"**：

```
不填充:  [a1][a2][a3] | [b1][b2]        ← | 是段边界
配对:    (a1,a2) (a3,b1) (b2,…)         ← 第二对跨段 ✗

填充一个空位:
         [a1][a2][a3][∅] | [b1][b2]
配对:    (a1,a2) (a3,∅) (b1,b2)         ← 没有一对跨段 ✓
```

**关键是空位用 `w=0` 的 entry 实现，在现有合并数学下天然是恒等元**：

- 内容：`alpha = wa/(wa+0) = 1` → `k_out = ka`，真实 entry 原样通过
- 计数：`w_total = wa + 0 = wa`
- 锚点：空位初始化成 `p_lo=+INT_MAX, p_hi=-1`，`min/max` 自动返回真实 entry 的锚点；
  `sum_wp` 初始化成 `0`，加法后不变（`w=0` 的 entry 对加权和的贡献本就是 0）

**注意一个陷阱**：`compact` 里有 `w_total.clamp(min=1e-8)`，两侧都为 0 时
`alpha = 0` → `k_out = kb`（垃圾值，但 `w=0`）。所以**填充槽必须在读出侧被掩码
屏蔽，不能只靠 `w=0` 自然消失**。

**上面只验证了均值和锚点，Σ/Γ 的 rank-1 路径要单独核实——它不是自动成立的。**
核对 `compact()` 现有实现（`log_kv_cache.py:723-816`）：`frac_a = wa/w_total`、
`frac_b = wb/w_total`、`cross_frac = frac_a·frac_b`。`wb=0` 时 `frac_a=1,
frac_b=0` 精确成立，于是 Σ 的三个 factor 是：

```
factor_a     = sqrt(frac_a · s2a) · sua = sqrt(s2a) · sua        ← 就是 A 自己，未受扰动
factor_b     = sqrt(frac_b · s2b_pad) · sub_pad = sqrt(0) · sub_pad = 0 · sub_pad
factor_cross = sqrt(cross_frac) · dk = 0 · (ka − k_pad)
```

**`factor_b`/`factor_cross` 的安全性不是"乘 0 天然为 0"能保证的**——`0 × 有限数 = 0`
没错，但 `0 × NaN = NaN`、`0 × Inf = NaN`。所以 `merge(A, pad) == A` 成立的真实
前提是**填充槽的 `k`、`v`、`sigma_u`、`sigma2`、`gamma_a`、`gamma_b`、`gamma`
全部是有限的零值**，不能是未初始化的垃圾内存——一旦有一个字段带着 NaN/Inf，
即使它的权重是 0，也会通过这几个乘法把 NaN 传染进一次原本应该是恒等操作的合并里，
污染一个真实槽。

**硬性实现规则**：填充槽的**全部**字段（不只是 `w`/锚点）必须显式 `zero_()`，
不能依赖"权重为 0 所以内容无所谓"的直觉。这不是新发明——`_append_level0`
（`log_kv_cache.py:1069-1076`）在 `second_order=False` 时已经对着同一批 buffer
做了这件事："Zero in place instead of materializing zero tensors to copy from"。
填充槽只是把同一条纪律用在另一个触发条件上。

**对应的单测（S0.1，必须有，不是"w=0 恒等元"那条的可选补充）**：`merge(A, pad)`
要复现 `A` 的**完整**统计元组——`k, v, w, p_lo, p_hi, sum_wp, sigma_u, sigma2,
gamma_a, gamma_b, gamma`——逐位一致，不能只测均值和锚点。给定上面的推导，
`_rank1_psd_from_factors`/`_rank1_cross_from_factors` 在只有一个非零 factor 时，
Gram 矩阵本就是秩 1 且只有一个非零对角元，幂迭代在这个退化情形下不是"近似收敛"
而是精确解，所以这条单测应当断言逐位相等（或 `TestRank1Approximation` 已用的
同一档 float 容差），不是近似值。

#### 填充槽的实际写入模板

`insert_pad_entries(cluster, level=0, count)` 不应该绕开普通追加路径；它应该循环
`count` 次，把一个"零权重 entry"追加到该 cluster 的 level 0，然后复用同一套
`_binary_carry`/级联进位逻辑。这样 pad 和真实 token 经过完全相同的配对顺序，replay
时也不需要第二套高层插入语义。

每个填充槽必须按下面这张表初始化；这张表是实现约束，不是建议：

| 字段/掩码 | 填充值 | 原因 |
|---|---|---|
| `w` | `0` | 让 pad 在均值与计数上成为恒等元 |
| `pad_mask` | `true` | 标注纯 pad/做断言；compact 输出侧必须按 `w_out==0` 重算 |
| `k`, `v` | 全 0 且 finite | 防止 `0 × NaN/Inf` 经 `dk` 或均值路径污染真实 entry |
| `sigma_u`, `sigma2` | 全 0 且 finite | 防止 rank-1 PSD 路径里的零权重 factor 传播 NaN/Inf |
| `gamma_a`, `gamma_b`, `gamma` | 全 0 且 finite | 防止 rank-1 cross 路径里的零权重 factor 传播 NaN/Inf |
| `p_lo` | `+INT_MAX` | 让 `min(real.p_lo, pad.p_lo)` 返回真实锚点 |
| `p_hi` | `-1` | 让 `max(real.p_hi, pad.p_hi)` 返回真实锚点 |
| `sum_wp` | `0` | `w=0` 的位置贡献必须为 0 |

两个实现不变量必须同时成立：

1. **compact 不变量**：`compact(real, pad)` 与 `compact(pad, real)` 在有效输出上都等价于
   `real`；`compact(pad, pad)` 允许产生内容为 0 的无效 entry，但输出必须保持
   `w=0`，并在读出侧被过滤。**`pad_mask` 不能用输入两侧的 OR 直接传播**，否则
   `compact(real, pad)` 会把一个真实输出误标成 pad；compact 输出侧的 `pad_mask`
   应按 `w_out == 0` 重新生成，或由同等语义的有效位反推。
2. **读出不变量**：扁平化 slot 池时先用 `w > 0` 生成 entry 级有效位，再做锚点去重和
   slot 展开；`pad_mask` 只用于调试断言/统计纯 pad 占用，不参与把真实 `w>0` entry
   排除出读出。任何 `p_lo=+INT_MAX, p_hi=-1` 的纯 pad entry 都不能进入
   `dedup_anchors` 或 `apply_rope`。

`n_total_c` 只统计真实 token，不随 pad 增加；否则下一次 `count = (-n_total_c) mod
2^ℓ_block` 会把"为了对齐插入的空位"也当作真实成员，导致边界越补越偏。需要记录 pad
占用时使用 `pad_mask`/`level_count`，不要复用 `n_total_c`。

#### 代价是指数的，这直接定死了 `ℓ_block`

要保护到第 ℓ 层不被跨段污染，段的起始下标必须是 `2^ℓ` 的倍数（第 ℓ 层的一个 entry
对应第 0 层的 `2^ℓ` 个）：

```
每个段边界的填充浪费  ≤  2^ℓ_block − 1  个槽位
```

代入 32k 参考配置（1320 个 entry），假设全文约 100 个段边界：

| `ℓ_block` | 每边界浪费 | 总浪费 | 占预算 |
|---:|---:|---:|---:|
| 1 | 1 | 100 | 8% |
| 2 | 3 | 300 | 23% |
| 3 | 7 | 700 | **53%，不可接受** |

**所以 `ℓ_block` 实际只能取 1 或 2**，这不是拍脑袋，是指数增长逼出来的。

#### `PAD_INSERT(cluster, level, count)` 的字段定死

**`level` 恒为 0，`count` 有闭式公式，两者都不是运行时才决定的自由量**：

```
count = (-level_count[cluster, 0]) mod 2^ℓ_block   # 读现有 buffer，不需要新计数器；见下方更正框
level = 0                                            # 恒定，v1 不支持在别的层直接插 pad
```

**为什么 `level` 恒为 0**：填充遵循"和普通 entry 同一条纪律"（本节开头已经定的
原则）——普通 token 只在 level 0 追加，靠自然的进位级联往上传播到更高层；填充槽
同理，**从不直接插到 level ≥ 1**。往 level 0 插入 `count` 个 `w=0` 空位，`_binary_
carry` 的标准级联逻辑会自动把它们和真实成员一起向上折叠，折叠出的高层空位天然
对齐——不需要一个"往 level 3 直接插空位"的分支，那种分支也没有良定义的语义（高层
entry 覆盖一个跨度，"插一个空的高层 entry"意味着什么本身就不清楚）。所以 `level`
字段现在纯粹是**日志格式的自描述占位**，v1 里读到的值必须恒为 `0`，为将来（如果
真的出现直接高层填充的需求）保留 schema 空间，但当前不使用。

**为什么 `count` 是"把某个累积计数补到下一个 `2^ℓ_block` 的倍数"所需的余数**：
标准的对齐到 2 的幂边界的公式，和内存分配器的 padding 计算是同一件事。

> **一处曾经写错的计数器（两个人独立发现同一个坑，落地方式不同，这里合并成一条
> 结论）**：早期版本用 `n_total_c`（§5.5/§5.6 定义的、单调不减的簇**真实**物理
> 规模，不含 pad）来算这个余数。**这在第一次填充之后就是错的**：`n_total_c`
> 只数真实 token，不数已经插过的 pad，而 `PAD_INSERT` 真正要对齐的是**level 0
> 的逻辑插入流长度**（真实 token 和 pad 混在一起、按到达顺序排成的那条流），
> 两者从第一次填充起就分叉。举一个 `ℓ_block=1` 的反例：段 A 有 3 个真实 token，
> `n_total_c=3`，`count=(-3) mod 2=1`，插 1 个 pad，level 0 流变成 4 个（对齐）；
> 段 B 来了 1 个真实 token，若用 `n_total_c` 计数（不含 pad），此时 `n_total_c=4`
> （3 真实 + 1 新真实，pad 从不计入），下一次开新段时公式给出 `count=(-4) mod
> 2=0`——**但 level 0 流的真实长度是 `3+1(pad)+1=5`，是奇数，下一个边界前明明
> 还需要再插 1 个 pad 才能对齐，公式却说不需要**。用真实 token 数去驱动一个要
> 对齐"真实+pad 混合流"的计算，从设计上就注定会在第一次填充后失准，不是边界
> 条件疏漏。
>
> **修法：直接读现有的 `level_count[cluster, 0]`，不引入新计数器**。
> `level_count[cluster, 0]` 是这个簇 level 0 **当前**的占用数——它已经在
> §5.13 的 buffer 表里，是现有 `_binary_carry` 机制本来就要维护的状态，不是
> 新增负担。关键认识：**只要 `B′` 是 `2^ℓ_block` 的倍数**（默认 `B′=8`，
> `ℓ_block∈{0,1,2}` ⇒ `2^ℓ_block∈{1,2,4}`，均整除 8），"level 0 当前占用数
> mod 2^ℓ_block"和"该簇从建立以来的累积逻辑插入数 mod 2^ℓ_block"**永远相等**
> ——因为每次 level 0 满 `B′` 个触发进位时，`B′ ≡ 0 (mod 2^ℓ_block)`，进位不
> 改变这个余数，两个量的 mod 值全程同步，不需要真的去维护一个独立的全局累积
> 计数器。**这比引入 `n_unit_c` 更好，不只是更省一个 buffer**：Ward 合并
> （§5.6）之后 `level_count[cluster, 0]` 会被合并过程直接重写成正确的新值
> （因为合并本身就是对 ladder 结构的重建），后续 `PAD_INSERT` 读到的自动就是
> 合并之后的真实状态；而一个独立维护的全局计数器在 Ward 合并时需要额外一步
> "把两个簇的计数器合并成新计数器"的手工同步逻辑，且合并后"从簇建立以来的累积
> 插入数"这个概念本身在语义上已经模糊（两个簇的历史被拼在了一起）——用
> `level_count` 就没有这个问题，它只关心"现在"，天然对 Ward 合并免疫。
>
> **前提要显式校验**：`B′` 必须是 `2^ℓ_block` 的倍数，否则上面的等价关系不
> 成立，回到"第一次填充后就分叉"的老问题。默认值 `B′=8` 对 `ℓ_block∈{0,1,2}`
> 天然满足，但如果 `B′` 被实验覆盖成一个不是 4 的倍数的值（比如 `B′=6`），必须
> 在构造时一并校验 `B′ % (2**ℓ_block) == 0`，不满足则硬失败——和 `ℓ_block ∈
> {0,1,2}` 那条校验（§5.21-2）放在同一处，同一类"参数组合失配就拒绝，不静默
> 算错"的处理方式。
>
> `n_total_c` 保持不变、继续只数真实 token——它是 Ward 代价（§5.6）和 §5.8
> 空间界要的量，这两处算的是"这个簇的真实内容有多少"，混入 pad 计数会让 Ward
> 代价虚高（簇看起来比实际更"大"）、让 §5.8 的 `L_c = ⌈log₂(n_total_c/B′+1)⌉`
> 层数估计虚高——这条边界依然成立，只是不需要为它另开一个计数器来对齐 pad，
> 复用 `level_count` 就够。

代入前面的表可以直接验证一致性：`ℓ_block=2` 时 `count ∈ {0,1,2,3}`，最坏 `count=3`，
正好等于"每边界浪费 ≤ `2^ℓ_block−1` = 3"这行——**这条公式和上面代价表用的是同一个
量，不是巧合，是同一个约束的两种写法**。`count=0` 的情况（`level_count[cluster,0]`
本来就已经对齐）意味着这次不需要填充，此时不应该产生 `PAD_INSERT` op（省一个
`OP_max` 名额）。

#### 更深一层：阻断不创造预算，它只是换了合并哪一对

ladder 的自然行为——level 0 存单 token、满了才合并——**本身就已经是"能负担就保持
分开，负担不起才合并"**。真正发生的每一次合并都是预算逼出来的。阻断某一对不会凭空
多出槽位，只会让**别的某一对**被合并掉。

所以问题的正确提法是"既然必须合并，该合并哪一对"——这是个**合并顺序**问题，而固定
下标的 Fenwick 配对没有表达这个选择的接口。于是设计空间收敛成两个自洽的终点：

| 方案 | 跨度质量 | 工程代价 |
|---|---|---|
| **Fenwick + 对齐填充（`ℓ_block ≤ 2`）** | 底层干净，高层污染 | 极低，`compact` 零改动，向量化保持 |
| 扁平表 + 贪心最小代价相邻合并 | 严格更优 | 需 per-(B,G) gather，失去二进制计数器的干净核算 |

第二个就是最初那个"连续性约束 Ward 合并"，在存储层作为 Fenwick 的**替代**回来了。
它的合并代价可以直接取"内容 Ward 代价 + 跨度增量"，两个误差源统一在一个准则里。

**选哪个由 S0.2 的一个数字决定**：一个簇平均有几个 segment，相对于 `B′`。
`S ≪ B′` → 每层至多 S/B′ 的 entry 被污染，对齐填充够用；`S ~ B′` 或更多 → Fenwick
的固定配对在跟数据对着干，值得上第二个方案的工程量。**v1 先走方案一，把这个分叉点
显式登记为 S0.2 的输出。**

### 5.12 预算定尺：`L_alloc` 按均衡界，顶层做饱和累加

矩形预分配 `K_max × L_max × B′`（其中 `L_max = ⌈log₂(N/B′)⌉`）隐含假设是**每个簇都
独自装下整条序列**——而由 §5.6 的"两个最坏情况互斥"，这在物理上不可能同时发生。

真实上界由均衡界给出（§5.8 的凹性）：

```
E_max(n, K) = K · B′ · ⌈log₂( n / (K·B′) + 1 )⌉
```

| 上下文 | 矩形超配 `K·B′·L_max` | 真实上界 `E_max` | 超配倍数 |
|---|---:|---:|---:|
| 32k（K=15）| 15×8×13 = 1560 | 15×8×9 = 1080 | 1.44× |
| 1M（K=20）| 20×8×17 = 2720 | 20×8×13 = 2080 | 1.31× |

（对照极端不均衡：一个簇吃掉全部 32k、其余 14 个各 1 个 token，总 entry 数只有约
110——远小于均衡时的 1080，再次验证均衡才是最坏。）

**所以 ladder 深度按均衡界定尺**：

```
L_alloc = ⌈log₂( N / (K_max · B′) + 1 )⌉ + δ        # δ = 2~3 安全余量
```

**代价与兜底**：极端不均衡时（某簇独吞整篇文档）它会超出 `L_alloc` 层。处理方式不是
报错，而是**让最顶层变成饱和累加器**——顶层 entry 的 `w` 允许超过名义的 `2^L`，
继续吸收进位。这个降级方向正确：一个吞掉整篇文档的簇，本来就该被压得最狠。

实现上不需要新机制——`compact(k1,v1,w1, k2,v2,w2)` 本来就是按 `w` 加权、
**不要求两侧等宽**，所以宽度超标在数学上现成支持。要改的只是 `_counts[ell]` 那套
"同层等宽"的簿记假设（**和对齐填充要改的是同一处，见 §5.19-2**）。

### 5.13 数据结构（buffer 清单）

按 (B, G) 独立，所有张量前两维都是 `(B, G)`。

**簇级元数据**（小，`K_max` 量级）：

| buffer | shape | dtype | 用途 |
|---|---|---|---|
| `centroid` | `(B,G,K_max,d)` | fp32 | 语义身份，路由用 |
| `n_eff` | `(B,G,K_max)` | **fp32** | centroid 混合权重。**必须浮点**——`γ` 衰减会产生非整数。**只喂 §5.5 的 centroid 更新，不进 Ward 代价**（§5.5/§5.6 的更正框）|
| `n_total` | `(B,G,K_max)` | int32 | 簇的真实物理规模（**只数真实 token，不含 pad**），单调不减、从不衰减。**Ward 代价（§5.6）和 §5.8 的空间界都用这个**。`§5.11` 的 `PAD_INSERT` 对齐**不用这个**，直接读 `level_count[cluster,0]`，见 §5.11 的更正框 |
| `p_hi_c` | `(B,G,K_max)` | int32 | 该簇最近一次收到成员的位置（join cost + segment 判定）|
| `current_segment` | `(B,G,K_max)` | int32 | 该簇当前最新的 segment id，下一次开新段用 `current_segment+1`（见 §5.4"Phase 1 缺持久 segment 状态"一节）。**写入点和 `p_hi_c` 同一档**：Phase 3a（紧跟 Phase 1，向量化）/3b（内联进 Phase 2）逐簇更新，不是批末统一写入，见 §5.4"Phase 3 拆成 3a/3b"一节；`ward_merge_only` 合并时取 `max` |
| `alive` | `(B,G,K_max)` | bool | 槽位占用。**每个头实际用几个簇可以不同**，`K_max` 只是共享上界 |

**entry 存储**（主体，全索引布局，不用自由表）：

```
(B, G, K_max, L_alloc, B′, ·)
```

每 entry 存 `k̄_raw(d)`、`v̄(d)`、`w`（**fp32/int32，不跟 activation dtype 走**）、
`p_lo`/`p_hi`（int32）、`sum_wp`（**int64**，`p_mid` 由它在读出时算出）、以及现有
rank-1 统计 `σu/σ2/γa/γb/γ`（约 3d，activation dtype，fp16/bf16 均可，这些是内容
向量不是计数）。按 `K=15, B′=8, L_alloc=11` 是 **1320 个 entry**，对比 vanilla 的约
2560 槽——**我们更小**。

> `sum_wp` **必须是 int64**：量程是 `Σ w_j p_j ≤ n²`，1M 上下文下达 `10¹²`，int32
> 会静默溢出。
>
> `w` **不能沿用现有 `level_w` 那样跟着 activation dtype 建 buffer**（现有
> `log_kv_cache.py:346-349` 是 `torch.zeros(..., dtype=dtype)`）。fp16 整数精确
> 表示上限是 2048、溢出上限 65504，而 1M 上下文下一个高冗余大簇的 `w` 可以到几十
> 万，必须显式声明 `w` 的 buffer 为 fp32 或 int32，`log(w/M)` 之前再转 fp32——这条
> 在 §5.20-B 的改动对照表里也有，这里是实现者最先会看到的地方，直接标出来，不要
> 让人照抄旁边 `k̄_raw`/`v̄` 的 dtype 建错。

**ladder 簿记**（对现有代码改动最大的地方）：

| buffer | shape | dtype | 用途 |
|---|---|---|---|
| `level_count` | `(B,G,K_max,L_alloc)` | int16 | 每簇每层已占用数。现有代码是全局 `_counts[ell]` 标量，现在要**按簇**独立维护二进制计数器状态 |
| `pad_mask` | `(B,G,K_max,L_alloc,B′)` | bool | 哪些槽是 `w=0` 的对齐填充（§5.11 的陷阱）|

**路由决策日志**（§11-A 的硬性要求）：

| buffer | shape | dtype | 用途 |
|---|---|---|---|
| `op_log` | `(B,G,OP_max,4)` | int32 | **操作日志，不是 token→cluster 映射**，只记 token 归属不足以重建结构，完整语义/顺序/重放算法见 §5.21-2。**生命周期和本表其它 buffer 不同，且分配只发生在训练路径**：不是 `reset_parameters()` 原地 `zero_()` 复用的持久 buffer，而是只在 `LogKVStreamTrainingAttention.forward()`（训练）内部，每次调用开头重新绑定成全新分配的张量；`CausalSelfAttention._log_kv_training_forward()`（推理/生成路径的真实入口，**不是** `LogStructuredKVCache.forward()`——后者恒定 `raise RuntimeError`，见 §5.21 的更正框）调用共享路由逻辑时传 `record_op_log=False`，完全不分配、不写入这两个字段——gating 规则、以及"448MB 只是单个 in-flight forward 的代价，梯度累积/pipeline 会按并发数相乘"，见 §5.21-2 新增的两节 |
| `op_log_len` | `(B,G)` | int32 | `op_log` 当前**有效**行数——`op_log[b,g,:op_log_len[b,g],:]` 才是已写入的合法内容，之后的行是未写入/未定义，**任何遍历 `op_log` 的代码（重放、`scan_op_log`、S0.8 对拍）都必须先按这个长度截断，不能扫整个 `(OP_max,4)`**，见 §5.21-2 的更正框 |

容量与内存：`OP_max = 4·T_max`，这不是经验估计，是有推导的硬上界，见 §5.21-2。
32k 下 `(1,8,4·32768,4)` int32 ≈ 16MB/层，28 层共 **约 448MB**——是 §5.21-2 那份
正确性升级的代价，不是可选项。`op_log_len` 本身 `(B,G)` int32，相对 448MB 可
忽略不计，不需要单独进内存账目。

**per-head 尺度估计**（§5.2）：**形状是 `(n_layer, G)`，不是 `(B, G)`**——`s_h`
是 §5.21-4 定案的**离线标定常量**,标定的是模型本身在每个 (layer, KV头) 上的 key
尺度,和当前跑的是哪个 batch 元素无关。它应该是从标定 artifact 加载的常量张量、
随 config 走,使用时（例如 §5.2 的 `λ_new = λ_rel·s_h` 计算）再广播到 batch 维,
**不是每次 `reset_parameters()` 都要初始化的 per-cache 运行时 buffer**。之前把它
写成 `(B,G)` 运行时 buffer 是和"离线标定"这个决定不一致的残留——§5.19-8 那条
"`s_h` 要在 `reset_parameters()` 里初始化成一个合理常数"说的是**没有标定值可用时
的兜底默认**（比如刚加进配置、还没跑过标定 pass 的层）,不是"每次 reset 都要重新
估计",两者不矛盾但容易读错,这里一并澄清。

**两个布局上的好处**：

1. **读出不需要 gather**。entry 躺在固定位置，`get_attention_state()` 只要 `reshape`
   成 `(B,G,K_max·L_alloc·B′,·)` 再配一个由 `level_count`/`pad_mask`/`w>0` 生成的
   有效位掩码。没有 indirection，没有 free list，没有 per-element gather。
2. **簇分配器是平凡的**。`K_max` 小时找空位就是 `(~alive).float().argmax(-1)`，
   Ward 合并要的 pairwise 距离是 `(B,G,K,K)`，都不值一提。

### 5.14 新模块 `litgpt/log_kv_position.py`

无状态、无可学参数、CPU 可测。

```python
# 锚点合并：精确、可结合。lo/hi 是幂等半格，sum_wp 是整数加法
def merge_anchors(lo1, hi1, swp1, lo2, hi2, swp2):
    lo  = torch.minimum(lo1, lo2)
    hi  = torch.maximum(hi1, hi2)
    swp = swp1 + swp2            # int64；不要在这里除，除法留到读出（否则累积舍入）
    return lo, hi, swp


# 质心锚点：读出时才由累加器算出，并夹回 [lo, hi]
def mid_anchor(lo, hi, sum_wp, w):
    """round-half-up，全整数运算（理由见下方"舍入规则"）。sum_wp/w 都是 int64。
    **前提：调用方必须先过滤掉 w=0 的纯 pad/dead entry（见下方 dedup_anchors 的
    说明），这个函数不对 lo>hi 的倒置区间负责。** 对纯 pad entry（`lo=+INT_MAX,
    hi=-1`，§5.11 的合并恒等元哨兵），`clamp(mid, lo, hi)` 按 `min(max(x,lo),hi)`
    的标准定义展开是 `min(max(x, INT_MAX), -1) = min(INT_MAX, -1) = -1`——一个
    确定但无效的坐标，直接喂给 `_rotate_at_anchors` 会用 `-1` 去索引
    `cos_cache`，Python/张量的负索引语义会**静默环绕到最后一个位置**而不是报错，
    产出一个看似合法实则完全错误的旋转结果，比越界崩溃更危险。"""
    ww  = w.clamp_min(1)
    mid = torch.div(2 * sum_wp + ww, 2 * ww, rounding_mode="floor")   # = round-half-up
    return torch.clamp(mid, lo, hi)


# 锚点 -> 任意 key 空间向量的旋转版本：标准 RoPE 数学，在几个确定的整数坐标上
# 各转一次。k_raw 和 sigma_u/gamma_a 是同一个函数的两次不同调用（见下方说明），
# 不是两套机制。
def _rotate_at_anchors(content, anchors, cos_cache, sin_cache, rope_n_elem):
    """anchors: (..., S, M) int64，**确定的整数坐标**（p_lo/p_hi 是真实成员位置，
    p_mid 是质心，两者对这个函数完全一样——它不关心坐标"真不真实"，只要求是单一
    确定的整数，rho 就恒为 1，不存在相消）。返回 (..., S, M, d)。
    调用前必须 clamp 无效锚点（哨兵值会越界，见 §5.19-4）。"""
    cos = cos_cache[anchors]        # (..., S, M, rope_n_elem)
    sin = sin_cache[anchors]
    x   = content.unsqueeze(-2)[..., :rope_n_elem]
    h   = rope_n_elem // 2
    rot = torch.cat((-x[..., h:], x[..., :h]), -1)
    roped = x * cos + rot * sin
    tail  = content.unsqueeze(-2)[..., rope_n_elem:].expand(*roped.shape[:-1], -1)
    return torch.cat([roped, tail], -1)


def materialize_anchor_keys(k_raw, anchors, cos_cache, sin_cache, rope_n_elem):
    return _rotate_at_anchors(k_raw, anchors, cos_cache, sin_cache, rope_n_elem)


def materialize_anchor_directions(sigma_u_raw, gamma_a_raw, anchors, cos_cache, sin_cache, rope_n_elem):
    """Σ/Γ 的 key 空间方向必须和 k_raw 同一套旋转，理由见下方"为什么 Σ/Γ 也要转"。
    sigma2/gamma（标量特征值）和 gamma_b（value 空间方向）不经过这个函数——
    value 从不被 RoPE，标量没有方向可转。"""
    sigma_u_eff = _rotate_at_anchors(sigma_u_raw, anchors, cos_cache, sin_cache, rope_n_elem)
    gamma_a_eff = _rotate_at_anchors(gamma_a_raw, anchors, cos_cache, sin_cache, rope_n_elem)
    return sigma_u_eff, gamma_a_eff


# 去重 + mass bias 摊薄因子 + 无效锚点净化，三件事在一个函数里做，顺序固定
def dedup_anchors(lo, hi, mid, w):
    """
    输入 lo/hi/mid/w: (..., S) —— 每个 entry 一份。
    返回矩形张量，**不做变长去重**（GPU attention 需要固定形状 + mask，不是
    ragged list）：
        anchors:    (..., S, 3) int64   —— 固定 3 槽，顺序恒为 [lo, mid, hi]
        slot_valid: (..., S, 3) bool    —— 这个虚拟槽是否参与 attention
        M:          (..., S)    int64   —— slot_valid.sum(-1).clamp_min(1)，**返回前
                                            已经 clamp**，调用方用 log(w / M) 而不是
                                            log(w) 做 mass bias（§2.3），不需要、也
                                            不应该自己再 clamp 一次

    计算顺序固定为下面两步，**顺序不能换**——先处理"entry 本身是否有效"，
    再在有效 entry 内部做去重，因为无效 entry 的 lo/hi 本身就是哨兵值，
    先去重会用哨兵参与比较，产出未定义结果：

    1. entry_valid = (w > 0)                          # 覆盖§5.11的显式 pad
       #                                                和从未写入过的原生空槽
       #                                                ——两者在存储上都是
       #                                                w=0，不需要区分来源
       对 entry_valid=False 的 entry：三个槽的 slot_valid 全部置 False，
       **三个槽的 anchors 全部覆写成安全哨兵 0**（不是 INT_MAX/-1，那一对是
       §5.11 merge 阶段的恒等元哨兵，只在"作为 merge 的输入"时安全；到了这里
       已经是 merge 之后的最终读出阶段，绝不能把 INT_MAX/-1 传给
       `_rotate_at_anchors` 去索引 cos_cache——理由见 mid_anchor 的 docstring）。
       这一步必须先于第 2 步执行，否则第 2 步会拿 INT_MAX/-1 之类的哨兵去和
       其它候选比较"是否重复"，比较结果没有意义。

    2. 对 entry_valid=True 的 entry，在 [lo, mid, hi] 三元组内部去重：
       第一次出现的位置 slot_valid=True，其后数值相同的位置 slot_valid=False
       （anchors 数值本身保留不覆写——重复值本来就等于第一次出现的值，覆写与否
       不影响正确性，但不覆写更便于调试时肉眼核对）。
       p_lo==p_hi 的年轻 entry（level 0、只有一个真实成员）三槽数值相同，
       去重后收敛为 M=1，与 CLAUDE.md §2.3 的论证一致。
    """
    ...
```

**为什么 `M_s` 现在是 per-entry 张量而不是标量**：早期的伪代码签名 `(unique_anchors,
M)` 没说清楚 `M` 的形状，读起来容易以为是一个全局标量或者某种变长列表长度。矩形化
之后 `M` 就是普通的 `(..., S)` int 张量，`log(w_s / M_s)` 是逐元素运算，和现有
`log_kv_slot_attention` 的其它逐槽张量运算完全同构，不需要特殊处理。

**为什么 `M` 必须在 `dedup_anchors` 内部就 `clamp_min(1)`，不能留给调用方**：无效
entry（`w=0`）三个槽的 `slot_valid` 全部是 `False`，`slot_valid.sum(-1)` 对这类
entry 算出 `M=0`。如果不 clamp，调用方算 `w/M` 就是 `0/0`——**这个 NaN 和 `w`
本身是否等于 0 无关，是除法本身的 0/0，`log_kv_slot_attention` 现有的
"`slot_w >= 1 by construction`"这条不变量（`log_kv_cache.py:1626`）到语义簇路径
上不再成立，必须显式补回来**。补法完全类比 `mid_anchor` 已经在用的
`ww = w.clamp_min(1)`（§5.14）：`M.clamp_min(1)` 之后，无效 entry 的
`w/M = 0/1 = 0`，`log(0) = -inf`（IEEE754 良定义，不是 NaN），`λ·(-inf)` 在
`λ≠0` 的分支里是良定义的 `-inf`（不是 `0·(-inf)` 那种会产出 NaN 的模式——现有
代码用 `if lam != 0.0:` 门控防的正是那一种，这里从一开始就没有落入那个模式）。
`score.add_(-inf)` 让该槽分数变成 `-inf`，随后现有的 `masked_fill_(~mask, -inf)`
再把它显式盖成 `-inf` 一次——**两者顺序不需要改变，`log_kv_slot_attention` 现有
的"先加 bias、后 mask"这个顺序原封不动地对语义簇路径安全**，`M.clamp_min(1)`
这一处补丁就足够，不需要像"先构造 mask、再算 bias"那样重排整个计算顺序。这也是
为什么这个 clamp 被放进 `dedup_anchors` 内部而不是留给每个调用方各自记得写一遍：
`M` 的唯一合法用途就是做这个除法，把安全性钉在产出 `M` 的地方，调用方就不可能
漏掉。

#### `dedup_anchors`/`materialize_anchor_keys` 的输出到 `log_kv_slot_attention` 输入之间还缺一步展开

**`dedup_anchors` 的 `M` 和输入的 `w` 都是 per-entry 张量（`(...,S)`，`S` 是
entry 数），但下面"新增第三个掩码参数"一节里 `log_kv_slot_attention` 要的
`M_s`/`slot_w` 是 per-virtual-slot 张量（`(B,G,S_pooled)`，`S_pooled=S·3`，
flatten 了每个 entry 展开出的 3 个虚拟槽）——中间必须有一次显式广播，前面的
文字从未把这一步写成代码，容易被实现者跳过，或者对着两个不同含义的轴形状
硬凑出一个不对的 reshape。**

沿用§5.13"读出不需要 gather"一节的约定，entry 张量在喂进这条流水线之前，
已经从 `(B,G,K_max,L_alloc,B′,·)` reshape 成 `(B,G,S,·)`，
`S=K_max·L_alloc·B′`——下面 `lo`/`hi`/`mid`/`w`/`k̄_raw`/`v̄` 全部在这个
形状下：

```python
anchors, slot_valid_3, M = dedup_anchors(lo, hi, mid, w)   # (B,G,S,3)/(B,G,S,3)/(B,G,S)

slot_k_3 = materialize_anchor_keys(k_raw, anchors, cos_cache, sin_cache, rope_n_elem)
                                                              # (B,G,S,3,k_dim)——
                                                              # 每个虚拟槽的 key 在
                                                              # 不同锚点位置各转
                                                              # 一次，天然不同

# v̄/w/M 是"entry 内容"，不随锚点 a 变化——3 个虚拟槽共享同一份，只是广播
# （不经过 _rotate_at_anchors，value 从不被 RoPE）：
v_3      = v.unsqueeze(-2).expand(*v.shape[:-1], 3, v.shape[-1])   # (B,G,S,3,v_dim)
                                                                      # value_{s,a}=v̄_s
                                                                      # （CLAUDE.md §2.3）
slot_w_3 = w.unsqueeze(-1).expand(*w.shape, 3)                      # (B,G,S,3)
M_3      = M.unsqueeze(-1).expand(*M.shape, 3)                      # (B,G,S,3)

# 五者用同一种 flatten（合并 (S,3) 那一对轴），保证顺序对齐；带内容维
# （k_dim/v_dim）的两个张量合并的是导数第三、第二维，不带内容维的三个
# 张量合并的是最后两维。S 取自 w（形状 (...,S)，末维无歧义就是 S）——
# 不取自 anchors（形状 (...,S,3)，倒数第二维才是 S，容易数错）：
S_pooled   = w.shape[-1] * 3
slot_k     = slot_k_3.reshape(*slot_k_3.shape[:-3], S_pooled, slot_k_3.shape[-1])
slot_v     = v_3.reshape(*v_3.shape[:-3], S_pooled, v_3.shape[-1])
slot_w     = slot_w_3.reshape(*slot_w_3.shape[:-2], S_pooled)
slot_valid = slot_valid_3.reshape(*slot_valid_3.shape[:-2], S_pooled)
M_s        = M_3.reshape(*M_3.shape[:-2], S_pooled)
```

`M_s`/`slot_w` 里每个 entry 的 3 个虚拟槽拿到的是**同一个**标量（entry 级的
`M`/`w`，不随锚点变化）——这是故意的，不是偷懒广播出来的近似。mass bias
公式 `λ·log(w_s/M_s)`（§2.3）里的 `w_s`/`M_s` 描述的是"这个 entry 整体代表
了多少原始 token、这些原始 token 被这个 entry 展开成了几个虚拟槽"，两个量
的定义域都是 entry 而不是虚拟槽；3 个虚拟槽只是同一个 entry 在 3 个不同
位置上的只读投影，`w`/`M` 的份额怎么在它们之间分摊，交给 softmax 按各自的
`score` 竞争，不需要（事实上也不应该）把 `w`/`M` 人为拆成三份、分别赋给三个
虚拟槽——拆分反而是错的：`log(w/M)` 会在虚拟槽维度上被重复稀释，而且已经
展开出来的 3 个虚拟槽本来就不是三个各自独立的 1/3 个 entry，它们共享同一份
底层内容，区别只在旋转它们的锚点位置不同。

#### 虚拟槽展开必须扣上 `log_kv_slot_attention` 现有的 `causal_tail`/`mask` API

**现有 `causal_tail` 机制的假设和语义簇的虚拟槽展开不兼容，必须新增一个第三种
掩码，不能复用现有两种。** 核对 `log_kv_cache.py:1477-1545` 确认：

- `mask`（`(T_q, S)` bool）和 `causal_tail`（int）**互斥**（`causal_tail`
  非零时传 `mask` 直接 `raise ValueError`）。
- `causal_tail` 假设**最后 `causal_tail` 个 slot 是逐 token 对齐的 in-flight
  精确 chunk**（`causal_tail == T_q` 是硬校验），它之前的所有 slot **无条件
  可见**——现有实现完全没有"pooled 区域里某些 slot 无效，需要挡掉"这个概念，
  因为位置分桶方案里每个 pooled slot 永远代表真实存在的合并结果，不存在"这个
  slot 是去重后的占位、不该被 attend"的情形。
- `append_exact_tokens`（`log_kv_cache.py:1435-1474`）把 in-flight chunk
  拼在 pooled slot **之后**（`k_all = cat([slot_k, k_new], dim=2)`）——**flatten
  顺序是 pooled 在前、exact 在后，这是既有约定，语义簇路径必须原样保留**，
  否则 `causal_tail` "最后 `causal_tail` 个是 in-flight"这条假设直接失效。

语义簇路径新增的 `slot_valid`（`dedup_anchors` 的输出，标记锚点去重后哪些虚拟槽
是重复/无效的）只覆盖 **pooled 区域**——**exact in-flight chunk 从不参与锚点
展开**（它是逐 token 精确条目，每个 token 天然只有一个真实位置，不需要
`p_lo/p_mid/p_hi` 三个候选，也就没有"去重"这回事），所以 `slot_valid` 的形状是
`(B,G,S_pooled)`（`S_pooled = K_max·L_alloc·B′·3` 去重前的上界，实际展开后
flatten 成一维），**不覆盖、也不需要覆盖 exact 尾部**。

**新增第三个、与另外两个正交的掩码参数**：

> **更正（这一轮修的）：`M_s` 之前只在表 B/mass bias 公式里提过，从没
> 正式列进这个签名，容易被当成"只是内部实现细节，不需要调用方传"。**
> `M_s` 和 `slot_valid` 是同一批新增参数，**shape 与作用域也完全对齐
> `slot_valid` 的既有约定**：`(B, G, S_pooled)`，只覆盖压缩 levels 那段
> 前缀（`dedup_anchors` 产出、经同一次 flatten 展开到 `S_pooled`），不
> 覆盖 exact 后缀（recent window + `causal_tail` 覆盖的 in-flight
> chunk）——因为 exact 后缀的每一槽都是单个真实 token，天然 `w=1,
> M=1`，`log(w_s/M_s) = log(1/1) = 0`，和现有（未引入语义簇之前）对
> exact 槽的 mass bias 行为完全一致，不需要调用方为这段额外构造
> `M_s`，函数内部对 `S_pooled` 之外的位置隐式按 `M_s≡1` 处理。
>
> **更正（这一轮修的）：上一版这个签名把 `lam` 整个漏掉了。** 真实现有签名
> （`litgpt/log_kv_cache.py:1477-1497`）是 `q, slot_k, slot_v, slot_w,
> scale, mask=None, lam=1.0, causal_tail=0, slot_sigma_u=None, ...`——
> `lam`（mass bias 系数，即 `log_kv_lambda`，见 §5.1 参数表）排在 `mask`
> 和 `causal_tail` 之间，是**现有**参数，不是这次改动新增或移动的。上一版
> 只列出"不变"的 `mask`/`causal_tail` 和"新增"的 `slot_valid`/`M_s`，中间
> 漏了 `lam`，容易被读成"这个参数被顺带移除或换位置了"。`lam` 本身**不受
> 这次改动影响**——`λ·log(w_s/M_s)` 公式里的 `λ` 就是这个 `lam`，语义
> 簇路径只改了这个公式除以什么（`w_s/M_s` 而不是 `w_s`），没有改 `lam`
> 本身的传参方式，补全签名只是让这一点在这里也看得见。

```python
def log_kv_slot_attention(
    q, slot_k, slot_v, slot_w, scale,
    mask=None,             # 不变：(T_q, S) bool，仍与 causal_tail 互斥
    lam=1.0,                # 不变：mass bias 系数（log_kv_lambda），排在
                             # mask 和 causal_tail 之间，这次改动没有移动它
    causal_tail=0,          # 不变：仍要求 causal_tail == T_q
    slot_valid=None,      # 新增：(B, G, S_pooled) bool，只盖 pooled 前缀，
                           # 可以和 causal_tail 同时使用，也可以和 mask 同时使用
                           # ——它和另外两者不是同一个轴（entry 级有效性 vs
                           # query-time 因果可见性），不存在互斥关系
    M_s=None,              # 新增：(B, G, S_pooled) int，和 slot_valid 同轴、
                           # 同作用域（只覆盖 pooled 前缀）；mass bias 内部改用
                           # λ·log(w_s/M_s)，S_pooled 之外隐式 M_s≡1
    ...
):
```

**形状契约必须在函数入口显式断言，不能只在 docstring 里说一句"只覆盖
pooled 前缀"就当作调用方自然会保证。** 记 `S_total = slot_k.shape[-2]`
（`slot_k` 拼接了 pooled 前缀和 exact 后缀之后的总宽度）：

- `S_pooled ≤ S_total`——`slot_valid`/`M_s` 的宽度不能超过 `slot_k`
  实际拥有的总宽度，否则下标越界。
- `S_total − S_pooled ≥ causal_tail`——pooled 前缀之后剩下的宽度必须
  至少能装下 `causal_tail` 那批 in-flight 精确 token；等号成立时剩下
  的部分恰好全部是 in-flight chunk（`recent_count == tail_query_count`
  的情形），大于号成立时说明 recent window 里还有比 in-flight chunk
  更早、但仍在 pooled 前缀之后的精确 token（`recent_count >
  tail_query_count`），两种情形函数都要正确处理，不能假设恰好相等。
- **`w[i] == 0 ⟹ slot_valid[i] == False`（单向蕴含，不是等价）**：
  `dedup_anchors` 保证无效 entry（`w=0`）的三个槽 `slot_valid` 全部
  是 `False`（§5.14），但反过来不成立——**有效** entry（`w>0`）内部
  重复的锚点槽同样 `slot_valid=False`，这是去重的正常结果，不是
  "该 entry 无效"的信号，调用方/单测不能把 `slot_valid=False` 误读成
  `w=0` 的充分条件。
- **exact 后缀（`S_pooled` 之后的全部位置，含 recent window 和
  in-flight chunk）由调用方保证 `slot_w` 恒为 `1`**——这不是
  `log_kv_slot_attention` 内部校验的东西（它只是隐式按 `M_s≡1` 处理
  `S_pooled` 之外的 mass bias，不反过来检查 `w` 的取值），但
  `get_attention_state()` 的构造本身保证了这一点（recent window 每个
  槽是单个真实 token，见 `litgpt/log_kv_cache.py:1246-1336` 的
  `slot_w`赋值），调用方不需要、也不应该为这段额外传 `w≠1` 的值。

**应用方式**：`slot_valid` 是**逐 slot、不随 query 变化**的一维掩码（不像
`mask` 是 `(T_q,S)`），所以可以用一次广播 `masked_fill_` 覆盖 score 张量的
pooled 列（`score[..., :S_pooled]`），代价和 `causal_tail` 现有的"只填 tail
切片"同一量级——不需要构造一个 `(T_q, S)` 的全尺寸掩码，`slot_valid` 本身已经
是比 `mask` 更便宜的表示。应用顺序上，`slot_valid` 的 `masked_fill_` 和
`causal_tail`/`mask` 的 `masked_fill_`互不依赖，谁先谁后不影响最终结果（都是
把对应位置置 `-inf`，结合律成立），实现时可以按顺手的顺序各做一次。

**必须补的单测**：构造一个同时有 pooled 无效槽（需要 `slot_valid` 遮住）和
in-flight exact chunk（需要 `causal_tail`）的合成场景，断言两种掩码同时生效
——pooled 区域的无效槽在所有 query 上都拿不到注意力权重，exact 区域仍然正确
respects 逐 token 因果关系，且这条路径下的输出与"手工构造等价的 `(T_q,S)`
`mask`（同时编码两种约束）"数值一致，验证"两个正交掩码分别加"和"揉成一个
掩码"是同一件事，只是前者更便宜。

#### `slot_valid`/`M_s` 在语义模式下不是可选项；`get_attention_state()` 的返回结构需要重新设计

**上面的签名把 `slot_valid`/`M_s` 写成 `=None` 默认值，这只回答了"传了之后
形状/掩码怎么用"，没有回答"什么时候必须传"。这不是无关紧要的措辞缺口：如果
调用方在语义模式下漏传（`slot_valid=None`），函数不会报错，只会静默按"没有
无效槽"处理。** 后果比听起来更糟，且和 `λ` 是否为 0 强相关：

- **`λ≠0` 时**，靠 `w=0` entry 的 `log(w/M)=log(0)=-inf` 能顺带压掉**纯 pad/
  dead entry**（因为它们的 `w` 本身就是 0），但压不掉**有效 entry 内部的重复
  锚点**（`p_lo==p_mid` 等，§5.14 `dedup_anchors` docstring）——这类槽的 `w>0`
  （和它没被去重的兄弟槽共享同一个 `w`），`log(w/M)` 不为 `-inf`。漏传
  `slot_valid` 会让这些重复锚点被当成**独立的额外证据**，把该 entry 的
  softmax 质量按 `M` 倍放大（`M∈{1,2,3}`），而不是均摊——`M_s` 存在的意义
  正是防止这个放大，`slot_valid` 缺失时它形同虚设。
- **`λ=0` 时**（`log_kv_slot_attention` 文档里明确列出的消融旋钮，"built-in
  ∝1/w long-range forgetting curve"）更严重：mass bias 这一项**根本不会被
  加**（现有代码 `if lam != 0.0:` 门控，`log_kv_cache.py:1625-1628`），连
  "纯 pad/dead entry 靠 `log(0)=-inf` 被动压掉"这条安全网也不存在了。此时
  `slot_valid` 是**唯一**挡住无效槽（含锚点=0 的哨兵位置）获得非零 attention
  的机制，缺了它不是"退化成稍差的近似"，是"pad/dead entry 的垃圾内容混进
  softmax"。

**修法：`slot_valid`/`M_s` 在语义模式下由 `get_attention_state()` 自动、无条件
产出，不是调用方按需申请的可选项。** 判断依据是 cache 自身的构造模式
（`log_kv_semantic_clusters=True`），不是某个新增的调用参数——是否需要
`slot_valid`/`M_s` 完全由 cache 是不是语义簇模式决定，调用方没有"要不要"的
选择权，也就不存在"忘了传"这种调用方过失（只要 `get_attention_state()` 自己
写对）。

> **更正（这一轮修的）：上一版"`S_pooled=0` 时是空张量"这句话和 §5.13"读出
> 不需要 gather"一节的既有设计直接矛盾，必须收回。** §5.13 明确"entry 躺在
> 固定位置，`get_attention_state()` 只要 `reshape` 成
> `(B,G,K_max·L_alloc·B′,·)`"，§5.14"per-entry→per-virtual-slot 展开"一节
> 据此把 `S_pooled` 定义为 `w.shape[-1]*3`，其中 `w` 就是这个 reshape 出来的
> **固定宽度** `K_max·L_alloc·B′` 张量——这是"矩形预分配"这条贯穿全篇的架构
> 决定的直接推论：entry 存储从不随实际占用量收缩或增长，`w=0` 的槽（从未写入
> 过或已被 §5.11 显式 pad）和 `w>0` 的真实内容槽躺在同一个固定形状的张量里，
> 靠 `w>0`/`slot_valid` 掩码区分，不靠改变张量宽度区分。**`S_pooled` 因此是
> 一个只由 `K_max/L_alloc/B′` 决定的编译期常量，恒等于
> `K_max·L_alloc·B′·3`，不随"是否已经 flush 过""pooled 里有多少真实内容"
> 变化——序列刚开始、一次 flush 都还没发生时，`S_pooled` 依然是这个满值，
> 只是这满值里的每一个槽此刻都 `w=0`，`slot_valid` 因此在整个 `S_pooled`
> 宽度上恒为 `False`，不是张量本身缩成宽度 0。** 唯一让 `S_pooled` 变成 0 的
> 情形是 `K_max=0` 或 `L_alloc=0` 或 `B′=0`（配置错误，不是运行时状态）。
>
> **这个更正带来一个必须显式处理的后果：pooled 区域整体无效时，`softmax`
> 的这一段分数会整体是 `-inf`，调用方不能假设"总有几个 pooled 槽有效"。**
> 但这不会导致某个 query 的整行分数全 `-inf`（那才是真正危险的、会让 softmax
> 产出 NaN 的情形）——`causal_tail`/recent window 覆盖的 exact 后缀**不受
> `slot_valid` 约束**（§5.14"新增第三个掩码参数"一节：`slot_valid` 只覆盖
> pooled 前缀），且任何走过 `log_kv_chunk_attention`/`LogKVStreamTraining
> Attention.forward()` 的 query 位置至少能因果地看到它自己所在的 in-flight
> chunk，这部分从不被掩码，所以每一行分数向量恒有至少一个有限值，softmax
> 恒良定义。这条不变量值得写进单测：构造一个"pooled 区域整体无效（序列刚
> 开始）"的合成场景，断言 softmax 输出不含 NaN/Inf，且数值上等价于"pooled
> 区域根本不存在，只对 exact 后缀做 attention"。

**位置分桶（legacy）模式永远不产出 `slot_valid`/`M_s`**（值恒为 `None`）——
它从不做锚点展开，没有"重复/无效虚拟槽"这个概念，`None` 在这条路径上是合法
的稳定值，不是"忘了实现"的占位符。

**这个决定顺带暴露了一个更大的接口问题，必须一并解决**：`get_attention_
state()` 现有实现（`log_kv_cache.py:1246-1369`）按 `with_stats` 布尔值返回
**裸 3-元组或 8-元组**，位置对应关系全靠调用方记住。语义模式再叠加
`slot_valid`/`M_s`，理论上会衍生出 **4 种不同长度/顺序的返回值**（legacy×
无 stats=3、legacy×有 stats=8、语义×无 stats=5、语义×有 stats=10，且"语义×
有 stats"这一档此前完全没人定义过顺序），而这份规格通篇反复在抓的正是"位置
参数摆错顺序"这一类 bug（`experiments.md` S0.8 3b 一节的调用签名错误就是活
生生的例子）
——继续用裸位置元组只会不断制造同一类风险。**不再扩展位置元组，改用一个
具名结构**：

```python
class CacheAttentionState(NamedTuple):
    slot_k: torch.Tensor
    slot_v: torch.Tensor
    slot_w: torch.Tensor
    # 语义模式恒为张量（S_pooled 可以是 0 但不是 None）；legacy 模式恒为 None：
    slot_valid: torch.Tensor | None = None
    M_s: torch.Tensor | None = None
    # with_stats=True 时是张量；with_stats=False 时恒为 None，不区分 legacy/语义：
    slot_sigma_u: torch.Tensor | None = None
    slot_sigma2: torch.Tensor | None = None
    slot_gamma_a: torch.Tensor | None = None
    slot_gamma_b: torch.Tensor | None = None
    slot_gamma: torch.Tensor | None = None
```

`get_attention_state(with_stats=False)` 在任何模式下都返回同一个类型
`CacheAttentionState`，只是不适用的字段取 `None`——调用方一律按**字段名**
取值（`state.slot_k`、`state.slot_valid`），不再按位置解包，这一类改动本身
就让"第 4 个位置传成第 5 个参数"这整个 bug 类别在语法层面消失，不需要调用方
自觉小心。`with_stats` 参数保留（性能旋钮，`second_order_scale==0` 的调用
方仍可以传 `False` 跳过 5 个 rank-1 张量的 gather/cat），不因为这次改动被
取消。`log_kv_slot_attention` 自身的参数列表不变（仍是关键字参数，不接收
这个结构体本身），调用方从 `state` 上取字段后按现有方式逐个传入。

§5.15"`get_attention_state()` 额外返回有效位掩码与每 entry 的 `M_s`"一节
（原文只有一句话）改为指向本节这个具体结构。

**为什么 Σ/Γ 也要按锚点转，不能只转 `k_raw`。** §5.10/§5.20-B 说"Σ/Γ 的统计空间
post-RoPE → pre-RoPE，数学不变，只是喂进去的张量换了"——这句话覆盖了**累积**这一步
（Chan merge 在 pre-RoPE 空间做，正确），但没覆盖**读出**这一步。现有打分/读出公式
（`log_kv_cache.py:1500-1501`）是：

```
score_s = scale·(q · k_s) + 0.5·scale²·sigma2_s·(q · sigma_u_s)² + λ·log(w_s)
read_s  = v_s + scale·gamma_s·(q · gamma_a_s)·gamma_b_s
```

`q` 是 post-RoPE（query 从来都是），如果 `sigma_u_s`/`gamma_a_s` 现在存的是
pre-RoPE 方向却直接拿去和 post-RoPE 的 `q` 做点积，两边活在不同的坐标系里，点积
的值没有意义——**这不是精度损失，是算错了坐标系**，和当初"pre-RoPE k 不能直接喂
attention"是同一类错误。`gamma_b`（value 空间方向，value 从不 RoPE）和 `sigma2`/
`gamma`（标量特征值，没有方向）不受影响，只有 `sigma_u`/`gamma_a` 这两个"key 空间
方向"字段需要物化。

**做法和 `k_raw` 完全对称，复用同一个旋转原语**（上面 `_rotate_at_anchors`）：每个
entry 展开出 `M` 个虚拟槽时，`sigma_u`/`gamma_a` 也各自展开出 `M` 个版本
（`materialize_anchor_directions`），和 `k_eff_a` 用**同一个锚点、同一次旋转**——
数学上这是合法的，因为旋转是线性映射，一个方向向量在 pre-RoPE 空间代表的协方差
主轴，转到某个确定位置的 post-RoPE 空间后仍然是那个位置上的协方差主轴，和
`k_raw → k_eff_a` 的道理一模一样。**不新增任何持久存储**——`sigma_u`/`gamma_a`
仍然是每 entry 存一份（pre-RoPE），`M` 份只在读出时瞬时物化，和 `k_eff` 同样是
读出槽池的一部分，不是 cache 内存的一部分（§4 那两笔账的区分在这里依然适用）。

**单测（S0.1，纯 CPU，不依赖 dump）**：
- **Σ/Γ 锚点物化的嵌套精确性**：单 token entry（`M=1`）时，`materialize_anchor_
  directions` 对 `sigma_u`/`gamma_a` 的旋转应与直接对该 token 位置做
  `model.apply_rope` 逐位一致——和 `materialize_anchor_keys` 的嵌套精确性单测
  结构完全对称，因为底层是同一个 `_rotate_at_anchors`。
- **坐标系回归**：构造一个简单二阶修正非零的合成 entry，断言"score 里的二阶项
  用旋转后的 `sigma_u_eff` 算"和"用未旋转的 `sigma_u` 直接点乘 `q`"两者**不相等**
  （只要该 entry 的锚点不在原点）——这条测的是"没有人漏转"，而不是"转得准不准"，
  是防回归最便宜的一条。

**S0.8 3b（`experiments.md`"S0.8"一节）明确只走 `with_stats=False` 的一阶路径，
不覆盖这里的 Σ/Γ 二阶修正。** 3b 要验证的是"批量近似路由/compaction 与严格
串行参考构造出的 cache，attention 读出是否一致"——这个问题只关心**槽的成员
划分和位置表示**（哪些 token 进了哪个槽、entry 的 `p_lo/p_hi/sum_wp`），和
Σ/Γ 的旋转数学是两件正交的事：Σ/Γ 的正确性已经由本节这两条单测独立覆盖
（嵌套精确性、坐标系回归，纯 CPU、不需要批量/串行两条路径对拍）。若 3b 也
把 `with_stats=True`、`slot_sigma_u/sigma2/gamma_*` 一并纳入比较，一是要
在 §5.14"`dedup_anchors`/`materialize_anchor_keys` 的输出到
`log_kv_slot_attention` 输入之间还缺一步展开"那节的基础上再定义五个 rank-1 字段
各自的展开/mask/`CacheAttentionState` 字段顺序（`materialize_anchor_
directions` 对 `sigma_u`/`gamma_a` 同样要展开成 3 个虚拟槽、同样要被
`slot_valid` 遮蔽——`gamma_b`/`sigma2`/`gamma` 不经过旋转但仍需要按同一个
`slot_valid` 广播/遮蔽，和 `slot_w`/`M_s` 是同一类"entry 级、不随锚点变化"
的量），二是会让 `error_3b` 超标时无法区分"是路由分歧导致的，还是二阶修正
本身的近似误差导致的"——把两个独立误差源混进同一个数字，违背了 3b 存在的
本意（诊断路由/compaction 层面的分歧）。**决定：3b 固定
`get_attention_state(with_stats=False)`，第一次实现时不需要考虑
`slot_sigma_u`/`slot_gamma_*` 的展开顺序；如果日后需要验证"批量近似路由 +
二阶修正"组合起来是否也保持一致，那是一个独立的、需要新起一项的实验，不是
往 3b 里加参数。**

**这个范围决定留下一个必须显式接住的后果（这一轮补的）**：3b 只保证 rank-1
Σ/Γ 旋转数学正确（本节两条单测），从没有任何测试覆盖过"批量近似路由/
compaction 下，Σ/Γ 的聚合状态本身（`sigma_u`/`sigma2`/`gamma_a`/
`gamma_b`/`gamma` 经 `compact()`/`_binary_carry()` 的 Chan-merge 累积出的
值）是否也和严格串行参考一致"——Σ/Γ 的 Chan-merge 依赖的正是 3b 在测的
那同一套路由/compaction 结构（哪些 token 被合并进哪个 entry），3b 已经
证明的"路由分歧 <5%"不能自动推出"Σ/Γ 聚合分歧也 <5%"，这是一个未经
验证的推论，不是已经覆盖的情形。§7 消融表"rank-1 Σ/Γ"一行把"关/现有
构造/delta-rule 构造"列为独立扫描轴，"现有构造"这一档一旦在 Stage 2 跑
eval，就是在没有类似 3b 这样的批量 vs 严格串行验证的情况下，把一条完全
未经验证的数值路径喂进真实模型输出——不能假设它"大概率也没事"。

**决定：Stage 1 的语义簇初次实现里，`second_order_scale` 默认固定为 0**
（等价于 §7 消融表"rank-1 Σ/Γ"一行的"关"这一档），和 v1 不碰训练目标
（CLAUDE.md §10 pin 系列死因清单、本文档"训练梯度"一节）是同一类范围
决定——先把能验证的部分（一阶路由/compaction，3b 覆盖）钉实，再决定要不要
扩大范围，而不是在没有验证手段的地方直接打开开关。**"现有构造"这一档要
在 Stage 2 跑之前，必须先有一个类似 3b、专门针对 `with_stats=True` 路径
的独立验证（可以叫 3c，但这里不展开定义它的具体做法——3b 现在已经足够复杂，
提前把 3c 的细节也定死容易在 Σ/Γ 尚未启用时就锁死一个可能不合适的设计；
真正要开这个口子时再照着 3b 的模式单独设计）**，在那之前"rank-1 Σ/Γ"这一行
的消融只能停留在"关"这一档，"现有构造"/"delta-rule 构造"两档不能跑。
- **嵌套精确性**：单 token entry（`p_lo=p_hi=p_mid`）→ M=1，输出与 `model.apply_rope`
  在该位置上逐位一致。
- **合并的结合律**：任意二叉合并顺序**逐位一致**——`sum_wp` 是整数加法，这条应当
  精确成立而非近似。**这正是旧的 `p_mid` 继承规则挂掉的地方（§2.2 的更正框），
  所以三个等权成员的两种结合顺序必须作为定向回归用例写进去。**
- **`p_lo`/`p_hi` 永远真实**：随机构造合并树，断言这两个锚点都属于原始成员位置
  集合（`p_mid` 是质心，**不**满足这条，别误写进断言）。
- **`p_mid` 落在区间内**：`p_lo ≤ p_mid ≤ p_hi` 恒成立（clamp 之后）。
- **舍入规则**：`p_mid` 必须用 **round-half-up 的整数实现** `(2·sum_wp + w) // (2·w)`，
  单测要覆盖 `w=2` 且两成员位置和为奇数的情形（此时真值恰好是 `k+0.5`）。

> **为什么不是 `floor`，也不是浮点 `round`**：
> - `floor` 对**所有**非整数结果都向下取，系统性左偏约 0.5 个位置；而 `round` 只在
>   恰好平局时才有 ±0.5 的偏差。平局并不罕见——`w=2` 的 entry 有约一半会遇到。
> - 浮点 `round` 在 1M 上下文下不安全：`sum_wp` 量程到 `10¹²`，**超出 fp32 的精确
>   整数范围**（`2²⁴≈1.7e7`），必须 fp64 才不丢位。
> - 更关键的是 **§11-A 的重放要求逐位可复现**。整数运算天然满足；浮点除法在不同后端/
>   不同 kernel 下末位可能不同，而 `p_mid` 的一位之差会改变锚点、改变去重后的 `M`、
>   进而改变 mass bias。**所以这里必须是整数算术，不是"用整数比较快"的问题。**
> - 平局取上（偏向 `p_hi`）是个约定：entry 内更晚的成员更"新鲜"。`2·sum_wp` 在
>   int64 下最大约 `2×10¹²`，不会溢出。
- **mass bias 的计数守恒**（§2.3 那个 bug 的回归测试，**必须有**）。注意断言要写对，
  下面两条是不同强度的命题：
  - **可以断言（精确，不依赖 score）**：M 个虚拟槽的计数因子之和等于单槽的，即
    `Σ_a (w/M)^λ = M·(w/M)^λ`，在 `λ=1` 时精确等于 `w`。**这才是 `/M` 强制的不变量**，
    去掉 `/M` 会得到 `M·w`，正好被这条抓住。
  - **只在等 logit 下成立**：M 槽的 softmax 总质量 `= (w/M)^λ · Σ_a exp(s_a)` 等于
    单槽的 `w^λ·exp(s)`，**需要所有 `s_a` 相等且 `λ=1`**（`λ≠1` 时还差一个
    `M^(1-λ)`）。测这条必须先把 M 个锚点强制取同一位置（或旁路 RoPE）使 logit 相等。

  > **不要断言一般情况下的 softmax 质量恒等。** 不同锚点的 `k_eff_a` 不同 ⇒ `s_a`
  > 不同 ⇒ 总质量本来就会变——**这正是锚点展开的目的**（位置敏感的检索靠它实现），
  > 不是需要被修掉的偏差。
- **`w=0` 填充是恒等元**：`merge(A, ∅) == A`，**覆盖完整统计元组，不只是均值和锚点**
  ——`sigma_u/sigma2/gamma_a/gamma_b/gamma` 也要断言逐位相等，理由和填充槽字段必须
  全零初始化的硬性规则见 §5.11 那段推导（0 × 有限数才安全，0 × NaN/Inf 不安全）。
- **v2 对照**：保留 z 统计量实现与其 Dirichlet 闭式解单测，供 §8 消融使用。

**这条 v2 对照单测同时是"新代码复现旧数学"的正式正确性检验，不是可有可无的消融
配件。** 在 `anchor_mode=z` 且 gain 强制取 honest-decay（即 v2 的 β=1 分支，`gain≡1`）
的合成序列上，验证物化出的 pooled key 与手算的 Dirichlet 闭式解 `sin(wθ/2)/(w·sin(θ/2))`
逐位一致——这就是现有 LogKV 的位置数学，一位不差。**这条测试只需要几个 token 的合成
数据，不需要 K_max=1、不需要 eval、不需要 GPU。**

### 5.15 `litgpt/log_kv_cache.py`

- 新增 §5.13 的全部 buffer。
- 键改存 **pre-RoPE**。recent window 同样存 pre-RoPE + 位置，读出时统一走
  `materialize_anchor_keys`——**一条代码路径**，保证嵌套性质在生产里被真实执行。
- `compact()`/`_binary_carry()` 增加锚点合并（一行 `merge_anchors`），Σ/Γ 改在
  pre-RoPE 内容空间统计（§5.10）；`_counts[ell]` 的"同层等宽"假设要放开（§5.19-2）。
- 路由 `_route()`：按 §5.2–§5.6 实现，`@torch.no_grad()`，与 cache 更新同路径。
  **必须同时把路由决策写进 `op_log` 供 backward 重放**（§11-A、§5.21-2，硬性要求）。
- `get_attention_state()` 返回类型改为 `CacheAttentionState`（具名结构，见上方
  "`slot_valid`/`M_s` 在语义模式下不是可选项"一节），不再是裸位置元组；语义模式下
  无条件（不受调用方控制）额外产出 `slot_valid`/`M_s` 两个字段；
  `log_kv_slot_attention` 增加可选槽有效性掩码参数（fp32 分数上填 `-inf`），与现有
  `causal_tail` 正交；**mass bias 改用 `λ·log(w_s / M_s)`**。

### 5.16 `litgpt/model.py`

核心的接口性改动：cache 现在需要 **pre-RoPE 的 k** 和该 token 的**绝对位置索引**，
模型本来就持有这两样，所以这部分是传参而非新计算；`build_log_kv_cache`/
`set_log_kv_cache`/`enable_log_kv_training` 三处透传；读出侧需要访问 RoPE cache
做锚点物化。

> **但这远不是全部改动面——不要按"多传两个参数"来估工作量。** 完整清单在 §5.21-1：
> `q` 仍需 post-RoPE，所以要**同时持有 `k_roped` 和 `k_raw`**（现有代码把两者写进同一个
> `k` 变量）；training 与 inference 是**两个不同的调用点**；`append_exact_tokens` 的
> in-flight chunk 必须用 post-RoPE 且与 `w=1` entry 物化出的键**逐位一致**；
> `_log_kv_pending` 挂起的元组要带 `k_raw`；`model.py:805-808` 那段 expected-RoPE 注释
> 描述的正是被替换掉的机制，必须同步改写。

### 5.17 矩形张量与 batching

- **按 (B, G) 独立聚类**——不同头关注不同语义，共享簇是错的（§2.5）。
- **预分配 + 掩码**：entry 张量固定为 `K_max × L_alloc × B′`（**不含 `SEG_max`**，
  见 §5.11），占用不满时掩码屏蔽。张量始终矩形，不引入 ragged 布局。
- 读出时的虚拟槽池为 `entry 数 × M_max`，是瞬时张量，不是持久 cache（§4）。
- ragged/paged 布局（以及 §4 提到的 shared block pool）留到工程化阶段。

### 5.18 实现顺序（每一步都有可验证的中间态）

0. **Stage 0 的 dump 脚本**（规格见 experiments.md）——它是所有 Stage 0 结论的输入，
   排在生产代码之前（§5.21-5）。S0.0–S0.7 和 S0.8 的第 1/2/3a 项只需要这一步。
1. **`log_kv_position.py` + 单测**（纯 CPU，不依赖任何 dump）。**S0.8 3b 的前置项**。
2. **CPU 参考实现的路由**（朴素串行版，慢但正确）——它同时是 S0.8 的对照基准，
   不要跳过。**连同 `log_kv_slot_attention()`/`get_attention_state()` 按
   §5.14/§5.20-B 扩展出 `CacheAttentionState`/`slot_valid`/`M_s`——这是一次
   breaking 的 API 迁移（不是"纯加法式改动"，见下方"更正"框与 §5.20-B 调用点
   迁移清单最后一行），需要在同一次改动里原子迁移全部约 30 处调用点（生产代码
   6 处 + 测试约 25 处），但改动本身是机械的位置解包换字段访问、不涉及新算法，
   是 S0.8 3b 唯一需要在第 0 步之外补的前置代码**，见 §5.21-5 的更正框；3b 之外
   的 S0.0–S0.7 不依赖这一步。

   > **更正（这一轮修的）：`CacheAttentionState` 不是"纯加法式改动"。** §5.14
   > 定案"`get_attention_state(with_stats=False)` 在任何模式下都返回同一个
   > 类型 `CacheAttentionState`"（恒定 10 个字段，不用的字段填 `None`）——这
   > 意味着现有 `slot_k, slot_v, slot_w = c.get_attention_state(with_stats=
   > False)` 这类 3-变量位置解包（`log_kv_cache.py:1738` 现状）、以及
   > `with_stats=True` 的 8-变量解包（`model.py:1200`、
   > `tests/test_log_kv_cache.py:1096` 现状），在返回类型改成 10-元组之后
   > 会直接 `ValueError: too many values to unpack`——**不论 legacy 还是
   > 语义模式，因为这个返回类型改动对两种模式都生效，不受 `log_kv_
   > semantic_clusters` 开关裹住**。§5.19-1 的"默认关闭字节等价"CI 闸门管
   > 的是**数值**在开关关闭时是否不变，管不到**函数签名/返回元组长度**变了
   > 导致调用方在 Python 层面直接报错这类问题——上一版拿这条 CI 闸门当这次
   > 改动的安全网是范畴错误，两件事正交。正确的定位是：这是一次**必须原子
   > 完成的 breaking 迁移**（`get_attention_state()`/`append_exact_tokens()`
   > 的定义与全部约 30 个调用点在同一个改动里一起改，见 §5.20-B 调用点迁移
   > 清单最后一行），但迁移动作本身是机械的（位置解包 → 按字段名取值，纯
   > 语法层面的重写，不涉及任何新算法决策），所以仍然是"是否要投入 Stage 1"
   > 这道门不需要关心的小额、低风险前置工作，只是"低风险"不等于"不改变
   > 调用方代码"，措辞必须精确。
3. **`K_max=1` 单簇路径**：验证退化到"单条位置序 ladder"。**不要求、也不预期**
   与现有 LogKV 数值对齐（下方"更正"框），当消融参考点看待；这一步真正的正确性
   检验是 `anchor_mode=z` 的 CPU 单测（同一个更正框末尾）。
4. **多簇路由 + 向量化**（§5.4 三阶段），拿第 2 步的参考实现测分歧率。
5. **段对齐填充**（§5.11）+ `level_count` 簿记改造。
6. **训练路径**：`op_log` 的保存与重放（§11-A、§5.21-2）。

> **`K_max=1` 的 eval-time 数字不是正确性闸门，即使实现完全正确也不会复现旧数字。**
> 原因是 §2.1 的 ladder level 0 从"2-token 合并"改成"单 token（`w=1`）"——这是设计里
> 明说的**不可选、承重**的改动（§3 needle 论证、§5.9 都靠它），跟 `K_max` 无关。所以
> `K_max=1` 只是把**簇轴**退化成 1 条 ladder，**层轴**（level 0 的宽度）仍然是新的、
> 更细的。真正验证"新代码复现旧数学"的是上面那条 `anchor_mode=z` 的 CPU 单测——它
> 绕开层轴差异，只测位置公式本身。`K_max=1` 的 eval 数字请当**消融参考点**看待，
> 不要设容差、不要拿它当 pass/fail（§6 Stage 2、§12-E）。

### 5.19 实现 tips 与易错点

1. **默认关闭必须逐字节等价。** 照 `importance_pooling` 的先例，用 `torch.equal`
   对比开关前后的 slot tensors 作为 CI 测试。这是所有后续对比实验的地基。

2. **`_counts[ell]` 的"同层等宽"假设要在两处放开**——对齐填充（§5.11）和顶层饱和
   （§5.12）。**这是整个实现里最集中的风险点**，建议一开始就把 level 簿记设计成
   "每 (簇, 层) 存 (count, 含填充数)"，而不是先按等宽写完再回来补。

3. **`w=0` 填充槽必须显式掩码，且填充槽的每个字段都必须显式清零。** `compact` 里
   的 `w_total.clamp(min=1e-8)` 会让两个空槽合并出 `alpha=0` → `k_out=kb`
   （垃圾值），需要在掩码层拦掉，不能指望它自然消失。**更隐蔽的一条**：Σ/Γ 的
   rank-1 合并路径会用 `frac_b`/`cross_frac`（`wb=0` 时精确为 0）去乘填充槽的
   `sigma_u/sigma2/gamma_*` 和 `k−k_pad`——`0×有限数=0` 但 `0×NaN=NaN`，
   所以填充槽如果只清零了 `w` 和锚点、其余字段是未初始化内存，一次看似安全的
   `w=0` 合并会把 NaN 传染进真实槽（详见 §5.11、§5.14）。

4. **锚点哨兵值会越界。** 空槽用 `p_lo=+INT_MAX, p_hi=-1` 让 min/max 自动正确，
   但 `cos_cache[anchors]` 会因此索引越界。物化前必须 `clamp(0, N-1)` 或先按有效位
   掩码筛掉——**这是最容易在长序列上才暴露的崩溃**。

5. **`op_log`（连同 `op_log_len`）要在 `forward()` 结束时存进 `ctx`，且
   `reset_parameters()` 对它俩必须是"重新绑定成全新张量"，不能是像其它
   buffer 那样的"原地 `zero_()`"。** `LogKVStreamTrainingAttention.forward`
   开头就调 `cache.reset_parameters()`，而 backward 要重放 forward 期间记录的
   路由；**定案是把 `op_log`/`op_log_len` 和 `q`/`k_raw`/`k_roped`/`v` 一起存
   进 `ctx`**（§5.21-2 的"`op_log` 的有效长度"一节有完整推导和理由）。
   **这确实是 cache 对象生命周期里的一个特例**——其它 buffer（ladder/
   centroid/`level_count`/`alive`/...）继续用原地 `zero_()` 复用；只有
   `op_log`/`op_log_len` 改成每次 `forward()` 开头重新分配一块全新存储、
   把属性名重新指过去，**不能沿用"原地清零复用"这条其它 buffer 都在用的
   规则**——原地清零会让 `ctx` 里存的（不管是不是克隆过）内容跟着被清空，
   这正是训练会静默用错误梯度的根源，也是曾经在这里踩过的坑（完整推导和
   "为什么不能直接克隆代替重新分配"见 §5.21-2"训练峰值显存"更正框）。
   backward 只读 `ctx` 里这份专属分配的张量，不需要关心 cache 对象后续
   发生了什么。**这个时序冲突不处理的话，训练会静默地用错误的梯度。**

6. **路由的距离计算强制 fp32。** §11-A 的重放确定性依赖它；TF32 下 einsum 的归约
   顺序差异足以在阈值边界上翻转一个决策。

7. **Ward 合并的 `(K,K)` 距离矩阵要屏蔽对角线和 dead 槽位**，否则 `argmin` 会选到
   自己或空槽。

8. **`s_h` 要在 `reset_parameters()` 里初始化成一个合理常数**，否则第一条序列开头的
   阈值是垃圾，而 §5.6 说过早期路由错误不可恢复。

9. **softmax 行至少要有一个有效槽**，否则全 `-inf` 会产生 NaN。簇很少时大部分槽被
   掩掉，这个前置条件比现在更容易被违反——现有代码把它写在 docstring 的
   preconditions 里，改动后要重新确认。

10. **先测 `K_max=1` 再测多簇。** 单簇路径把路由完全短路，能把"位置表示对不对"和
    "聚类对不对"这两类 bug 分开——否则一起上，出问题无法归因。

### 5.20 与现有实现的复用边界（哪些一行不用改）

这一节的目的是防止实现时**过度改写已经能用的部分**。被替换的只有两个设计决定——
**槽成员划分规则**和**位置表示**；之前做的数值基础设施全部是资产。

#### A. 完全复用，一行不改

| 组件 | 为什么能直接用 |
|---|---|
| `compact()` 的加权均值 | 按 `w` 加权、不要求两侧等宽；且已核实配对是**时间序相邻**的（§5.7）|
| Chan-style 二阶矩合并 + rank-1 截断 | 纯代数，与内容语义无关。`_pair_rank1_stats`、`_dominant_eigvec_small`、`_rank1_psd_from_factors`、`_rank1_cross_from_factors` 全部原样 |
| `log_kv_slot_attention` 的打分/读出结构 | 公式骨架 `score = 点积 + 二阶项 + λ·log(质量因子)`、`read = v̄ + scale·γ(q·γa)·γb` 不变；**但质量因子从 `w` 变成 `w/M`，这个改动记在下表 B，不要以为这行说的是「连质量因子也不变」** |
| GQA 的 rf 折叠、fp32 分数缓冲、`causal_tail` | 与压缩机制正交 |
| `LogKVStreamTrainingAttention` 的流式重放**框架** | 骨架、内存论证、per-block 梯度正确性论证全部不变（但重放的**依据**要换，见 C）|

**一个顺带的统一**：Ward 合并代价 `(n_a n_b)/(n_a+n_b)·‖μ_a−μ_b‖²` 正是
`_pair_rank1_stats` 已经在算的那个量的迹。**聚类要的距离、簇合并要的代价、二阶修正
要吸收的残差，是同一个量**，不需要引入任何新的数值原语（§5.2）。

> **`_binary_carry` 不属于这张表——它是核心控制流重写，不是"一行不改"。** 只有
> **合并算子本身**（"level 空放下、非空 compact 后带着翻倍宽度上浮"这套算术）复用；
> **驱动它的控制流**（现在靠 `_counts` 的 host 端镜像做 Python 标量判断）必须删掉，
> 换成对全部 (B,G,K) 的掩码并行 carry（§5.21-3）。放进下表 B，不放这里。

#### B. 需要改动（按改动量排序）

| 组件 | 改动 |
|---|---|
| `_binary_carry()` 的驱动逻辑 | **核心控制流重写，不是增量改动**。合并算子本身复用，但驱动它的 `_counts` host 镜像 + Python 标量分支必须整体替换成按 level 静态循环、对 (B,G,K) 掩码并行的向量化 carry（§5.21-3）|
| `_counts[ell]` | 全局标量 → **删除**（不是扩展成 `(B,G,K,L)`）。host 镜像的存在理由是避免 GPU sync，语义簇路径下每个 (batch,头,簇) 独立进位，同步点会变成每 flush 一次，必须走 §5.21-3 的向量化方案 |
| Σ/Γ 的统计空间 | post-RoPE → pre-RoPE。**累积数学不变**，只是喂进去的张量换了；**但读出侧必须新增一步**——`sigma_u`/`gamma_a` 现在是 pre-RoPE 方向，不能直接和 post-RoPE 的 `q` 点积，要走 §5.14 的 `materialize_anchor_directions`，和 `k_raw→k_eff` 对称展开成 `M` 份（§5.14 那段"为什么 Σ/Γ 也要转"）|
| `compact()` 签名 | 多带 `(p_lo, p_hi, sum_wp)` 走 `merge_anchors`，一行 |
| `n_c` | 拆成 `n_eff`（centroid 混合，`γ` 衰减）和 `n_total`（Ward 代价 + §5.8 空间界，单调不减）——原来单个 `n_c` 两处混用会让 Ward 误判长历史簇是"小簇"（§5.5/§5.6 的更正框）|
| mass bias | `λ·log(w)` → `λ·log(w/M)`（§2.3，必须做的正确性修正）|
| `get_attention_state()` 返回类型 | **这一轮更正**：不是"多返回一个字段"这么简单——返回类型从裸位置元组改成 §5.14"slot_valid/M_s 在语义模式下不是可选项"一节新增的 `CacheAttentionState`（具名结构），语义模式下无条件带上 `slot_valid`（entry 级有效位掩码去重后展开到 per-virtual-slot）与 `M_s`（同样是 per-virtual-slot，不是"每 entry 一个"字面意义上的粒度，值在同一 entry 的 3 个虚拟槽间相同，见 §5.14"per-entry→per-virtual-slot 展开"一节） |
| `get_attention_state()`/`append_exact_tokens()` 的调用点 | **不只是这两个函数自己的定义要改，所有消费它们返回值的调用点都要跟着从位置解包换成按字段取值**，见下面单独一行的完整清单 |
| cache 入口 | 收 pre-RoPE k + 绝对位置，而不是 post-RoPE k |
| `level_w`/entry `w` 的 dtype | 不能继承 activation dtype（现有 `log_kv_cache.py:346-349` 是 `torch.zeros(..., dtype=dtype)`，跟着 fp16/bf16 走）。fp16 整数精确表示上限是 2048、溢出上限 65504；1M 上下文下一个高冗余大簇的 `w` 可以到几十万，**必须 fp32 或 int32**，`log(w/M)` 之前再转 fp32 |

**`get_attention_state()`/`append_exact_tokens()` 调用点迁移清单（这一轮补的，
`CacheAttentionState` 落地时必须机械过一遍，不是自然会跟着改）**——核对当前
代码库确认的完整调用点列表，都要从"位置解包成 `slot_k, slot_v, slot_w[,
sigma_u, ...] = ...`"换成"接住一个 `CacheAttentionState`，按字段名取值"：

| 位置 | 现状 |
|---|---|
| `litgpt/log_kv_cache.py:1246` | `get_attention_state()` 自己的定义——返回类型改造的起点 |
| `litgpt/log_kv_cache.py:1435`（`append_exact_tokens()`）| 消费 `get_attention_state()` 的输出、拼接 in-flight chunk，再产出下一层要用的元组——输入输出都要跟着换成 `CacheAttentionState` |
| `litgpt/log_kv_cache.py:1738/1754`（`log_kv_chunk_attention()`）| 训练流式 attention 的共享构件，`second_order_scale==0.0`/`!=0.0` 两个分支各调用一次 `get_attention_state()` |
| `litgpt/model.py:1209/1219`、`1284/1338`、`1397/1409` | **三处**独立的流式调用点，每处都是"`get_attention_state()` 接着 `append_exact_tokens()`"这个模式的一次独立重复，三处都要改，不能只改一处漏两处 |
| `litgpt/log_kv_diag.py:624` | 诊断工具对 `append_exact_tokens()` 的调用，语义簇路径下诊断输出的字段也要跟着扩展，否则诊断工具会在语义模式下悄悄丢信息 |
| `tests/test_log_kv_cache.py`、`tests/test_log_kv_diag.py` | 分别约 22、3 处直接调用（`grep -c` 实测）。**这一轮更正：上一版"legacy 测试不强制改写"是错的，必须删掉这句话**——`CacheAttentionState` 恒定 10 个字段（§5.14），`get_attention_state(with_stats=False)` 返回的是一个**完整的 10-元组**（不用的字段填 `None`，不是"3 个字段的元组"），`slot_k, slot_v, slot_w = c.get_attention_state(with_stats=False)` 这类 3-变量解包会直接 `ValueError: too many values to unpack`；`with_stats=True` 同理，8-变量解包对上 10-元组同样会炸。`NamedTuple` 支持位置解包，但前提是解包变量数与字段数**完全相等**，不是"随便解包几个都行"——这条前提在 legacy 测试里从不成立。**所有 22+3 处直接调用，不论 legacy 还是语义模式，都必须在同一次改动里迁移成按字段名取值，没有例外、没有过渡期**：这个返回类型改动对**任何模式**都生效（§5.14"`get_attention_state(with_stats=False)` 在任何模式下都返回同一个类型"），不是被 `log_kv_semantic_clusters` 开关裹住的语义模式专属改动，legacy 调用同样会在 Python 层面直接报错，不是"数值算错"这种能被 §5.19-1 字节等价 CI 捕捉的问题——那条 CI 门检查的是"开关关闭时数值是否不变"，管不到"函数签名/返回元组长度变了导致调用方 `ValueError`"这类问题，这两件事是正交的，不能把后者的责任推给前者。 |

**二阶修正的精度应该变好，不是变差**：D1 自检里 width≥8 的失真，根因之一是 Σ 统计在
post-RoPE 空间——槽内 token 位置不同，`R(p_j)k_j` 之间的差异里混着**位置相位方差**，
而这部分方差方向弥散、rank-1 抓不住，白白吃掉唯一的那个秩。改到 pre-RoPE 之后 Σ 只量
内容方差，而簇内内容同质正是聚类在优化的目标 → **协方差谱更集中，rank-1 截断误差应
系统性下降**。这意味着"二阶修正只把 niah 从 0.032 提到 0.0827"这个偏弱的结果，可能
有一部分是被位置相位污染拖累的，而不是方法本身的上限。S0.4 可顺带验证。

#### C. 唯一一处"看起来能直接复用、实际上不能"

`LogKVStreamTrainingAttention` 的 docstring 把重放确定性论证为：

> "compaction is pure mean-pooling with **count-based binary carries**"

**"只依赖计数、不依赖数据"这个前提被语义路由打破了。** 框架保留，但重建 cache 时的
依据必须从"重算路由"换成"重放 `op_log` 记录的操作序列"（§11-A、§5.21-2）。不处理不报错，
只会让梯度属于另一个函数——**这是全部改动里最容易静默出错的一处**。

### 5.21 开工前必须定死的五个决定

这五条都会在实现到一半时把人卡住，且都不是"写着写着就知道了"的类型。**在写第一行
生产代码之前必须有结论**，否则返工成本极高。

#### 5.21-1 pre-RoPE 接口改动的完整影响面（远不止"多传两个参数"）

现状（`model.py:790-816`）的实际顺序是：

```
qkv.split → norm_q / norm_k（Qwen3 是 norm_qk=True, type="default"）
          → apply_rope( k[..., :rope_n_elem] )        ← 只旋转前段
          → cat( k_roped, k[..., rope_n_elem:] )      ← 尾部内容通道原样
          → 交给 LogKV（training / inference 两条路径）
```

所以**"pre-RoPE k"精确地说是"qk-norm 之后、apply_rope 之前"**，不是 qkv 投影的原始
输出。取错位置会让簇的度量落在未归一化的空间里，`s_h` 标定直接失效。

**受影响的地方（逐条都要处理，不能只改一个入口）**：

| 位置 | 影响 |
|---|---|
| `q` 必须保持 post-RoPE | query 照常旋转。所以要**同时持有 `k_roped` 和 `k_raw`**，现有代码把两者写进同一个 `k` 变量，得拆开 |
| 两条调用路径 | `_log_kv_train_lowmem_forward`（training）与 `_log_kv_training_forward`（inference prefill/decode）是**两个不同的调用点**，都要传 |
| in-flight chunk | `append_exact_tokens` 把当前 chunk 作为精确槽拼在扁平池尾部做 chunk 内因果——这条路**必须用 post-RoPE k**。它与"cache 里 `w=1` entry 物化出来的键"必须逐位一致，**这是一条硬性单测**（两条路径不同，但数学上应当同一） |
| `_log_kv_pending` | 奇数长度 prompt 会把末尾单 token 挂起、与首个 decode token 配对。**挂起的元组必须同时带上 `k_raw`**，否则它被 flush 进 cache 时没有 pre-RoPE 形态 |
| partial rotary | 尾部 `[rope_n_elem:]` 旋不旋转都一样，所以**只有前 `rope_n_elem` 维需要区分 raw/roped**。存储上可以只存 raw，物化时旋前段、拼尾部（§5.14 已如此） |
| GQA | `k` 在这一步已经是 `(B, G, T, hs)` 的 per-KV-group 形态，**写入侧不涉及 rf 折叠**，折叠只发生在读出侧的 `log_kv_slot_attention`。这条是好消息 |
| 那段注释 | `model.py:805-808` 明确写着 expected-RoPE 的设计前提（"LogKV mean-pools the FULL key … expected rotation over the span"）——**那正是被替换掉的东西，注释必须同步改**，否则下一个读代码的人会按旧模型理解 |

**上表说了"要同时持有 `k_roped` 和 `k_raw`"，但没把最关键的一条钉死：autograd
`Function` 的训练目标本身有没有变**。核对现有 `LogKVStreamTrainingAttention`
（`log_kv_cache.py:1838` 起）的 docstring 后确认，这条此前被隐式地答错了，必须
先定死它，`Function` 签名、`ctx.save_for_backward` 存什么、要不要 detach 这些
接口细节才有意义——顺序反了会把"怎么传参"和"训练目标要不要变"混成一个问题。

**现有训练目标是 stop-gradient through cache commit，这是不可动摇的 v1 前提，
不是可以顺手改掉的实现细节**：docstring 原文——"gradient reaches q/k/v only
through each token's own block (the cache commits detached copies)"——是整个
`LogKVStreamTrainingAttention` 存在的数学基础。它把 backward 需要的计算图限制
在**每个 `train_block` 局部**，这正是它能把训练内存从 `O(T·S)` 压到
`O(T + train_block·S)` 的原因（docstring 开头那段"the naive training graph
saves... hundreds of GB"正是为了避免这件事才写了这整个 Function）。**v1 必须
原样保留这条前提，不能让语义簇路径悄悄改变训练目标。**

> **一处曾经写错的设计**：上一轮文档给 `backward()` 写了一个 `grad_k_raw`，
> 声称要"通过 `op_log` 重放链路对 `compact()` 的加权统计求解析梯度"，把 cache
> 写入这条路径变成可微分的。**这和上面那条 stop-gradient 前提直接矛盾**——如果
> cache 写入可微分，某个 token 的梯度就要一路穿过它所在簇此后全部的合并历史，
> 传回到构成每一次合并结果的**所有**原始 token，这正是 docstring 开头"避免跨越
> 整个流的计算图"这句话要防止的情形，等于把这个 Function 存在的理由重新引入了
> 一遍。要不要真的做"cache 写入可微"是一个训练目标层面的决定，不是一个"顺手在
> 写 `Function` 签名时可以捎带手做的接口改动"——上一轮把两个问题混在了一起，
> 答案是错的：**v1 不改训练目标，`k_raw` 和 cache 写入这条路完全不参与反向
> 传播，和现状代码里 `v` 的处理方式完全一样**（`v` 也是既参与 cache 写入的数值
> 计算、又从这个 Function 拿不到梯度——`k_raw` 现在只是显式地遵守同一条早就
> 存在的前提，不是新规则）。

```python
# 现状（单 k，post-RoPE）：
@staticmethod
def forward(ctx, q, k, v, cache, scale, train_block, second_order_scale, *pin_args):
    ...
    ctx.save_for_backward(q, k, v)
    ...

@staticmethod
@once_differentiable
def backward(ctx, grad_y):
    q, k, v = ctx.saved_tensors
    ...
    return grad_q, grad_k, grad_v, None, None, None, None, ...
```

**v1 改动（接口变了，训练目标没变）**：

```python
@staticmethod
def forward(ctx, q, k_raw, k_roped, v, cache, scale, train_block, second_order_scale, *pin_args):
    ...
    # op_log/op_log_len 不是这个 Function 的输入参数——它们是 forward 处理完
    # 整条序列全部 block/flush 之后，cache 对象上累积出来的最终状态，在这里
    # 读出来一起存进 ctx（§5.21-2"op_log 的有效长度"一节的定案）。
    # 不需要 .detach().clone()——cache.op_log/op_log_len 不是 reset_parameters()
    # 原地 zero_() 复用的持久 buffer，而是每次 forward() 开头被重新绑定成
    # 全新分配的张量（见下方"训练峰值显存"更正框），下一次 reset_parameters()
    # 只会让 cache.op_log 这个属性名指向另一块全新存储，不会动这次 ctx 里存的
    # 这个对象——两者从一开始就是不同的张量，没有别名，不存在需要克隆去
    # 撇清的共享关系
    ctx.save_for_backward(q, k_raw, k_roped, v, cache.op_log, cache.op_log_len)
    ...

@staticmethod
@once_differentiable
def backward(ctx, grad_y):
    q, k_raw, k_roped, v, op_log, op_log_len = ctx.saved_tensors
    # op_log/op_log_len 是 forward 那次调用专属分配的张量，本来就不会被
    # 任何其它 forward()/reset_parameters() 调用触碰——重放只读它们，也不需要
    # 关心 cache 对象在这之间发生了什么
    ...
    grad_q       = ...   # 不变：in-flight exact attention 对 q 的梯度
    grad_k_roped = ...   # 不变：in-flight exact attention 对 k 的梯度，和现状代码里
                          # grad_k 的计算方式完全一样，只是改了变量名
    return grad_q, None, grad_k_roped, grad_v, None, None, None, None, ...
    #             ^^^^ k_raw 的梯度槽位恒为 None——cache 写入这条路从设计上就
    #                  不可微，和 grad_v 只从 in-block exact attention 来、
    #                  cache 内容本身不贡献梯度是同一条已有规则
```

**`k_raw` 的角色和现状代码里的 `v` 完全对称，不是新引入的一类张量**：两者都
（a）参与 cache 写入的数值计算（在 `forward()` 方法体内部已有的
`with torch.no_grad():` 块里，`v` 今天就是这么处理的），（b）从这个 Function
拿到的梯度恒为 `None`。`k_roped` 则和现状代码里的 `k` 完全对称：参与 in-flight
exact attention，正常传播梯度，算法和现状一字不改，只是改了变量名。**这个
`Function` 的 `k_raw` 输入参数本身，传进来要不要 detach 都不影响这个函数内部
的行为**——`forward()` 整个方法体本来就在 `torch.no_grad()` 里跑，
`Function.forward` 本身也从不记录图，传一个 `requires_grad=True` 的张量进来
不会意外泄漏梯度出这个函数，`backward()` 稳定返回 `None` 也不依赖这个。

> **但这句话说的是"这个 Function 的输入参数"，不是"调用方可以在计算
> `k_roped` 之前把上游张量本身 detach 掉"——这是两件事，容易被前一句误导着
> 做错。** `model.py` 里 `k_raw = norm_k(...)` 和 `k_roped =
> apply_rope(k_raw[..., :rope_n_elem])` 共享同一个上游张量（qk-norm 输出）；
> `k_roped` 的梯度要一路回到这个上游张量、再回到 `norm_k`/qkv 投影的参数，
> 靠的是 `apply_rope` 这个**普通、未被这个自定义 `Function` 包裹**的可微分
> 算子正常参与 autograd。**如果实现时图省事，在算出 `k_roped` 之前就先对
> `norm_k(...)` 的输出整体 `.detach()`**（比如想着"反正 `k_raw` 不需要
> 梯度，不如从源头一次性切断"），会把 `k_roped` 的梯度链也一起切断——这不是
> "这个 Function 的 `k_raw` 输入该不该 detach"这个局部问题，是**调用方在
> 两个下游消费者分叉之前，对共享上游做了不该做的全局 detach**。正确做法是
> `k_roped` 必须从**未经任何 detach** 的 `k_raw`（即 qk-norm 的原始输出）
> 计算得到，"`k_raw` 不产生梯度"这件事完全由传给这个 `Function` 的**那个
> 具体调用**、以及它的 `backward()` 恒返回 `None` 来保证，不需要、也不应该
> 在调用方源头做任何 detach。

**`op_log` 存在的理由因此比上一轮写得更清楚了：它不是为了支持某种新梯度路径，
是为了让 backward 的重放能正确重建"某一时刻 cache 的（依然完全 detached 的）
内容"，从而让 in-flight exact attention 那部分重算出正确的数**——这和
§5.21-2"重放完全不需要 centroid"那段已经确立的结论完全一致，需要更正的只是
上一轮误加的 `grad_k_raw` 这一条，op_log 重放算法本身的设计不受这次修正影响。

**如果未来真的要做"cache 写入也可微"这个更激进的方向**：那是一次独立的、
训练目标层面的重新设计，至少需要（i）先证明新的计算图仍然有界（不能退回
`O(T·S)`，否则重新引入这整个 Function 存在的理由要解决的 OOM 问题）、（ii）一个
naive reference 实现（哪怕慢、哪怕只能跑小规模，用来核对解析梯度公式对不对）、
（iii）reference 和 replay 版本的 backward 数值对拍。这三条现在都没有着落，
所以明确列为**不在 v1 范围内**，不是"以后顺手加上"的小事。

#### 5.21-2 `op_log`：完整的操作语义、顺序、重放算法

原设计写的是"每 token 的簇 id + 是否开新段"。**这不够**：`K_max` 满时会 Ward 合并、
释放槽位、**复用 cluster id**——同一个 id 在合并前后指的是不同的簇，仅凭 token 归属
无法重建结构。backward 重放会建出一个 forward 从未 attend 过的 cache（§11-A）。上一轮
补的六类操作解决了"记什么"，但没定"什么顺序、谁消费哪个 token、centroid 要不要重放"
——这三条不定死，实现到重放逻辑那一步会直接卡住。

**六类操作分两组，这个区分是重放算法的基础**：

```
(op_type, arg0, arg1, arg2)   # 每条 4 个 int32

── 主操作：每个 flush 出来的 token 恰好触发一个，且互斥（直接对应 §5.3 的三路判定）──
NEW_CLUSTER   (slot_idx,  0,        token_idx)   # 语义 novelty：在哪个空槽建
                                                  # 新簇，segment 恒为 0（新簇的
                                                  # 第一段），token_idx 是这个
                                                  # token 在整条序列里的绝对位置
NEW_SEGMENT   (cluster,   new_seg,  token_idx)   # 时序打断：同簇开新段
JOIN          (cluster,   segment,  token_idx)   # 归入既有簇的当前段

── 结构操作：不消费 token，作为主操作的副作用穿插出现 ──
WARD_MERGE    (keep_slot, free_slot, -1)     # 给某个 NEW_CLUSTER 腾位，必然紧邻其前
PAD_INSERT    (cluster,   level,    count)   # 给某个 NEW_SEGMENT 做对齐填充，必然紧邻其前
CARRY         (cluster,   level,    resulting_count)  # ladder 进位，**非权威、可选**，见下方说明
```

> **`arg2` 对三类主操作的含义，这一轮从"未使用的 `-1`"改成了`token_idx`**——
> 见前面§5.4"op_log 跨 Phase 的顺序契约"一节：重放不能再靠一个隐式递增的
> `token_ptr` 去猜"这条主操作对应哪个原始 token"，因为 Phase 1（direct）和
> Phase 2（orphan）各自向本地缓冲追加 op 的顺序，和这些 token 的真实到达顺序
> 并不是同一个顺序（Phase 1 全体先写，Phase 2 全体后写，两者内部各自保序，
> 但交替出现的 direct/orphan 之间不保证全局顺序）。`token_idx` 把"这条 op
> 消费哪个 token"从"隐式位置"变成"显式字段"，重放不再需要假设这两个顺序
> 相同。

> **一处曾经和下面的重放伪代码对不上的字段**：`NEW_CLUSTER` 原来写的是
> `(slot_idx, -1, -1)`——arg1（对 `JOIN`/`NEW_SEGMENT` 而言就是 `op.segment`）
> 填 `-1`。但下面的重放循环对**三类主操作一视同仁**，统一调用
> `append_to_ladder(..., op.cluster, op.segment)`，从不对 `op.type` 做特判。
> 这意味着一个新簇的第一个 token 会被 append 成 `segment=-1`——一个不存在的
> 段号，下游任何按段号索引/比较的逻辑（§5.11 的对齐填充、§2.1 的 segment 语义）
> 遇到它都是未定义行为。**改法是让 `NEW_CLUSTER` 的 arg1 显式填 `0`**（新簇的
> 第一段天然是段 0），而不是给重放循环加一个"只有 `NEW_CLUSTER` 特殊处理
> segment"的分支——**字段自洽比处理逻辑自洽更简单**：让日志格式本身对所有主
> 操作保持同一套字段语义，重放循环就能保持完全通用、不需要按 op 类型分叉，这也
> 是为什么选择改数据格式而不是改重放代码。

**顺序规则**：主操作严格一一对应"某个 token"，每条都显式携带这个 token 的
`token_idx`（arg2，见上面的更正框）；结构操作插在触发它的主操作旁边
（`WARD_MERGE` 在它服务的 `NEW_CLUSTER` **之前**——先腾位再建簇；`PAD_INSERT`
也在它服务的 `NEW_SEGMENT` **之前**——先把上一个段的尾部对齐填好，再让新段的
第一个 token 进场；`CARRY` 跟在导致进位的那次 ladder 追加后面）。**op_log 的
线性顺序不等于 token 到达顺序**——精确的顺序契约是§5.4"op_log 跨 Phase 的
顺序契约"一节给出的三条：同一逻辑簇自己的主操作序列按真实到达顺序、结构操作
紧邻它服务的主操作、不同簇之间的相对顺序不重要。Phase 1（direct）整体先写、
Phase 2（orphan）整体后写，满足这三条，但**不**等于"整条 op_log 就是把 `m`
个 token 按 `τ=0..m-1` 原样排一遍"——这也是为什么主操作必须显式带 `token_idx`
而不能靠位置推断。

> **一处曾经写错的顺序，务必别再犯**：早期版本把 `PAD_INSERT` 写在它服务的
> `NEW_SEGMENT` **之后**。§5.11 自己举的例子已经说明了正确顺序应该是什么样——
> `[a1][a2][a3][∅] | [b1][b2]`，空位要插在**旧段尾部、新段第一个成员之前**。
> 但按"PAD 在 NEW_SEGMENT 之后"这条错误顺序重放，`NEW_SEGMENT` 会先把 `b1`
> append 进 ladder，`PAD_INSERT` 才轮到执行——填充槽这时候只能插在 `b1`**后面**，
> 变成 `a3, b1, ∅`，跨段配对的问题根本没解决，§5.11 整节的动机落空。**填充必须
> 在新段第一个 token 被真正 append 之前完成**，所以 `PAD_INSERT` 必须先于它服务的
> `NEW_SEGMENT`，不是之后。

**重放算法**（backward 只需要这一个循环，不需要重新跑 §5.2–§5.6 的任何一步）：

```
for op in op_log[:op_log_len]:   # 只遍历有效前缀，§5.13 新增的 op_log_len；
                                   # 扫整个静态 (OP_max,4) buffer 会把尾部未
                                   # 写入的行当成合法 op，见下方"op_log 的
                                   # 有效长度"一节
    if op.type in {NEW_CLUSTER, JOIN, NEW_SEGMENT}:
        # 消费 op.token_idx 指向的原始 (k_raw, v, pos)——不是隐式递增的计数器，
        # 见 §5.4"op_log 跨 Phase 的顺序契约"一节：op_log 的线性顺序不等于
        # token 到达顺序，必须用 op 自己携带的 token_idx 显式索引，位置必须
        # 一起传，锚点 (p_lo/p_hi/sum_wp) 是绝对位置的函数，缺了 pos 这一步
        # 没法定义。按 op 指定的 (cluster, segment) 追加进该簇的 ladder —— 用的是
        # compact()/_binary_carry() 那套纯算术，不依赖任何浮点比较，所以给定
        # 同样的 (token, pos, cluster, segment) 四元组，结果必然逐位相同。
        append_to_ladder(k_raw[op.token_idx], v[op.token_idx], pos[op.token_idx],
                          op.cluster, op.segment)
    elif op.type == WARD_MERGE:
        ward_merge_only(op.keep_slot, op.free_slot)   # §5.6 的 1-2 步（纯状态
        # mutation，不写 op_log——这条 WARD_MERGE 是在读，不是在写，§5.6 已更正）
        # 不建新簇——新簇由紧随其后的 NEW_CLUSTER 这条主操作负责，走上面的分支
    elif op.type == PAD_INSERT:
        insert_pad_entries(op.cluster, op.level, op.count)   # §5.11，必须在它
        # 服务的 NEW_SEGMENT 之前执行到——这靠 op_log 里的写入顺序保证，此处的
        # 遍历只是忠实按顺序回放，不需要额外判断"是不是该我了"。
    elif op.type == CARRY:
        # 非权威：不做任何状态改变，已经隐含在 append_to_ladder/insert_pad_entries
        # 里发生。仅当 op_log 是用调试 build 记录的（见下方说明）才有内容可断言：
        assert live_entry_count(op.cluster, op.level) == op.resulting_count
```

**关键认识：重放完全不需要 centroid，一步都不需要。** centroid（以及驱动它的 Phase
1/2/3 批量路由、Ward 代价矩阵、join cost 的浮点比较）**只用于决定"这个 token 该去
哪"**——这个决定的结果已经被 op_log 忠实记录了。而 `compact()`/`_binary_carry()` 那套
决定"槽内数值算出来是多少"的数学，只依赖"哪些 token 按什么顺序进了哪个 (簇,段)"，
和 centroid 的具体数值毫无关系（centroid 是纯路由元数据，不参与 attention 读出，
§2.4）。所以 backward 重放**跳过整个路由阶段**，直接把 op_log 当作"已经决定好的
安置计划"来执行——这样一来，路由阶段所有的浮点敏感操作（distance 比较、argmin、
Ward 代价）都被彻底隔离在 forward 侧，backward 侧只剩纯整数索引 + 确定性算术。

**`CARRY` 为什么标"非权威、可选"，以及第三个字段现在存什么**：上面的重放循环从不
读 `CARRY`，进位效果完全由重复调用 `append_to_ladder`/`insert_pad_entries` 自然
产生——这是设计使然，不是遗漏。`CARRY` 存在的唯一价值是**给调试/对拍提供一个可断言
的检查点**：`resulting_count` 记录"这次进位完成后，`(cluster, level)` 上存活的
entry 数"，重放时可以据此断言"我这一步重算出的 ladder 状态和 forward 当时观察到的
状态一致"，在 S0.8（批量 vs 严格串行的对拍）里直接有用。**生产 `op_log` 默认不写
`CARRY`**——它对最终结果没有贡献，纯粹是 `OP_max` 预算里的死重（§5.21-2 的容量
推导本就已经把它算作"低一个量级、可并入余量"的部分，去掉它只会让预算更宽松，不会
让已经证明的上界失效）；只在专门的调试/对拍 build 里打开，此时它才需要
`resulting_count` 这个字段有真实内容，其余情况可以直接省略这一整类 op。

**批量前向写入和串行重放必须逐位等价，这是一个需要显式验证的契约，不是自动成立
的**。forward 侧的实际实现**不是**上面这个 per-token 的 Python 循环——§5.4 的
Phase 1/2/3 是批量路由（一次处理整个 flush 批，多达 128 个 token），§5.21-3 进一步
要求 ladder 的实际写入/进位也是**向量化的**（`(B,G,K,L)` 掩码并行 carry，不是逐
token 调用）。

> **更正：`op_log` 的线性顺序不是"forward 实际执行的物理顺序之外的另一个逻辑
> 顺序"，它就是 forward 的真实执行顺序，两者被构造成完全一致，不是两个需要
> 分开论证再对齐的东西。** 早期表述说 op_log 的顺序是"如果这批 token 被逐个
> 串行处理，会产生的顺序"，暗示存在一个假想的、更细粒度的"真串行顺序"，
> op_log 只是它的一个**逻辑**近似——这个说法不精确，且和 §5.4"op_log 跨
> Phase 的顺序契约"一节的结论矛盾：op_log 的物理顺序（Phase 1 全体先写、
> Phase 2 全体后写）**不等于**把全批 `m` 个 token 按真实到达顺序 `τ=0..m-1`
> 原样排一遍（direct 和 orphan 交替出现时，两者不是同一个顺序）——但这不是
> "近似"或"简化"，而是**被证明为安全的、故意选择的**顺序：只要三条顺序契约
> （同簇内部保序、结构操作紧邻主操作、跨簇顺序不重要）成立，"Phase 1 组
> 先、Phase 2 组后"就是 forward 真实发生的事，op_log 忠实记录了它，不多不少。
> 下面两条理由针对的是**同一个 Phase 内部**（尤其是 Phase 1 的向量化批处理）
> 是否等价于对这个 Phase 自己负责的 token 子集做严格串行处理，不是跨 Phase
> 的顺序问题——那个问题已经在§5.4解决，这里不重复。

重放算法（上面那个 for 循环）假定"按 `op_log` 顺序逐条串行执行"和"forward 的
批量向量化实现"产生**逐位相同**的最终 ladder 状态，这个假定成立的理由和它对
实现提出的具体要求：

- **成立的理由**：`compact()`/`_binary_carry()` 的正确性只依赖"同一个簇收到的成员
  按什么**相对顺序**到达"，不依赖"这些成员是被一次一个地处理，还是被一批一起写入
  的"——`p_lo`/`p_hi` 取 min/max、`sum_wp` 是整数加法，这两类运算本身与批处理粒度
  无关；真正对顺序敏感的是 `compact()` 的"时间序相邻配对"语义（§5.7 已核实的
  docstring），只要落进同一层的成员在被配对之前，其相对到达顺序和串行版一致，
  配对结果就一致。
- **对实现的具体要求**：批量写入必须**保序**——一个 flush 批内，凡是路由到同一个
  `(cluster, level=0)` 的 token，必须按它们的真实到达顺序被写入/参与 carry，
  不能因为向量化 scatter 而被打乱（Phase 1 内部是 direct 子序列的压缩下标 `t`，
  Phase 2 内部是 orphan 处理的串行顺序，两者各自都是真实到达顺序的子序列，
  §5.4 已经论证过跨这两者不需要额外保序）。这一点对当前设计**几乎是免费的**
  ——Phase 1 的路由结果 `c*[t]` 本就是逐位置索引对齐的张量，没有引入任何重排；
  真正需要小心的是**批量 carry 本身**：如果一批里有 `K > 1` 个 token 落进同一个
  此前尚有空位的 `(cluster, level=0)`，向量化实现必须用等价于"依次单个 carry
  调用 `K` 次"的方式处理这批到达（标准二进制计数器的"批量加 `K`"和"逐一自增
  `K` 次"在最终数字模式上永远一致，这是可以直接引用的经典结果），而不是任何
  会重排到达顺序或跳过中间进位状态的捷径。
- **必须落地成单测**（S0.1/S0.8 之间，纯 CPU 可测）：构造一批同簇 token，分别用
  (a) 批量向量化路径、(b) 把该批的 `op_log` 结果拿去跑上面的串行重放循环，断言两条
  路径产出的 ladder 张量（`k̄/v̄/w/p_lo/p_hi/sum_wp/σu/σ2/γa/γb/γ` 全部字段）逐位
  相等。这条测试没通过之前，`op_log` 重放的正确性论证是**未经验证的假设**，不能
  当作已解决问题写进实现顺序表。

**批量路由（§5.4）留下的一个未解决风险，需要 S0.8 专门测**：Phase 1 用**冻结的
centroid** 并行分配一整个 flush 批（多达 128 个 token），但 §5.3 的 join cost 里
`p_hi_c`（该簇最近一次收到成员的位置）如果也在整批内冻结，同一批里连续多个 token
被分到同一个簇时，**除第一个之外**看到的都是"批次开始前"的 `p_hi_c`，会把彼此明明
紧挨着的 token 误判成"隔了很久"，触发不必要的 `NEW_SEGMENT`。**这不是路由准确率的
损失，是段计数会被系统性推高**，连带推高 `PAD_INSERT` 的频率——直接侵蚀预算（§4）。
缓解方向是在 Phase 1 内部对 `p_hi_c` 做一次按簇分组的前缀扫描（只需遍历批内**出现过
的簇数**，通常远小于批大小，不是回到逐 token 串行），把"批内同簇的最近成员位置"
更新进去再判定 join cost。**这条修正是否必要、批冻结造成的段膨胀有多大，由 S0.8
连带 S0.2 的 segment 数统计来判定**——如果膨胀量在 §5.11 对齐填充的预算内可以吸收，
可以先不修，留作已知近似。

**容量：`OP_max` 现在有一个可证明的硬上界，不是经验估计——但这个上界本身有前提，
不是对任意 `ℓ_block` 都成立。** 逐项数一遍：

```
主操作     恰好 T 条（每 token 一条，互斥，精确）
WARD_MERGE 最坏 O(T)（病态输入下，K_max 绑定后几乎每个 token 都需要先腾位再建簇——
           这不是罕见事件的假设，是 §4"最坏情况"本身就该覆盖的场景）
PAD_INSERT 最坏 O(T)（每条 NEW_SEGMENT 至多配一条——**前提是 `count` 字段把一次
           边界的全部填充聚合进同一条 op**，不是每个填充槽一条 op，否则这一项
           会变成 O(T·2^ℓ_block)，指数项直接压垮整个上界）
CARRY      O(T/B′)（标准二进制计数器的摊还论证：n 次自增的总进位次数是 O(n) 而非
           O(n log n)——第 i 层每 2^i 次自增才翻一次，比其它三项低一个量级，可并进
           余量而非单独扩容）
```

**`PAD_INSERT` 那一行的 O(T) 依赖三个前提，缺一个都会让 `OP_max = 4·T_max` 不成立**：

1. **`ℓ_block ≤ 2`**。这不是为了让 §5.11 那张"每边界浪费 `2^ℓ_block−1` 槽位"的表
   本身好看——那张表管的是槽位预算，跟这里的 op 计数是两件事。真正的原因是
   `count` 字段的取值范围必须有界：若 `ℓ_block` 可以取到 Stage 0 探索范围
   `L_alloc`（32k 下约 11~15），单次填充就可能要求 `count` 达到 `2^11` 量级，超过
   单层容量 `B′`（512）——这时"一条边界一条 `PAD_INSERT` op"这个聚合假设本身
   失效（填充量超出该层能装下的范围，必须拆成多条、甚至跨层的 op），O(T) 退化成
   O(T·ℓ_block) 甚至更差。**`ℓ_block ≤ 2` 把 `count` 的上界摁在 3，永远不会撞
   `B′`，这是 O(T) 成立的充分条件。**
2. **`PAD_INSERT` 的实现确实按 `count` 聚合，不是把每个填充槽拆成独立 op**——这是
   op 格式 `(cluster, level, count)` 里 `count` 字段存在的唯一理由；实现时若偷懒
   展开成 `count` 条独立 op，这一项直接退化成上面第 1 点里的指数情形，且退化的
   触发条件与 `ℓ_block` 是否 ≤ 2 无关（哪怕 `ℓ_block=2`，展开成独立 op 也会让
   常数从 1 变成 3，只是没那么致命）。
3. **`PAD_INSERT` 先于它服务的 `NEW_SEGMENT`**（§5.21-2 上面的顺序规则）——这条
   只影响语义正确性，不影响计数，但错误顺序产生的 op 仍然占用 `OP_max` 里的一个
   位置，所以列在这里作为"这三条同时成立，`OP_max=4·T_max` 才是安全上界"的一部分。

**生产路径必须在 cache 构造时（不是训练/推理运行到一半才发现）硬校验
`ℓ_block ∈ {0,1,2}`**：

```python
if log_kv_seg_block_level not in (0, 1, 2):
    raise ValueError(
        f"log_kv_seg_block_level={log_kv_seg_block_level} 超出生产路径支持范围 "
        f"{{0,1,2}}——OP_max 的 PAD_INSERT 聚合假设、§5.11 的槽位预算表都只在这个"
        f"范围内成立，更大的值只能用于 Stage 0 离线扫描"
    )
if log_kv_cluster_entries % (2 ** log_kv_seg_block_level) != 0:
    raise ValueError(
        f"log_kv_cluster_entries (B'={log_kv_cluster_entries}) 必须是 "
        f"2**log_kv_seg_block_level ({2 ** log_kv_seg_block_level}) 的倍数——"
        f"§5.11 的 PAD_INSERT 对齐公式直接读 level_count[cluster,0] 而不维护独立的"
        f"累积计数器，这个等价关系的前提就是 B' 整除 2**ℓ_block，默认组合"
        f"（B'=8, ℓ_block∈{{0,1,2}}）天然满足，覆盖 B' 时必须一并检查"
    )
```

**Stage 0 的 S0.0 豁免于这条校验**：S0.0 扫 `(g_max, ℓ_block)`（§7）是在 dump 出来
的 `k_raw` 上做纯 CPU/NumPy 模拟，复现簇/段边界的计数逻辑，**完全不经过 `op_log`
或真实 cache 构造路径**——`ℓ_block` 在那里只是一个统计口径参数，不会真的驱动一次
GPU cache 写入，所以不受这条校验约束。等 S0.0 选出生产候选值后，才会以该值（必然
落在 `{0,1,2}` 内）进入 Stage 1 实现，届时这条校验才第一次生效。

**结论（三条前提成立时）**：`OP_max = c · T_max`，`c` 取一个能覆盖"1 主操作 + 1
WARD_MERGE + 1 PAD_INSERT + CARRY 余量"的小常数，**`c = 4` 足够**。这比早期"`T_max +
2·K_max·(预期合并次数)`"的经验估计更大（后者隐含假设合并很少见，是期望而非最坏
情况），但现在是**证明过的上界**，可以像现有 `_count_tokens` 对 `max_seq_length`
那样，**溢出直接硬失败**（`raise RuntimeError`），不做动态扩容、不做静默截断——
矩形预分配 + 硬失败是这个项目一贯的选择（§5.17），`OP_max` 没有理由是例外。

32k 下 `(1,8,4·32768,4)` int32 ≈ 16MB/层，28 层约 **448MB**——比早期估的
224MB 贵一倍，但那 224MB 本来就是经验值，不是这次算出的真实上界。这是正确性的价格，
不是可选项。

#### `op_log` 的有效长度：`op_log_len`/`local_op_len`，以及它和 `reset_parameters()`/backward `ctx` 的关系

**buffer 表只给了 `op_log` 的静态形状 `(B,G,OP_max,4)`，但前面的重放循环
`for op in op_log` 隐含假设"整条 op_log 都是合法内容"——这从来不成立，必须显式
补上"哪些行有效"的规格，否则实现者要么读到尾部垃圾数据，要么各自发明一套不
兼容的约定。**

**`op_log_len: (B,G)` int32**（已加进 §5.13 的表）：`op_log` 当前写到第几行。
追加一条 op 就是 `op_log[b,g,op_log_len[b,g],:] = new_op; op_log_len[b,g] += 1`；
任何读 `op_log` 的代码（重放、`scan_op_log`、S0.8 对拍脚本）都必须先用
它截断成 `op_log[b,g,:op_log_len[b,g],:]`，未写入的尾部行内容未定义，不能假设
它们是全零或任何特定哨兵值。

**`local_op_len`**：本地缓冲（§5.4"本地缓冲 vs 持久 `op_log`"一节）同样需要一个
长度指针，道理和 `op_log_len`完全一样，只是作用域是"当前这一个 flush 批"：批
开始时置 0，Phase 1（一次性写入这一批已知数量的主操作）和 Phase 2（逐个 append
orphan 的主操作和结构操作）都在写入的同时推进它，超过 `local_op_cap` 立即硬
失败（§5.4 已有的规则）。批处理完，提交进持久 `op_log` 就是一次定长拷贝加两个
指针的更新：

```
op_log[b, g, op_log_len[b,g] : op_log_len[b,g] + local_op_len[b,g], :]
    = local_buffer[b, g, :local_op_len[b,g], :]
op_log_len[b,g] += local_op_len[b,g]
```

这条拷贝本身不需要任何逐条判断——`local_op_len[b,g]` 已经精确就是这一批要
提交的行数，`op_log_len` 全程只做加法，不需要重新扫描或去重。

**`op_log`/`op_log_len` 和 `reset_parameters()`/backward `ctx` 的关系，此前
留了两个选项没有二选一，这里定案**：§5.19 那条 tip 曾经写"`op_log` 要么存在
`ctx` 里，要么在 reset 时显式豁免"——两个选项都列出来但没有拍板，会让实现者
自己猜。**定案：存进 `ctx`，不做 reset 豁免。** 具体地，`forward()` 处理完
整条序列的最后一个 block/flush 之后（此时 `op_log`/`op_log_len` 已经是这次
forward 完整、最终的状态），把它们和 `q`/`k_raw`/`k_roped`/`v` 一起存进
`ctx`。`backward()` 从 `ctx` 读回这份快照来重放，**不读取、也不依赖 cache
对象自己的 `op_log`/`op_log_len`**——cache 对象上的这两个属性在下一次
`forward()` 开头被 `reset_parameters()` **重新绑定到全新分配的张量**（不是
原地清空旧张量，两者的区别、以及为什么必须是前者，见下方"训练峰值显存"
更正框）是完全正常、预期内的行为，不需要任何特殊豁免逻辑。这样选的理由：
- 和 `q`/`k_raw`/`k_roped`/`v` 已经在用的模式完全一致，不需要给 `op_log` 发明
  第二套"豁免 reset"的特殊生命周期规则；
- `ctx.save_for_backward` 是 PyTorch autograd 的标准机制，`backward()` 只依赖
  `ctx` 这一个自包含的输入，不依赖"cache 对象在 forward 和 backward 之间没有
  被其它代码修改过"这个更脆弱、更隐式的前提——比如梯度累积场景下，同一个
  cache 对象可能在这次 forward 的 `backward()` 被调用之前，就被下一次
  `forward()` 调用并 reset 过；
- 不需要引入"reset 时哪些字段该跳过"这种容易被后续修改者忘记维护的例外
  清单。

> **更正（上一轮）：`ctx.save_for_backward(cache.op_log, cache.op_log_len)`
> 存的是裸引用，不是快照，必须改成 `.detach().clone()`。** `ctx.save_for_
> backward` 保存的是张量对象（指向底层 storage），不会自动 deep copy——这一点
> 上面"不做 reset 豁免"的论证里已经预设了"存进 ctx 的是一份独立快照"，但没有
> 显式写出**怎么**让它独立。本项目一贯的风格是预分配 buffer、`reset_parameters()`
> 原地 `zero_()` 复用（§5.19 tip 3、§5.11 的 pad 槽清零讨论都是同一个模式），
> 不是每次重新分配——`cache.op_log`/`cache.op_log_len` 大概率也是这样实现的。
> 这意味着如果只是把 `cache.op_log` 这个张量对象本身存进 `ctx`（不克隆），
> 下一次 `forward()` 里 `reset_parameters()` 对同一块底层 storage 做原地
> 清零时，`ctx` 里"存"的那个引用指向的数据也会被一起清空。**修法当时是显式
> `.detach().clone()`**——这解决了正确性问题，但引入了一个没算清楚的内存
> 代价，见下一条更正。

> **更正（这一轮修的）：`.detach().clone()` 会让训练峰值显存翻倍，这笔账
> 之前没算过，必须重新设计而不是接受这个代价。** `cache.op_log` 作为
> `reset_parameters()` 管理的持久 buffer，本身就占 448MB（32k/28 层，
> §5.21-2 已算过）；克隆一份进 `ctx` 又是一份独立的 448MB；`ctx` 持有这份
> 克隆直到 `backward()` 跑完才释放，而 cache 自己的 448MB 从来不会被释放
> （它是预分配、常驻的持久 buffer）。**这意味着从"这次 forward() 结束"到
> "这次 backward() 跑完"这段窗口——也就是每个训练 step 都会经过的窗口——
> `op_log` 相关的显存不是 448MB，是 448+448=896MB。** `CLAUDE.md` §4 只写了
> "训练期 op_log 约 448MB"，在有这次克隆之后已经不准确。
>
> **根因是"op_log 到底需不需要`reset_parameters()`那种`预分配一次、原地
> 复用`的持久 buffer 待遇"这个前提没有被重新审视过，只是照搬了其它 buffer
> 的既有模式。** 其它 buffer（ladder entry、centroid、`level_count` 等）
> 需要这种待遇，是因为反复分配大张量的开销真实存在，而它们的内容**不需要
> 活过它们自己所在的这次 forward() 调用**——backward 从不直接读取它们的
> 持久状态，而是通过重放 `op_log` **重建**需要的 ladder 张量（"重放完全
> 不需要 centroid"那段已经确立）。`op_log` 是唯一的例外：它是重放的
> **依据本身**，没有"重建"这回事，`backward()` 必须读到 forward 当时写下的
> 真实内容——这正是它需要活过 `reset_parameters()` 的原因，也是它和其它
> buffer 唯一的本质区别。
>
> **既然这个"必须存活到 backward"的需求是 `op_log` 独有的，解法也应该只
> 改 `op_log`，不是给所有 buffer 引入克隆。修法：`op_log`/`op_log_len`
> 不再是`reset_parameters()`原地`zero_()`复用的持久 buffer，改成
> `forward()`每次调用时用`torch.zeros(...)`重新绑定成一个全新张量对象**
> （不是"清空已有张量的内容"，是"让 `cache.op_log` 这个属性名指向一块
> 全新分配的存储"）。`forward()` 全程往这个新对象里写，结束时直接把它存进
> `ctx`——**不需要 `.detach().clone()`**，因为没有第二个持有者会去修改它：
> 下一次 `forward()` 调用会把 `cache.op_log` **重新绑定**到另一块全新存储
> 上，这次调用留在 `ctx` 里的那个对象完全不受影响（"重新绑定属性名指向新
> 对象"和"原地清零旧对象的存储"是两回事——前者不会影响任何仍持有旧对象
> 引用的人，后者会）。`reset_parameters()` 对**其它**buffer（ladder/
> centroid/`level_count`/`alive`/...）继续用原地 `zero_()` 复用，不受
> 影响——只有 `op_log`/`op_log_len` 这两个字段换成"每次重新分配"，因为
> 只有它们需要"活过下一次 reset"这条独有的性质。
>
> **代价重新核算**：训练期显存回到单份 448MB（32k/28 层），不再翻倍——
> `CLAUDE.md` §4 那句"约 448MB"现在又是准确的，不需要改成 896MB。**多付的
> 代价是分配频率**：`op_log` 从"预分配一次、之后每次 forward 原地复用"变成
> "每次 forward 都重新 `torch.zeros`"——但 PyTorch 的显存缓存分配器在稳态
> 训练循环里会把上一次调用释放的同尺寸块直接复用给这次的 `torch.zeros`
> 调用，不会真的每次都发起底层 `cudaMalloc`，所以这个代价在实践中很小，
> 远小于"峰值显存翻倍"这个代价。**这一轮相对上一轮的改变，是把
> `ctx.save_for_backward` 里的 `cache.op_log.detach().clone()` 换回不带
> `.detach().clone()` 的 `cache.op_log`**——上一轮加克隆是为了解决别名 bug，
> 这一轮发现更好的解法是让 `cache.op_log` 从一开始就不是一个会被复用/清零的
> 别名来源，克隆因此变得不再必要，下面的代码示例同步更新。

#### `op_log` 只能在训练路径分配——buffer 表的措辞和 CLAUDE.md 的"serving 不分配"必须显式对齐

CLAUDE.md §4 的第三笔账明确写着"serving/推理路径不做反向传播，**不分配、不
持有**这块内存"，但本节和 §5.13 的 buffer 表只说"每次 `forward()` 开头重新
绑定成全新分配的张量"——没有说这个 `forward()` 指的是哪一个。

> **更正（这一轮修的）：真正的推理/生成入口不是 `LogStructuredKVCache.forward()`，
> 是 `CausalSelfAttention._log_kv_training_forward()`。** 核对 `log_kv_cache.py:1217`
> 确认 `LogStructuredKVCache.forward()` 从一开始就是显式禁用的 stub——`nn.Module`
> 的 `forward()` 被覆写成直接 `raise RuntimeError(...)`，docstring 讲得很明确：
> "直接调用 cache 无法正确实现 LogKV，因为 attention 必须先纳入当前 token 再提交
> 它们"，本方案的 semantic-cluster 设计不改变这条限制、也没有理由重新启用它。
> 上一版把它当成推理路径的调用入口，会让实现者去一个恒定 `raise` 的方法上挂
> `record_op_log` 分支，字面照做直接不可执行。

训练/推理两条路径各有一个真正的调用入口：训练路径是 `LogKVStreamTrainingAttention.
forward()`（`log_kv_cache.py:1838` 起，自定义 `torch.autograd.Function`，由
`CausalSelfAttention._log_kv_train_lowmem_forward()` 调用）；推理/生成路径是
`CausalSelfAttention._log_kv_training_forward()`（`model.py:1095`，由
`CausalSelfAttention.forward()` 在 `input_pos is not None` 且 `isinstance(self.
kv_cache, LogStructuredKVCache)` 时以 `reset_cache=False, defer_last_single=True`
调用，`model.py:833-838`）——这是 `CausalSelfAttention` 自己的方法，不是
`LogStructuredKVCache` 的方法，内部只调用 `cache.get_attention_state()`/
`cache.add_recent()` 等 cache 方法，从不经过 `cache.forward()`。两个入口都不叫
`forward()` 意味着"哪个 forward() 该分配 op_log"这个问题本身问错了对象；如果不
显式说清楚是这两个函数中的哪一个，实现者完全可能把"每次 forward() 都重新绑定"
读成"两条路径共用的某个 forward() 都要"，推理/生成路径因此会白白背上 448MB/28
层的分配——这不是理论风险，是这句话字面上就能这样读。

**决定：`op_log`/`op_log_len` 的分配（`torch.zeros(...)` 重新绑定）和写入
（Phase 1/3a/2-含内联 3b 的 op 追加）只发生在 `LogKVStreamTrainingAttention.
forward()` 内部。** 两个入口共享同一套路由/flush/ladder 写入逻辑（DP-means
分配、Ward 合并、`compact`/`_binary_carry`——这些和是否训练无关，训推一致性
正是靠"同一条路径"保证的，见 CLAUDE.md §10 死因 1），但这套共享逻辑必须显式
接收一个 `record_op_log: bool` 参数：

```python
def route_and_flush_batch(..., record_op_log: bool):
    ...  # Phase 1（向量化）→ Phase 3a（向量化）→ Phase 2（串行，含内联
         # Phase 3b）完整跑一遍：DP-means 路由、Ward 合并候选选择与执行、
         # segment id / PAD_INSERT count 的向量化 scan、centroid/n_eff/
         # n_total/p_hi_c/current_segment 的 Phase 3a/3b 更新、ladder 物理
         # 写入——训练和推理完全一致，全部读写本地缓冲（local_buffer/
         # local_op_len），不受 record_op_log 影响，见上方"本地缓冲 vs
         # 持久 op_log"一节的更正框
    if record_op_log:
        commit_local_buffer_to_persistent_op_log(...)   # 只有这一步是训练
                                                           # 独有的：把这一批
                                                           # 已写满的本地缓冲
                                                           # 整体拷贝进跨批
                                                           # 持久的 op_log
                                                           # （§5.4 的
                                                           # local_op_len
                                                           # 定长拷贝公式），
                                                           # 供 backward 重放用
    ...
```

**上一版把 `append_ops_to_local_buffer(...)` 整体挂在 `if record_op_log`
之下，这是错的，必须更正**：本地缓冲从 Phase 1 第一次写入到 Phase 2/3b
最后一次写入全程无条件发生（见上方"本地缓冲 vs 持久 op_log"一节的更正
框）——如果字面照抄旧版本这段伪代码，`record_op_log=False` 的推理路径会
连本地缓冲都不构建，Phase 3a 无 op 可读，metadata 更新、Ward 合并候选选择、
ladder 物理写入全部失去输入，路由机制在推理路径上直接失效。真正只属于
训练路径的，只有"把已经写满的本地缓冲提交进跨批持久 `op_log`"这最后一步。

`LogKVStreamTrainingAttention.forward()` 调用时传 `record_op_log=True`，且
只有在这个分支里才会执行"`op_log`/`op_log_len` 重新绑定成全新张量"（forward
一开始，处理第一个 flush 批之前）和"批末把本地缓冲提交进持久 `op_log`"这
两步。`CausalSelfAttention._log_kv_training_forward()`（推理/生成，
`litgpt/generate/base.py`、`speculative_decoding.py` 等触发的 `GPT.forward()`
沿 `input_pos is not None` 分支最终调用到这里，见上方"更正"框）调用同一个
共享函数时传 `record_op_log=False`——跨批持久的 `op_log`/`op_log_len` 这两个属性在推理
路径上应该**从未被访问、从未分配**，不只是"分配了但不用"，因为哪怕只是每次
forward 都重新 `torch.zeros(...)` 一次 448MB 又立即丢弃，也是纯浪费的分配器
压力，且容易让人误以为这块内存"反正都要分配"从而不再警惕。**但这条 gating
只覆盖跨批持久的 `op_log`/`op_log_len`，不覆盖本节开头描述的本地缓冲**——
Phase 1/2/3 的路由决策、metadata 更新、ladder 写入本身，以及驱动它们的本地
缓冲，在 `record_op_log=False` 时同样完整执行，唯一被跳过的是"把这批已经
处理完的本地缓冲再拷贝一份进持久结构"这一步。

**不能靠 `torch.is_grad_enabled()` 做这个判断，必须是显式的调用路径/参数**：
本方案的路由决策（DP-means 距离比较、`argmin`）本身就**恒定** `no_grad`
（CLAUDE.md §10 死因 1"本方案路由同样 `no_grad`"），训练和推理走的是同一条
路由代码，所以"routing 这一步是否在 `torch.is_grad_enabled()` 下"这个信号
在两条路径上是一样的，不能用它区分。真正的区分点是"这次 forward 之后会不会
有一个对应的 `backward()` 调用需要重放 `op_log`"，这个信息只有调用方（是走
`LogKVStreamTrainingAttention.forward()` 还是走 `CausalSelfAttention.
_log_kv_training_forward()`）知道，必须显式传下去，不能从张量的
`requires_grad`/全局 autograd 模式反推。

#### "448MB" 只是每个 in-flight forward 的代价，不是训练期的固定开销——梯度累积/pipeline 会让它按并发数相乘

上面"训练峰值显存"这条更正框把峰值钉死在"单份 448MB"，但那个推导隐含一个
前提：**同一时刻最多只有一个 `ctx` 持有 `op_log` 快照在等待它的
`backward()`**。这个前提对"每次 forward 后立即调用对应的 backward"这种训练
循环成立，但不是任何训练循环都满足它——`ctx.save_for_backward` 是 PyTorch
autograd 的标准机制，只要**下一次 forward() 在这次 forward 对应的
`backward()` 跑完之前发生**，两个 `ctx`（连同它们各自的 448MB `op_log`）就
会同时存活，训练峰值显存因此是**448MB × 同一时刻并存的 in-flight forward
数**，不是一个固定常数。这条必须显式写清楚，否则"约 448MB"这句话会被不加
限定地当成训练期的总开销来做预算——这正是上面"训练期梯度累积场景下，同一个
cache 对象可能在这次 forward 的 `backward()` 被调用之前，就被下一次
`forward()` 调用并 reset 过"这句话已经承认、但没有展开算清楚代价的地方。

**这个仓库现在的梯度累积实现是安全的，不会触发相乘**：`litgpt/pretrain.py:
351-355` 的累积循环是——

```python
is_accumulating = state["iter_num"] % train.gradient_accumulation_iters(...) != 0
with fabric.no_backward_sync(model, enabled=is_accumulating):
    logits = model(input_ids)
    loss = chunked_cross_entropy(logits, targets)
    fabric.backward(loss / train.gradient_accumulation_iters(...))   # 每个
                                                                        # microbatch
                                                                        # 都立即调用
if not is_accumulating:
    optimizer.step()   # 只有这一步被推迟，backward() 每次都执行
```

`no_backward_sync` 只是跳过 DDP 的梯度 all-reduce，**每个 microbatch 的
`fabric.backward()`（进而每个自定义 `Function` 的 `backward()`）依然逐次
立即执行**，被推迟的只有 `optimizer.step()`。所以在这条训练循环下，任意
时刻最多只有一个 microbatch 的 forward 已完成、backward 未完成，`op_log`
峰值就是本节算出的单份 448MB，梯度累积的步数
（`gradient_accumulation_iters`）不参与这个乘法。

**但这是这个仓库当前训练脚本的性质，不是设计本身的保证——以下两类模式会按
并发的 in-flight forward 数把 448MB 相乘，必须显式排除或显式预算，不能假设
"训练期就是 448MB"对它们也成立**：

1. **累积 loss、只在最后调一次 `backward()`**（例如
   `losses = [model(x_i) for x_i in microbatches]; sum(losses).backward()`）
   ——所有 microbatch 的 forward 都先跑完、`ctx` 全部存活，直到最后那一次
   `backward()` 才会按拓扑逆序依次释放。峰值是
   `microbatch 数 × 448MB`。这个仓库当前不用这个模式（见上面 `pretrain.py`
   的分析），但如果未来任何训练脚本（包括 `litgpt/finetune/*.py` 或外部
   使用方）改成这种写法，必须重新核算这笔账，不能沿用"448MB"。
2. **pipeline 并行的 microbatch 调度**（GPipe 式，故意让多个 microbatch 的
   forward 领先于它们各自的 backward，以填满流水线气泡）——这是这种调度
   方式存在的意义本身，peak 是 `pipeline depth × 448MB`。本仓库 `extensions/`
   下的 thunder/xla 扩展如果引入这类调度，必须把这一条计入训练显存预算，
   `op_log` 不会因为"训练本来就该省显存"而自动免于这个乘法。

**activation checkpointing（`torch.utils.checkpoint`）不属于上面两类，但有
一个值得记录的低优先级浪费**：checkpoint 的标准实现是"先在 `no_grad()` 下跑
一次 forward 拿到输出（不建图，这次调用的 `ctx`——如果 `record_op_log=True`
的话——不会被任何东西持有，`op_log` 分配后立即可被回收，不会累积峰值），
backward 时再在有梯度的模式下重新跑一次 forward（重建图）紧接着执行这段的
backward"。所以 checkpointing **不会**把 in-flight 数推高（每个 checkpoint
段落任意时刻最多一个"有效"`ctx`），但会让 `op_log` 的分配次数变成两倍（一次
no_grad 的 throwaway 分配、一次 recompute 的真实分配）——按上面"`op_log`
只在 `LogKVStreamTrainingAttention.forward()` 里分配"这条已经成立，checkpoint
的 no_grad 首轮是否还需要真的分配 448MB、还是可以跳过，取决于 checkpoint
内部是否也传了 `record_op_log`（它应该传 `False`——首轮的目的只是拿输出值，
根本不会有人对它调用 `backward()`）；这是一个可以在实现阶段做的效率优化，
不是本节这笔账的正确性问题，这里只记录下来避免遗漏。

**结论，写进操作性规则**：默认训练循环（forward 后立即 backward，不论
`optimizer.step()` 是否被梯度累积推迟）下，"448MB"是准确的峰值数字；一旦
训练脚本改成"累积 loss 再统一 backward"或引入 pipeline 并行，必须显式按
"同一时刻 in-flight 的 forward 数 × 448MB"重新核算，不能沿用这个数字。

**`op_log`（完整 `(B,G,OP_max,4)`）和 `op_log_len`（`(B,G)`）一起进 `ctx`，
不做切片**：虽然不同 `(b,g)` 的有效长度不同，但保存前按最长有效长度裁剪成
ragged 张量既没必要也麻烦（batched 张量本来就不支持 ragged shape），直接连同
静态形状一起存、backward 时再用 `op_log_len` 截断读取，是最简单、和张量的
矩形约束天然兼容的做法——这和"矩形预分配 + 掩码"这个贯穿全文档的原则是同一
类选择。

#### 5.21-3 carry 必须全 GPU 向量化：`_counts` 的 host 镜像要**删掉**而不是扩展

现实现（`log_kv_cache.py:393` 附近）里 `level_count` 是设备张量，但 `self._counts` 是
它的 **host 端镜像**，注释写明存在理由就是避免 GPU sync；而 `if self._counts[0] >=
self.B`、`if self._counts[ell] == 0` 这些**控制流全读它**。

一旦按 `(B,G,K,L)` 展开，每个 (batch, KV头, 簇) 的进位深度都不同，**Python 标量分支
在语义上就不成立了**；硬要保留 host 镜像则每次 flush 每层都要一次设备→主机同步，
性能上等于自杀。

**决定：循环轴换掉。**

```
外层：for ℓ in range(L_alloc)      ← Python 循环，但界是静态的（11~15），与数据无关
内层：对全部 (B,G,K) 同时做带掩码的 carry —— 纯张量操作，不读任何标量
```

关键认识是**进位的级联深度虽然数据相关，但被 `L_alloc` 静态界住**。所以把"按需级联"
改成"固定跑 `L_alloc` 轮、每轮用掩码决定谁真的进位"，就把数据相关性从控制流挪进了
掩码。代价是不能提前退出，每次 flush 固定 `L_alloc` 次 kernel——而 §5.4 把 flush 粒度
从 2 提到 128 之后，flush 频率降了 64 倍，这个代价被吸收掉了。

**所以 `_counts` 的 host 镜像在语义簇路径上必须删除，不能扩展。** 保留它等于保留一个
每 flush 一次的同步点。CPU 参考实现（§5.18 第 2 步）可以继续用 Python 控制流——它本来
就只用于 S0.8 的对拍。

#### 5.21-4 `s_h` 先选**离线标定**，在线估计降级为消融

`λ_rel · s_h` 直接决定 `K_eff`、needle 隔离率和跨层可比性，它不是小空白，是阈值体系
的地基。两种方案的取舍：

| | 在线估计 | **离线标定（选这个）** |
|---|---|---|
| 阈值随序列演化 | 是 → **引入 §11-E 的新变体**（训练与推理的估计轨迹不同 ⇒ 簇划分不同）| 否，常量 |
| 序列开头 | 估计未收敛，而 §5.6 说早期路由错误**不可恢复** | 从第一个 token 起就正确 |
| 跨层可比性 | 各层估计器收敛速度不同，S0.2 曲线不可比 | 天然可比 |
| 成本 | 零 | 一次标定 pass |

**决定：v1 用离线标定。** 在留出样本上跑一次前向，记录每 (layer, KV group) 的
`E‖k − k̄‖²`（`k̄` 是整个标定集上的全局均值，定义见 §5.2），存成常量张量随
checkpoint/config 走。**标定常量必须写进 eval 的 metadata 字段**——否则不同 run
用了不同标定值却无从分辨，所有对比作废。在线估计作为后续消融项，不进 v1。

#### 5.21-5 Stage 0 的 dump 脚本排在生产实现之前

规格见 [`experiments.md`](experiments.md) 的"Stage 0 dump 规格"一节。它本来就是
Stage 0 全部结论的输入，却一直只有一句"dump 每层每头 pre-RoPE k/v"，没有可执行细节。
**这个脚本应当是本项目写的第一段代码**，排在 §5.18 的第 1 步之前。

> **更正（这一轮修的）："第一段代码"这句话覆盖的是 S0.0–S0.7 和 S0.8 的第
> 1/2/3a 项，不覆盖 S0.8 的 3b 项——3b 有它自己独立的、更晚的前置依赖，
> 之前没有显式说清楚，字面读会和 §5.18 的步骤顺序打架。** S0.8 3b（attention
> 读出的相对 L2 误差，§6 的硬性决策门）需要 `cache_batch`/`cache_serial` 这两个
> 由 §5.4/§5.18 第 2 步"CPU 参考实现的路由"构造出来的 cache 状态，还需要
> `log_kv_slot_attention()`/`get_attention_state()` 已经按 §5.14/§5.20-B
> 扩展出 `CacheAttentionState`/`slot_valid`/`M_s`（3b 的设计明确要求"直接、
> 原样传递生产函数的返回值，不做手工重建或平行实现"，理由是避免测试代码和
> 生产代码分叉——这条原则本身没有问题，但它意味着这两块代码必须先存在）。
> 按 §5.18 的步骤编号，这是"第 1 步（`log_kv_position.py`）+ 第 2 步（CPU
> 参考路由）+ `log_kv_slot_attention`/`get_attention_state()` 的接口扩展"，
> 晚于第 0 步（dump 脚本本身），字面上和"dump 脚本排在任何生产代码之前"
> 冲突。
>
> **这不是需要靠改期望解决的矛盾，是"生产代码"这个词在两处指代的范围不同，
> 需要把边界画清楚。** CLAUDE.md §0 用 Stage 0 的结果决定"是否继续 Stage 1
> （生产代码）"，真正被这道门挡住、需要先看到信号才值得投入的，是 §5.18
> 第 3–6 步——多簇路由的向量化实现、段对齐填充、`op_log` 训练路径重放，
> 这些是本方案唯一的、有实际工程量和回退风险的核心投入。3b 依赖的两块东西
> 性质不同：
> 1. **§5.18 第 2 步的 CPU 参考实现**在自己的定义里就是"朴素串行版，慢但
>    正确"——它存在的唯一目的是当分歧率测试的对照基准（S0.8 本来就需要它，
>    这条依赖从第一版起就写在第 2 步的说明里，不是这一轮新加的），不是要
>    部署的代码，写它不构成"要不要投入 Stage 1"这个决策的组成部分。
> 2. **`log_kv_slot_attention`/`get_attention_state()` 的接口扩展**
>    （`CacheAttentionState`、`slot_valid`、`M_s`）——**更正（这一轮修的）：
>    上一版说这是"纯加法式改动，不改变现有调用不传这些参数时的行为"，这句话
>    不成立，必须删掉。** `get_attention_state()` 的返回类型对**任何模式**
>    （包括 legacy）都改成恒定 10 字段的 `CacheAttentionState`（§5.14），
>    现有位置解包调用（`slot_k, slot_v, slot_w = cache.get_attention_
>    state(...)` 这类 3-/8-变量解包）在返回值变成 10-元组后会直接
>    `ValueError: too many values to unpack`——是**breaking 迁移**，不是
>    加法。真正成立的是：这次迁移**动作本身**是机械的（生产代码 6 处 +
>    测试约 25 处，全部是"位置解包换成按字段取值"这一种模式，不涉及任何
>    新算法决策，必须原子完成，见 §5.20-B 调用点迁移清单最后一行的更正
>    框），且是本节唯一真正触及 `litgpt/log_kv_cache.py` 生产文件的一步，
>    但改动量和风险与第 3–6 步的多簇路由实现不是一个量级——**它本身就是
>    Stage 0 决策门（3b）能够可信的前提**，所以把它划进"S0.8 需要先落地
>    的最小基础设施"比划进"等 Stage 0 结果出来再做的 Stage 1 投入"更准确。
>    §5.19-1 的"默认关闭字节等价"CI 闸门与这次迁移无关——那条闸门管的是
>    开关关闭时**数值**是否不变，管不到这里"函数返回元组长度变了、调用方
>    直接 `ValueError`"这类 Python 层面的接口问题，不能拿来当这次迁移的
>    安全网。
>
> `log_kv_position.py`（第 1 步）已经在 CLAUDE.md §0 的"下一步"清单里被列为
> 独立于 Stage 0 决策门之外的待办项，这里是同一个先例的延伸，不是新开的口子。
> 需要补写进实现顺序的具体位置见下方 §5.18 的对应说明。
