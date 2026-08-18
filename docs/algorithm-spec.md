# SemanticLogKV 算法规格与实现方案

> 本文件是 [`../CLAUDE.md`](../CLAUDE.md) 的第 §5 章，独立成文只为便于阅读和维护。
> **章节编号沿用全局编号**，所以文档间的交叉引用（`§2.3`、`§11-A`、`§5.19-2` 等）
> 全部有效。设计动机和为什么这样选，看 CLAUDE.md 的 §2；实验协议看
> [`experiments.md`](experiments.md)；风险与未决问题看
> [`risks-and-open-questions.md`](risks-and-open-questions.md)。
>
> **动手写代码之前先读 §5.19（实现 tips 与易错点）和 §5.20（复用边界），
> 以及 risks 文档的 §11（技术难点清单）。**

## 5. 算法规格与实现方案

### 5.1 记号与参数

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

**用原始（未归一化）pre-RoPE key 上的平方欧氏距离，不用 cosine。** 理由不是习惯，
而是这一个量同时是三件事：

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

Phase 2（串行，只处理 orphan）:
    s*[t] > λ_new 的 token 需要开新簇，它们之间还可能互相成簇
    在这批 orphan 内部跑一个小 DP-means（O(m_orphan²) 的距离矩阵即可）

Phase 3:
    批末统一更新 centroid 一次（§5.5）
```

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
分歧率**——S0.8。`p_hi_c` 冻结这条单独更值得关注：见 §5.21-2 里对它的展开分析
（同批内连续同簇 token 会因为看到"批前"的 `p_hi_c` 而被误判成隔了很久）。

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
和 `p_hi_c` 同一档：**Phase 3 是它唯一的写入点**（walk 本批 ops 时，每遇到一条
某簇的 `NEW_SEGMENT`，就把该簇的 `current_segment` 更新成这条 op 的 `segment`
字段——因为 Phase 3 本就按时间顺序逐簇处理，这个赋值天然收敛到本批最后一次
`NEW_SEGMENT` 的值，不需要额外的 `max`），`ward_merge_only` 步骤 1 合并两个既有
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
> 本身（`derive_final_cluster` 那节已经用 `(slot, epoch)` 版本化身份解决
> 过一次），不是 segment 机制独有的新问题。**结论：任何需要跨这类边界做
> 分组/画图的分析工具，必须按 `(slot, epoch, segment)`（复用
> `derive_final_cluster` 已经维护的同一套 `epoch`）做 key，不能只用裸
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
先固定记号：本批 `m` 个 token 按到达顺序排好，`c*[t]` 是 Phase 1 算出的目标簇，
`new_seg[t]` 是 §5.3 第二判据算出的布尔值（是否开新段，仍然用批前冻结的
`p_hi_c`，这部分近似不变）。

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
ladder 写入的要求同一个模式——写一个逐 token 串行的朴素参考实现（对每个到达
的 token，读当前 `current_segment`/`level_count[cluster,0]`，立即决定
`segment`/`count`，立即"执行"更新，供下一个 token 读到最新值），断言它和上面
向量化公式在任意合成批次（含同簇多次开新段、含多个不同簇交错到达）上逐位
一致。这条测试没通过之前，"segment id / PAD_INSERT count 的批量化是对的"这个
论证是未经验证的假设。

**上面只算出了每个 token 的 `segment`/`count`，还没说这些 op 怎么写进本地缓冲
——`PAD_INSERT` 和它服务的 `NEW_SEGMENT` 是两条 op，主操作永远只有一条，每个
token 展开出的 op 数量不一样，这是一个变长写入问题，Phase 1 是向量化路径，
不能逐 token 决定"我这条写在第几行"。**

```
extra[t]    = new_seg[t] and (count[t] > 0)   # 这个 token 除了主操作，还需要
                                                # 额外一条 PAD_INSERT
main_idx[t] = base + t + inclusive_prefix_sum(extra)[t]   # base 是本地缓冲
                                                             # 提交前的 local_op_len
                                                             # （Phase 1 总是本批
                                                             # 第一个写入者，实践中
                                                             # base=0，但公式写通用
                                                             # 形式）
若 extra[t]：pad_idx[t] = main_idx[t] - 1
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

**这个技巧的适用范围不止这里，顺带记一笔留给以后**：Phase 3 对 `n_eff`/`centroid`
的在线更新（§5.5）在没有 `γ` 衰减重启（即本批内该簇没有 `NEW_SEGMENT`）时会
逐项相消、化简成一个简单的批量加权和，但一旦出现 `NEW_SEGMENT`（触发 `γ` 衰减
重启），就需要和上面完全同构的"segmented reset scan"才能向量化——目前文档只说
Phase 3"逐簇按 §5.5 的公式更新"，没有展开怎么向量化这个递推，这里不重复推导，
只留一个指针：需要的话，套用上面 `steps_since`/reset 的思路，不是另一个新问题。

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

```python
def derive_final_cluster(
    op_log,
    token_offset: int = 0,
    initial_epoch: dict[int, int] | None = None,
    initial_parent: dict[tuple[int, int], tuple[int, int]] | None = None,
) -> tuple[dict[int, int], dict[int, int], dict[tuple[int, int], tuple[int, int]]]:
    """只读、离线，供调试/S0.8 对拍/下游分析工具使用；从不写回 op_log，也不是
    forward/backward 正确性契约的一部分——这一点必须显式声明，否则容易被误用成
    重放的一部分。对 WARD_MERGE 的合并方向做一次并查集(union-find)，但身份是
    (slot, epoch) 而不是裸 slot——见上方更正框，槽位复用后的新身份不能被误认成
    被合并走的旧身份。op_log 本身已经完整记录了每次合并的方向，这里只是在
    不可变记录上做一次只读遍历，和"篡改历史记录"是两件完全不同的事。

    **前提（函数不做任何隐式校验，调用方必须自己保证）**：
    1. `op_log` 参数必须是**有效前缀**（按 §5.13 新增的 `op_log_len` 截断，
       不是整个静态 `(OP_max, 4)` buffer——尾部未写入的行不是合法 op，混进来
       会产生未定义行为，见下方"op_log 的有效长度"一节）。
    2. 若这是从冷启动（空 cache，所有槽从未被 `alive` 过）开始的完整、连续
       日志：`initial_epoch`/`initial_parent` 留默认 `None`（等价于假设这段
       之前没有发生过任何 `NEW_CLUSTER`/`WARD_MERGE`，对冷启动完整日志这个
       假设成立）。
    3. 若这只是一段切片（比如某一个 flush 批自己的本地缓冲，或者想分段处理
       一条很长的 op_log 而不是一次性喂完）：**必须**传入这一段开始之前的
       `epoch`/`parent` 状态——用上一次调用本函数返回的 `final_epoch`/
       `final_parent` 直接传进来即可串联，见下方"串联调用"。不满足 1-3 条
       之一，结果不可信。

    返回三元组：(token_idx -> 最终槽号，token_idx 已经加上 token_offset；
    本段结束时的 epoch 状态；本段结束时的 parent 状态）。后两者可以原样作为
    下一段调用的 `initial_epoch`/`initial_parent`。"""
    epoch = dict(initial_epoch) if initial_epoch else {}    # 槽号 -> 当前 epoch
    parent = dict(initial_parent) if initial_parent else {}  # (槽号,epoch) -> (槽号,epoch)
    def find(v):
        while parent.get(v, v) != v:
            parent[v] = parent.get(parent[v], parent[v])
            v = parent[v]
        return v

    token_identity = {}               # token_idx（本段内，从 0 开始）-> 记录时的版本化身份
    token_ptr = 0
    for op in op_log:
        if op.type == NEW_CLUSTER:
            epoch[op.cluster] = epoch.get(op.cluster, -1) + 1   # 首次建立 -1->0，
                                                                  # 每次复用再 +1
            token_identity[token_ptr] = (op.cluster, epoch[op.cluster])
            token_ptr += 1
        elif op.type in (JOIN, NEW_SEGMENT):
            token_identity[token_ptr] = (op.cluster, epoch.get(op.cluster, 0))
            token_ptr += 1
        elif op.type == WARD_MERGE:
            keep_v = (op.keep_slot, epoch.get(op.keep_slot, 0))
            free_v = (op.free_slot, epoch.get(op.free_slot, 0))
            parent[find(free_v)] = find(keep_v)   # 只 union 当前这一代，不碰 epoch 本身

    final_slot = {token_offset + t: find(v)[0] for t, v in token_identity.items()}
    return final_slot, epoch, parent
```

**串联调用**：处理一条很长的 `op_log`（或按 flush 批分段的本地缓冲序列）时，
不需要一次性喂完整日志——`epoch`/`parent` 就是这段计算的全部"记忆"，把它们
从上一段的返回值原样传进下一段的 `initial_epoch`/`initial_parent`，`token_offset`
累加上一段消费掉的 token 数，就能保持和"一次性喂完整日志"完全等价的结果，
`find` 的并查集路径压缩也会在多段调用之间正确保留（因为 `parent` 是原样
传递，不是每段重新清空）。

**不新增任何持久 buffer,不改变 §4/§5.13 的内存账目**——`epoch`/`parent` 是
调用方自己持有、自己决定生命周期的普通 Python 对象，不属于 cache 状态的一
部分。S0.8 对拍、调试工具需要"token 最终去了哪"时调用它,forward/backward 的
正确性路径永远不依赖它、也不会被它影响。

**本地缓冲 vs 持久 `op_log`**：本节说的"本地 op 缓冲"是 Phase 1/2 处理**当前
这一个 flush 批**期间用的临时张量，**不是**跨批持久存在的 `op_log`
（`(B,G,OP_max,4)`，见 §5.13）。批内 Phase 1（一次性向量化写入）和 Phase 2
（串行循环，逐个处理 orphan）都往这个本地缓冲里追加，整个批处理完（Phase 3
之后）才**一次性**把本地缓冲追加进持久 `op_log`。这个两级结构本身不是新设计
——批量化路由本就需要一个地方暂存"这一批算出来的 op"；**它纯粹是一个提交
粒度问题，和上面推翻的重定向无关**——本地缓冲和持久 `op_log` 一样，条目一旦
追加就不再改写，两者遵守同一条不变量，只是提交时机不同。

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
    4. slot_idx = free_slot   # 释放出来的槽立即交给这个 orphan（组）的新簇使用
       —— 这一步开始就是 allocate_new_cluster(slot_idx, group) 这个共享原语
       的内容（见本小节末尾"K 未满/冷启动复用同一个原语"一段），**只**初始化
       结构性状态：alive[slot_idx]=true，
          centroid/n_eff/n_total/p_hi_c/current_segment 全部清零（"白纸"状态，
          不沿用旧簇残留值）。
          **数值元数据不在这里赋值**——统一交给 Phase 3 从本批完整的 op 序列
          重新构造（见下方"Phase 3 的簇级元数据更新"），避免和 Phase 3 的更新
          重复计入同一批 token
       append NEW_CLUSTER(slot_idx, 0, -1) 到本地缓冲，对应这个 orphan（组）
       里**到达顺序最早**的那个 token（绝对位置记为 p0）
       local_p_hi[slot_idx]    = p0   # Phase 2 私有的临时状态，只活在本次
       local_segment[slot_idx] = 0    # Phase 2 调用期间——不是 p_hi_c/全局
                                        # segment 计数，见下方说明

    若这个 orphan 组里还有其它成员（mini DP-means 分进同一临时簇的其余
    token，按到达顺序逐个处理）：**它们不能再产生 `NEW_CLUSTER`**——每个 token
    恰好对应一个主操作，`NEW_CLUSTER` 已经被组内最早的成员用掉了。**它们的
    `JOIN`/`NEW_SEGMENT` 判据必须读 `local_p_hi[slot_idx]`，不能读全局
    `p_hi_c[slot_idx]`**——全局 `p_hi_c` 此刻仍是第 4 步刚清零的占位值，要到
    整个批次处理完、Phase 3 跑完才会变成真实值；如果这里读全局值，
    `p_t − p_hi_c` 恒等于 `p_t − 0`，对任何有意义长度的文档都会远超 `g_max`，
    组内除最早成员外的所有成员都会被错误地判成"时序打断"，被迫各自开一个新
    segment——而它们本来就是同一次 mini DP-means 判定为彼此接近、大概率也在
    时间上紧挨着到达的一组 token。对每个后续成员（绝对位置 `p_t`），按组内
    到达顺序：

        if p_t − local_p_hi[slot_idx] > g_max:
            local_segment[slot_idx] += 1
            append NEW_SEGMENT(slot_idx, local_segment[slot_idx], -1) 到本地缓冲
            （照常触发 §5.11 的 PAD_INSERT，插在这条 NEW_SEGMENT 之前）
        else:
            append JOIN(slot_idx, local_segment[slot_idx], -1) 到本地缓冲
        local_p_hi[slot_idx] = p_t   # 不论 JOIN 还是 NEW_SEGMENT，p_hi 都要推进
                                       # 到"最近一次收到成员的位置"——这是 p_hi
                                       # 的定义（§5.13），和是否开新段无关

    这和"一个已存在的簇后续收到新成员"用的是同一套判据（§5.3 第二个判据），
    唯一的特殊之处是这个簇是本批刚建的，所以判据读的是 Phase 2 自己维护的
    局部状态而不是全局 buffer。
```

> **`local_p_hi`/`local_segment` 只是 Phase 2 处理单个 orphan 组时的临时脚本
> 状态，不是新增的持久 buffer，不进 §4/§5.13 的内存账目，Phase 3 也不需要读
> 它。** 它按 `slot_idx` 存在一个小 dict/scratch 数组里，某个 slot 被第 4 步
> 重新 `NEW_CLUSTER` 时（不论是首次建立还是本批内被 Ward 合并释放后再次复用）
> 直接覆盖重置，不需要跨 orphan 组保留，处理完当前 orphan 组即可丢弃。Phase 3
> 判断"这是不是一次新 segment"直接读 op 类型本身（`NEW_SEGMENT` vs `JOIN`），
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
里"数值元数据不在这里赋值，统一交给 Phase 3 重新构造"这条设计（依赖
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

**Phase 3 是本批"主操作 token 内容贡献"唯一的写入点——对 Phase 1 分配到的既有簇
和 Phase 2 新建的簇一视同仁，不再有第二个入口。** 这是相对上一轮的一处简化，
直接消掉了一类此前存在的 bug：上一轮让 Phase 2 的"新簇写进去"那一步（旧编号第 5
步）**同时**给新簇的 centroid/`n_eff`/`n_total`/`p_hi_c` 按 orphan（组）内容设
初值，随后 Phase 3 又对**整个**本地缓冲统一跑一遍同样的更新——这会把新建簇的
orphan token 计入两次：一次在 Phase 2 初始化时，一次在 Phase 3 统一更新时。上面
第 4 步已经改为"只置 `alive=true`，数值元数据全部清零"，原因就在这里：**Phase 2
不再对任何 token 的内容做数值更新，Phase 3 是唯一做这件事的地方，天然不会有
双计数**。

> **这句话必须精确到"token 内容贡献"，不能笼统说成"Phase 3 是这些字段唯一的
> 写入点"——后者是过强表述，会和 §5.6 的 `ward_merge_only` 直接冲突。**
> `ward_merge_only` 的步骤 1（"合并簇级元数据"）本来就会写 `μ_new`/`n_eff_new`/
> `n_total_new`/`p_hi_new`——这是必须存在的第二个写入来源，不是需要消灭的
> 冗余：它写的是**把两个既有簇已经积累的历史值合并成一个**，不处理本批任何
> 一个 token 的原始内容，和 Phase 3 处理的"这批新到达的 token 该怎么在线更新
> centroid"是两类不重叠的写入。两者不冲突，是因为它们在时间上和作用对象上都
> 天然分开：
> - `ward_merge_only` 只在 Phase 2 处理某个 orphan、触发 Ward 合并腾位时才执行，
>   写入的是 `keep_slot`（被保留的那个簇）的元数据——把 `keep_slot` 和
>   `free_slot` 合并前各自的历史值组合成一份新的历史值。
> - Phase 2 紧接着把 `free_slot`（被释放、即将复用为新簇的那个槽，和上面的
>   `keep_slot` 是**不同的槽**）的数值元数据清零，为它即将承载的新簇准备一张
>   白纸——这一步不读、不写 `keep_slot` 的任何字段。
> - Phase 3 在本批 Phase 1、Phase 2（含其中所有 Ward 合并）全部完成之后才运行，
>   它读到的每个簇的起点值——不论是 `keep_slot` 的"合并后历史值"还是
>   `free_slot`（复用后）的"清零白纸"——都已经是这一批唯一、确定的最终状态，
>   Phase 3 只需要在这个起点上按 §5.5 的公式叠加本批 token 的贡献，不需要关心
>   这个起点是"未被合并的原值"还是"刚被合并出来的新值"，两种情况用的是同一套
>   在线更新公式。
>
> 所以准确的分工是：**Ward 合并（`ward_merge_only` 步骤 1）是"结构性合并事件"
> 的元数据写入点，处理的是既有历史值的组合；Phase 3 是"本批 token 内容"的唯一
> 元数据写入点，处理的是新到达内容的在线更新。** 二者的写入范围（`keep_slot`
> 的历史值 vs. 本批所有被触碰的簇的新内容增量）不重叠，顺序上 Ward 合并总是
> 先于 Phase 3（它发生在 Phase 2 内部），所以 Phase 3 读到的永远是"本批全部
> 结构变动都已经落地之后"的起点，不存在竞争或覆盖。

Phase 3 的具体做法：按本批本地缓冲（Phase 1 + Phase 2 全部写完之后，`op_log`
从不改写，所以这就是最终版本，不需要额外等待或过滤任何"修正"）里的主操作
（`NEW_CLUSTER`/`JOIN`/`NEW_SEGMENT`），逐簇按 §5.5 的在线均值公式更新
`centroid`/`n_eff`/`n_total`/`p_hi_c`/`current_segment`（后者只在遇到
`NEW_SEGMENT` 时被那条 op 的 `segment` 字段覆写，`JOIN` 不改它）。**新建簇的
第一个成员不需要特判**——
Phase 2 已经把它的 `n_eff` 清零，§5.5 公式里 `n_eff_pre == 0 时 μ_c ← k` 这条
分支（本就是为"冷启动或 γ=0 归零"设计的）会自动接住"这是一个刚建立、从未有过
成员的簇"这个情形，和"γ=0 导致的段内归零重启"是同一段代码，不需要为"这个簇是
不是本批新建的"专门分叉。**顺序要求**：同一个簇收到的多个成员必须按它们在
本地缓冲里的相对顺序被应用（在线均值本身是顺序敏感的），不同簇之间彼此独立、
可以任意顺序或并行处理——这和 §5.21-2 对批量 ladder 写入提的"保序"要求是同一类
约束，同一个理由（"结果只依赖同簇成员的相对到达顺序，不依赖处理粒度"）。
**执行时机**：Phase 3 必须在本批 Phase 1 和 Phase 2 全部处理完之后才运行——这
本来就是三个 Phase 顺序执行的自然结果，现在不再有"重定向有没有执行完"这个额外
要等待的条件。

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
3. **Phase 3 元数据更新的正确性（S0.1，逐位精确，与第 1 条完全独立的一条
   断言，不要和"重放正确性"共用同一个测试）**：用同一批合成数据，分别用
   (a) 生产路径（Phase 1/2/3 完整跑一遍）得到的最终
   `centroid`/`n_eff`/`n_total`/`p_hi_c`/`current_segment`，和 (b) 把本批最终
   本地缓冲里原样的
   主操作序列喂给一个独立的、按 §5.5 公式逐簇顺序重算的参考实现（本质上是
   Phase 3 逻辑的一份朴素、非批量化复刻），断言
   两者逐位一致，且**新建簇（本批内 Ward 合并腾出槽位后建立的那些）的最终
   `n_total` 精确等于它实际收到的 token 数，不多不少**——这条直接抓住"Phase 2
   初始化和 Phase 3 更新双计数"这类 bug：如果双计数复现，这里的 `n_total`
   会系统性偏大。**这条断言存在的意义是把"重放/回放需要什么"和
   "调试工具/S0.8 对拍想知道什么"彻底分开成两个独立契约**——前者只服务
   backward 的梯度正确性，后者只服务分析工具，任何时候都不应该被混进同一个
   正确性等级里。

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
   **这是 `keep_slot` 元数据的合法写入点，和 §5.4 的 Phase 3 不冲突**——这里写的
   是"两个既有簇已积累的历史值怎么组合"，Phase 3 写的是"本批新到达 token 的内容
   贡献"，两类写入不重叠也不需要互相知道对方，详见 §5.4 那条更正框）
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
| `current_segment` | `(B,G,K_max)` | int32 | 该簇当前最新的 segment id，下一次开新段用 `current_segment+1`（见 §5.4"Phase 1 缺持久 segment 状态"一节）。**写入点和 `p_hi_c` 同一档**：Phase 3 逐簇 walk 本批 ops 时被覆写，`ward_merge_only` 合并时取 `max` |
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
| `op_log` | `(B,G,OP_max,4)` | int32 | **操作日志，不是 token→cluster 映射**，只记 token 归属不足以重建结构，完整语义/顺序/重放算法见 §5.21-2 |
| `op_log_len` | `(B,G)` | int32 | `op_log` 当前**有效**行数——`op_log[b,g,:op_log_len[b,g],:]` 才是已写入的合法内容，之后的行是未写入/未定义，**任何遍历 `op_log` 的代码（重放、`derive_final_cluster`、S0.8 对拍）都必须先按这个长度截断，不能扫整个 `(OP_max,4)`**，见 §5.21-2 的更正框 |

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

```python
def log_kv_slot_attention(
    q, slot_k, slot_v, slot_w, scale,
    mask=None,           # 不变：(T_q, S) bool，仍与 causal_tail 互斥
    causal_tail=0,        # 不变：仍要求 causal_tail == T_q
    slot_valid=None,      # 新增：(B, G, S_pooled) bool，只盖 pooled 前缀，
                           # 可以和 causal_tail 同时使用，也可以和 mask 同时使用
                           # ——它和另外两者不是同一个轴（entry 级有效性 vs
                           # query-time 因果可见性），不存在互斥关系
    ...
):
```

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
- `get_attention_state()` 额外返回有效位掩码与每 entry 的 `M_s`；
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
   排在生产代码之前（§5.21-5）。
1. **`log_kv_position.py` + 单测**（纯 CPU，不依赖任何 dump）。
2. **CPU 参考实现的路由**（朴素串行版，慢但正确）——它同时是 S0.8 的对照基准，
   不要跳过。
3. **`K_max=1` 单簇路径**：验证退化到"单条位置序 ladder"，与现有实现数值对齐
   （注意 §12 的容差问题）。
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

5. **`op_log`（连同 `op_log_len`）要在 `forward()` 结束时存进 `ctx`，不要给
   cache 对象的 reset 生命周期开特例。** `LogKVStreamTrainingAttention.forward`
   开头就调 `cache.reset_parameters()`，而 backward 要重放 forward 期间记录的
   路由；**定案是把 `op_log`/`op_log_len` 和 `q`/`k_raw`/`k_roped`/`v` 一起存
   进 `ctx`**（§5.21-2 的"`op_log` 的有效长度"一节有完整推导和理由），
   backward 只读 `ctx` 里的快照，cache 对象自己的 `op_log`/`op_log_len` 该在
   下次 `forward()` 被 `reset_parameters()` 清空就清空，不需要任何豁免逻辑。
   **这个时序冲突不处理的话，训练会静默地用错误的梯度。**

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
| `get_attention_state()` | 多返回一个有效位掩码与每 entry 的 `M_s` |
| cache 入口 | 收 pre-RoPE k + 绝对位置，而不是 post-RoPE k |
| `level_w`/entry `w` 的 dtype | 不能继承 activation dtype（现有 `log_kv_cache.py:346-349` 是 `torch.zeros(..., dtype=dtype)`，跟着 fp16/bf16 走）。fp16 整数精确表示上限是 2048、溢出上限 65504；1M 上下文下一个高冗余大簇的 `w` 可以到几十万，**必须 fp32 或 int32**，`log(w/M)` 之前再转 fp32 |

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
    # 必须 .detach().clone()，不能存裸引用——见下方"ctx 存的必须是快照，
    # 不是引用"一节，cache.op_log/op_log_len 是会被下一次 reset_parameters()
    # 原地清零复用的持久 buffer，直接存引用会让 backward 读到已经被清空的内容
    ctx.save_for_backward(
        q, k_raw, k_roped, v,
        cache.op_log.detach().clone(), cache.op_log_len.detach().clone(),
    )
    ...

@staticmethod
@once_differentiable
def backward(ctx, grad_y):
    q, k_raw, k_roped, v, op_log, op_log_len = ctx.saved_tensors
    # op_log/op_log_len 是 forward 结束时的独立克隆，不是 cache 对象的引用——
    # 重放只读这份快照，不碰 cache 对象自己的 op_log/op_log_len，也不需要关心
    # cache 对象在这之间发生了什么（包括被下一次 forward() 的
    # reset_parameters() 清空）
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
NEW_CLUSTER   (slot_idx,  0,        -1)      # 语义 novelty：在哪个空槽建新簇，
                                              # segment 恒为 0（新簇的第一段）
NEW_SEGMENT   (cluster,   new_seg,  -1)      # 时序打断：同簇开新段
JOIN          (cluster,   segment,  -1)      # 归入既有簇的当前段

── 结构操作：不消费 token，作为主操作的副作用穿插出现 ──
WARD_MERGE    (keep_slot, free_slot, -1)     # 给某个 NEW_CLUSTER 腾位，必然紧邻其前
PAD_INSERT    (cluster,   level,    count)   # 给某个 NEW_SEGMENT 做对齐填充，必然紧邻其前
CARRY         (cluster,   level,    resulting_count)  # ladder 进位，**非权威、可选**，见下方说明
```

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

**顺序规则**：`op_log` 按 token 到达顺序线性写入。主操作严格一一对应"下一个待处理的
token"；结构操作插在触发它的主操作旁边（`WARD_MERGE` 在它服务的 `NEW_CLUSTER` **之前**
——先腾位再建簇；`PAD_INSERT` 也在它服务的 `NEW_SEGMENT` **之前**——先把上一个段的
尾部对齐填好，再让新段的第一个 token 进场；`CARRY` 跟在导致进位的那次 ladder 追加
后面）。

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
token_ptr = 0
for op in op_log[:op_log_len]:   # 只遍历有效前缀，§5.13 新增的 op_log_len；
                                   # 扫整个静态 (OP_max,4) buffer 会把尾部未
                                   # 写入的行当成合法 op，见下方"op_log 的
                                   # 有效长度"一节
    if op.type in {NEW_CLUSTER, JOIN, NEW_SEGMENT}:
        # 消费 token_ptr 指向的原始 (k_raw, v, pos) —— 位置必须一起传，锚点
        # (p_lo/p_hi/sum_wp) 是绝对位置的函数，缺了 pos 这一步没法定义。
        # 按 op 指定的 (cluster, segment) 追加进该簇的 ladder —— 用的是
        # compact()/_binary_carry() 那套纯算术，不依赖任何浮点比较，所以给定
        # 同样的 (token, pos, cluster, segment) 四元组，结果必然逐位相同。
        append_to_ladder(k_raw[token_ptr], v[token_ptr], pos[token_ptr],
                          op.cluster, op.segment)
        token_ptr += 1
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
token 调用）。`op_log` 里的线性 token 顺序因此是一个**逻辑顺序**——"如果这批 token
被逐个串行处理，会产生的顺序"——而不是 forward 实际执行的物理顺序。重放算法（上面
那个 for 循环）假定"按 `op_log` 顺序逐条串行执行"和"forward 的批量向量化实现"
产生**逐位相同**的最终 ladder 状态，这个假定成立的理由和它对实现提出的具体要求：

- **成立的理由**：`compact()`/`_binary_carry()` 的正确性只依赖"同一个簇收到的成员
  按什么**相对顺序**到达"，不依赖"这些成员是被一次一个地处理，还是被一批一起写入
  的"——`p_lo`/`p_hi` 取 min/max、`sum_wp` 是整数加法，这两类运算本身与批处理粒度
  无关；真正对顺序敏感的是 `compact()` 的"时间序相邻配对"语义（§5.7 已核实的
  docstring），只要落进同一层的成员在被配对之前，其相对到达顺序和串行版一致，
  配对结果就一致。
- **对实现的具体要求**：批量写入必须**保序**——一个 flush 批内，凡是路由到同一个
  `(cluster, level=0)` 的 token，必须按它们在批内的原始位置顺序（`t=0..m-1`，也就是
  真实的到达顺序）被写入/参与 carry，不能因为向量化 scatter 而被打乱。这一点对
  当前设计**几乎是免费的**——Phase 1/2/3 的路由结果 `c*[t]` 本就是逐位置索引对齐的
  张量，没有引入任何重排；真正需要小心的是**批量 carry 本身**：如果一批里有
  `K > 1` 个 token 落进同一个此前尚有空位的 `(cluster, level=0)`，向量化实现必须
  用等价于"依次单个 carry 调用 `K` 次"的方式处理这批到达（标准二进制计数器的
  "批量加 `K`"和"逐一自增 `K` 次"在最终数字模式上永远一致，这是可以直接引用的
  经典结果），而不是任何会重排到达顺序或跳过中间进位状态的捷径。
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
任何读 `op_log` 的代码（重放、`derive_final_cluster`、S0.8 对拍脚本）都必须先用
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
对象自己的 `op_log`/`op_log_len`**——cache 对象上的这两个 buffer 在下一次
`forward()` 开头被 `reset_parameters()` 清空是完全正常、预期内的行为，不需要
任何特殊豁免逻辑。这样选的理由：
- 和 `q`/`k_raw`/`k_roped`/`v` 已经在用的模式完全一致，不需要给 `op_log` 发明
  第二套"豁免 reset"的特殊生命周期规则；
- `ctx.save_for_backward` 是 PyTorch autograd 的标准机制，`backward()` 只依赖
  `ctx` 这一个自包含的输入，不依赖"cache 对象在 forward 和 backward 之间没有
  被其它代码修改过"这个更脆弱、更隐式的前提——比如梯度累积场景下，同一个
  cache 对象可能在这次 forward 的 `backward()` 被调用之前，就被下一次
  `forward()` 调用并 reset 过；
- 不需要引入"reset 时哪些字段该跳过"这种容易被后续修改者忘记维护的例外
  清单。

> **更正（这一轮修的）：`ctx.save_for_backward(cache.op_log, cache.op_log_len)`
> 存的是裸引用，不是快照，必须改成 `.detach().clone()`。** `ctx.save_for_
> backward` 保存的是张量对象（指向底层 storage），不会自动 deep copy——这一点
> 上面"不做 reset 豁免"的论证里已经预设了"存进 ctx 的是一份独立快照"，但没有
> 显式写出**怎么**让它独立。本项目一贯的风格是预分配 buffer、`reset_parameters()`
> 原地 `zero_()` 复用（§5.19 tip 3、§5.11 的 pad 槽清零讨论都是同一个模式），
> 不是每次重新分配——`cache.op_log`/`cache.op_log_len` 大概率也是这样实现的。
> 这意味着如果只是把 `cache.op_log` 这个张量对象本身存进 `ctx`（不克隆），
> 下一次 `forward()` 里 `reset_parameters()` 对同一块底层 storage 做原地
> 清零时，`ctx` 里"存"的那个引用指向的数据也会被一起清空——`ctx` 拿到的从来
> 不是一份独立快照，只是一个指针。**PyTorch 的 saved-tensor 版本计数器可能会
> 在这种情况下让 `backward()` 访问 `ctx.saved_tensors` 时报错**（原地修改过的
> 张量被检测到，抛 `RuntimeError`），但这依赖 `reset_parameters()` 具体怎么
> 实现、要不要触发这个检测机制，不是一个可以依赖的安全网——不能指望"如果算错
> 了它会崩溃"来代替"从一开始就不让它有机会算错"。**修法就是显式
> `.detach().clone()`**：`.detach()` 确保不会意外带上任何计算图（`op_log`/
> `op_log_len` 本来就是不需要梯度的整数簿记，这一步更多是防御性的显式声明），
> `.clone()` 真正复制底层数据，之后 `reset_parameters()` 无论怎么原地清零
> 复用原 buffer，都不会影响 `ctx` 里这份独立副本。**`q`/`k_raw`/`k_roped`/`v`
> 不需要同样处理**——它们是这个 Function 的输入参数，是调用方（模型的前向
> 计算）每次新创建的激活张量，不是 cache 对象持有、会被 `reset_parameters()`
> 复用清零的 buffer，不存在这个别名风险，不需要类比着也加克隆。
>
> 代价：一次 `(B,G,OP_max,4)` int32 克隆，32k 下约 16MB/层——相对这个 Function
> 本来就要处理的激活量级可以忽略，用一次可忽略的拷贝换掉一类"取决于
> `reset_parameters()` 具体实现细节、可能表现为静默错误也可能表现为运行时
> 崩溃"的别名 bug，是明确划算的，不是可选的性能优化项。

**`op_log`（完整 `(B,G,OP_max,4)`）和 `op_log_len`（`(B,G)`）一起进 `ctx`，
不做切片**：虽然不同 `(b,g)` 的有效长度不同，但保存前按最长有效长度裁剪成
ragged 张量既没必要也麻烦（batched 张量本来就不支持 ragged shape），直接连同
静态形状一起克隆、backward 时再用 `op_log_len` 截断读取，是最简单、和张量的
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
