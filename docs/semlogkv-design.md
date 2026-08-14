# SemLogKV 设计与实验协议

> 语义簇取代位置分桶，簇级抽象位置向量取代 post-RoPE 均值池化。
> 2026-08-14 · 分支 `claude/semantic-cluster-log-compression-8ai32a`
> 对照基线（CLAUDE.md §6.1）：vanilla ACC 0.6113 / LongBench 0.1716 / LongBench_e 0.1918 / niah 0.0827；
> 稠密上限（§6.3）niah 0.9353。

## 0. 选型结论

推荐方案：**语义簇路由（DP-means / 非参预算）＋ 簇内 Fenwick ladder ＋ 相位因子化的簇级位置
向量（z 统计量）**，带 `K_max` 硬顶。

三者不是三个可选项，而是互为前提的一件事：

- 语义簇一旦成立，成员位置必然散布，post-RoPE 均值会发生相位相消 —— 所以**必须**先把位置从
  内容里抽出来；
- 反过来，因子化 `E[R(p)k] ≈ E[R(p)]·E[k]` 丢掉的耦合项由簇内内容方差控制，**只有聚类质量
  足够好，因子化才成立**；
- 非参预算是让 needle 便宜的唯一机制（见 §3）。

任意拿掉一个，另外两个都塌。

前几轮讨论过的另一条路 —— 保持时间连续性、按 Ward 代价贪心合并相邻槽 —— 不是被否决，而是
**降级为簇内 ladder 的合并顺序策略**：它作用于「一个簇内部谁跟谁合并」，本方案作用于「谁和谁
同簇」，两者正交可叠加。先做后者，因为病灶在成员划分而不是合并顺序（6.5 的重要性池化已经
证明，槽成员固定时调权重救不了 niah）。

| 方案 | 捞针机制 | 需要 CPT | 可 eval-time 预验证 | 天花板 |
|---|---|---|---|---|
| 现状（位置分桶 + 二阶修正）| 无，needle 被稀释 1/w | 已完成 | — | niah 0.0827 |
| Ward 连续合并 | needle 作为局部离群点推迟合并 | 否 | 是 | 中 |
| **SemLogKV** | needle 自成簇，永不合并，键值均精确 | 是 | 部分（见 §6 Stage 2）| 高 |

## 1. 位置与内容解耦：z 统计量

### 1.1 为什么非做不可

现在每个槽存 post-RoPE 键的均值 `E[R(p_j)·k_j]`。Qwen3 是 full rotary，head_dim 128 全部
维度参与旋转，频率 `θ_f` 对应波长从约 2π token（高频）到百万 token（低频）。当槽成员位置
散布 `s` 时，所有满足 `θ_f·s ≫ 2π` 的频率上 `E[e^{iθ_f p}] → 0` —— 而 full rotary 意味着
**这些频率搭载的内容维度会被一起清零**。

所以对散布上千 token 的语义簇直接套用现有池化，键的大部分维度会归零，位置信息和内容检索
信号同归于尽。这不是精度损失，是表示的湮灭。位置必须单独表示。

### 1.2 定义

每个槽在原有 `(k̄, v̄, w, Σ, Γ)` 之外，维护一个逐频率的复数统计量 —— 成员位置分布在 RoPE
频率集上的**特征函数**：

```
z_f = ( Σ_j w_j · e^{i·θ_f·p_j} ) / ( Σ_j w_j )      f = 0 … F-1,  F = rope_n_elem / 2
```

实数形式存两个张量 `z_re[f] = E[cos θ_f p]`、`z_im[f] = E[sin θ_f p]`。相位 `arg z_f` 是
圆均值位置，模长 `ρ_f = |z_f|` 是该频率上的位置集中度。同时键改存 **pre-RoPE 内容均值**
`k̄_raw = E[k^raw]`。存储代价：每槽 `rope_n_elem` 个 fp32，槽数是 O(log n) 量级，不改变渐近。

### 1.3 三条性质

**嵌套精确性。** 单 token 槽 `ρ=1`、相位 `= θ_f p`，重物化后**严格等于标准 RoPE** —— 精确
token attention 是本表示的特例。连续 span 且内容恒定时，闭式解正好是现有 `log_kv_cache.py`
docstring 里那个 Dirichlet 因子 `sin(wθ/2)/(w·sin(θ/2))` —— 现有方案也是它的特例。新表示
连续地包含旧方案和精确注意力，这给了单测两个可以逐位核对的锚点。

**合并精确封闭。** `z_merged = (w₁z₁ + w₂z₂)/(w₁+w₂)`，加权复数平均，精确、可结合、**零累积
误差**。对比内容侧 rank-1 截断误差会随层级累积（D1 自检测到的 width≥8 失真），位置这条轴
现在是完全无损的。

**β 重物化旋钮。** 读出时按频率复乘回内容：

```
ρ_f    = |z_f|
gain_f = ρ_f^(β-1)
k_eff  = apply_rope( k̄_raw, cos_eff = z_re·gain, sin_eff = z_im·gain )
```

- `β = 1`：诚实衰减，等价于现有 expected-RoPE 推广到任意位置集合。`gain ≡ 1`，零额外开销。
  散布簇的内容被压掉 —— 即 §1.1 的问题。
- `β = 0`：满模长。**内容信号 100% 保留**，位置以「每频率一个抽象相位」参与打分，query 侧
  照常 RoPE，相对位置几何依然成立。这就是「抽象的簇级位置向量，效果类似 token RoPE」。

关键工程红利：`z` 是精确存储的，**β 只在读出物化时生效，因此是 eval-time 可扫参数**，不需要
重新摄入缓存，完全兼容现有 `DIAG_ARGS` 扫参文化。

### 1.4 诚实的代价：因子化误差

现有的 `E[R(p)k]` 是**精确的一阶矩**（线性性）。换成 `E[R(p)]·E[k]` 会丢掉位置-内容耦合项：

```
E[R(p)k] = E[R(p)]·E[k] + Cov( R(p), k )
                          └─ 被丢弃，量级 ≈ (位置散布) × (内容方差)
```

这条误差项由簇内内容方差控制，也就是说**聚类越纯，因子化越准**。这正是 §0 说的互为前提，
也是 Stage 0 必须实测的量（S0.4）。

### 1.5 实现上的意外红利

litgpt 的 `apply_rope`（`litgpt/model.py:2074`）用的是「前后半重复」布局：`x1 = x[..., :d/2]`、
`x2 = x[..., d/2:]`，`out = x·cos + cat(-x2, x1)·sin`。这意味着维度 `f` 与 `f+d/2` 组成一个
复数对，且 `cos[f] == cos[f+d/2]`。两个直接后果：

- **单 token 的 z 不需要任何三角函数** —— `z_re = cos_cache[..., :F]`、
  `z_im = sin_cache[..., :F]`，直接切模型已经算好的 RoPE cache。
- **物化不需要 atan2** —— `gain = ρ^(β-1)` 直接乘在 `z_re/z_im` 上就是 `cos_eff/sin_eff`，
  复制成前后半后喂给现成的 `apply_rope` 数学。β=1 时 `gain ≡ 1`，连 pow 都省了。

整个位置机制因此只有约 40 行纯张量代码，没有新的数值原语。

## 2. 数据结构与预算

### 2.1 三层结构

| 层 | 内容 | 位置表示 | 规模 |
|---|---|---|---|
| recent window | 精确 token，不压缩 | `ρ=1`，即标准 RoPE | recent_size（1024）|
| 簇池 | K 个语义簇，DP-means 路由 | 每簇每 entry 一个 z | K ≤ K_max |
| 簇内 ladder | 该簇成员的 Fenwick 层级 | 合并时 z 精确加权平均 | B′ × L_max per cluster |

**簇内 ladder 是承重墙不是装饰。** 如果只有一层平面簇，needle 一旦没被 τ 隔离、而是进了一个
几千成员的大簇，就被稀释成 1/几千 —— 比现在的 1/32 更糟。簇内 ladder 保证新近成员保持细粒度。

**与现状的一处语义变化**：level 0 存单 token（w=1）而不是像现在这样存「2 token 合并」。这样
每个簇的 ladder 是它自己成员序列上的标准 Fenwick。带来一个关键性质，见 §3 末尾。

**读出端不感知簇结构。** attention 只见一个扁平槽池 `(k_eff, v̄, w, stats)` 加一个有效位掩码，
`log_kv_slot_attention` 的接口和数学不变。簇只是写路径的路由元数据。这是保持张量矩形的关键
设计。

### 2.2 预算：非参簇数

路由规则就是 DP-means：与各簇 centroid 的余弦相似度取最大，`≥ τ` 则加入，否则开新簇；`K` 满
时按 Ward 代价合并最近的两个簇腾位。这恰好是 CRP 混合模型的小方差极限，于是簇数天然带一个
理论形状：

```
K_n = 1 + Σ_{i=1}^{n-1} Bern( α / (α+i) )      E[K_n] ≈ α·ln n
```

因为是独立 Bernoulli 之和，Chernoff 直接适用，超出常数倍期望的概率是 n 的多项式小量 —— 所以
是**高概率界**而不只是期望好看。最坏 `K = n`，且在确定性版本里这只在「每个 token 都远离所有
已有簇」时发生，即数据真的没有可压缩结构。

> **必须提前接受的经验事实**：自然语言的类型数按 **Heaps 定律** `~ n^β`（β≈0.4–0.6）增长，
> 不是 log n。对应的是 Pitman-Yor 而非 CRP。所以真实的 `K_eff` 大概率是幂律。
>
> 这不是坏消息：`O(√n)` 依然是合格的次线性目标（32k 下约 181 簇，仍远少于现在的约 2560 槽）。
> 论文里写 `O(n^d)` 带可调 d 比硬说 log n 更站得住。但这必须在 Stage 0 实测（S0.2），
> 不能假设。

### 2.3 退化锚点

| 配置 | 等价于 | 用途 |
|---|---|---|
| `K_max=1, β=1` | ≈ 今天的 LogKV（单簇 = 位置序 ladder，诚实衰减）| **正确性闸门**：应复现 0.1716 / 0.0827 |
| `K_max=1, β=0` | 单簇但去掉位置衰减 | 单独隔离 β 的作用 |
| `K_max=16, β=0` | 完整设计 | 主实验 |
| `τ → 1` | 每 token 一簇（受 K_max 截断）| 接近稠密的上界探测 |

## 3. 为什么能解决捞针

逐项对比三种情形下 needle 的命运。

**稠密注意力**

```
score = scale·(q · k_needle)      value = v_needle
```

needle 靠 `q·k_needle` 显著高于 haystack 而胜出，读出的是它自己的 value。

**现有池化（needle 落在宽度 w 的槽里）**

```
k_slot   = ( k_needle + Σ_{j≠n} k_j ) / w
q·k_slot = (q·k_needle)/w + (w-1)/w · (q·k̄_hay)
score    = scale · 上式 + λ·log w
v_slot   = ( v_needle + Σ_{j≠n} v_j ) / w
```

两侧同时被稀释，这是关键。分数侧 needle 的判别信号被衰减 `w` 倍；`+λ log w` 补的是**计数
质量**而非内容，补不回稀释。更致命的是**读出侧**：即使这个槽赢得了注意力，返回的 `v_slot`
也已经是 haystack 均值 —— 而 Γ 修正是 rank-1，救不回一个被 w 倍稀释的方向。这解释了为什么
二阶修正把 niah 从 0.032 提到 0.0827（2.6×）却依然离 0.9353 极远：它修的是分数，修不了读出。

**SemLogKV**

```
needle 与所有 haystack centroid 余弦 < τ  ⇒  自成一簇
该簇成员数 = 1  ⇒  ladder 永不填满  ⇒  entry 永不合并

score = scale·(q · k_needle) + λ·log 1 = scale·(q · k_needle)   ← 与稠密逐位相同
value = v_needle                                                 ← 与稠密逐位相同
z: ρ=1, 相位 = θ_f·p_needle                                      ← 位置也精确
```

**分数、读出、位置三者同时恢复到稠密。** 这是「needle 自成簇」这个结构带来的，不是靠调权重
或加修正项凑出来的。§2.1 那处 level-0 语义变化（存单 token 而非 token 对）在这里兑现：小簇的
ladder 永远填不满，needle 的 entry 因此在整个序列生命周期里保持精确。

**代价只有一个槽。** haystack（重复的散文）语义高度同质、`K_eff` 小、压得很狠，省下的预算
转移给异常点。这就是为什么**自适应预算不是锦上添花而是唯一机制**：固定均匀预算在信息论上
无法同时做到「压缩 haystack」和「保留 needle」，因为它不允许预算转移。

**失败模式（必须在 Stage 0 证伪）**：τ 太松 → needle 并入 haystack 簇 → 退回稀释。τ 是全方案
最敏感的超参，S0.3 就是为它设计的。

## 4. 最坏情况的双重兜底

### 4.1 表现的地板：不会比现状更差

内容不可压缩时 `K` 增长 → 槽变多 → 质量向稠密平滑逼近。而当 `K` 撞上 `K_max`，强制 Ward 合并
把结构推回「少数大簇 + 位置序 ladder」，即 §2.3 的 `K_max=1` 锚点，也就是**今天的 LogKV
行为**。所以退化的终点是现状，不是比现状更差。

### 4.2 性能的天花板：硬顶而非期望

serving 不能接受期望有界但尾部无界。所以 v1 用**硬性 `S_max`**：预分配 `K_max × B′ × L_max`
个槽，占用随内容浮动但上界固定。理论上那个「不设上限、期望 O(log n)、最坏 O(N)」的优美版本
保留为论文里的曲线和 ablation，不进生产路径。

| 配置 | 槽数 | 说明 |
|---|---:|---|
| 现状 recent + B=512 × ~5 级 | ≈ 3584 | 对照基线 |
| SemLogKV recent 1024 + K16×B′8×L13 | ≈ 2688 | 预分配上界，典型占用更低 |

> **论文层面最大的攻击面**：自适应预算意味着 niah 变好可能仅仅因为多用了内存 —— 而 niah 恰恰
> 最容易把 K 推高。**从第一个实验开始就必须报告每个 benchmark 上的实际占用槽数**，并准备
> memory-matched 对照（把 vanilla 的 B 调大到相同槽数再比）。这一条如果留到最后补，整套实验
> 要重跑。

## 5. 实现方案

### 5.1 新模块 `litgpt/log_kv_position.py`

四个纯函数，无状态、无可学参数、CPU 可测。先写这个模块和它的单测，其余改动全部依赖它。

```python
# 单 token 的 z：直接切现成 RoPE cache，零三角函数（见 §1.5）
def position_stat_from_rope(cos, sin, rope_n_elem):
    """cos/sin: (..., T, rope_n_elem)，前后两半重复
    returns z_re, z_im: (..., T, F) fp32,  F = rope_n_elem // 2"""
    F = rope_n_elem // 2
    return cos[..., :F].float(), sin[..., :F].float()


# 加权复数平均：精确、可结合、零累积误差（性质 2）
def merge_position_stat(z_re1, z_im1, w1, z_re2, z_im2, w2):
    tot = (w1 + w2).clamp_min(EPS)
    a, b = (w1 / tot).unsqueeze(-1), (w2 / tot).unsqueeze(-1)
    return a * z_re1 + b * z_re2, a * z_im1 + b * z_im2


# z -> (cos_eff, sin_eff)，可直接喂给 apply_rope 的数学（性质 3）
def materialize_rope_factors(z_re, z_im, beta, rho_min=1e-3):
    rho  = torch.sqrt(z_re * z_re + z_im * z_im).clamp_min(rho_min)
    gain = rho.pow(beta - 1.0) if beta != 1.0 else None
    ch, sh = (z_re, z_im) if gain is None else (z_re * gain, z_im * gain)
    return torch.cat([ch, ch], -1), torch.cat([sh, sh], -1)


# 槽键物化：镜像 model.apply_rope 的数学，去掉 dim==3 约束
def materialize_slot_key(k_raw, z_re, z_im, beta, rope_n_elem):
    cos_eff, sin_eff = materialize_rope_factors(z_re, z_im, beta)
    x   = k_raw[..., :rope_n_elem]      # partial rotary: 尾部内容通道不旋转
    h   = rope_n_elem // 2
    rot = torch.cat((-x[..., h:], x[..., :h]), -1)
    return torch.cat([x * cos_eff + rot * sin_eff, k_raw[..., rope_n_elem:]], -1)
```

该模块的单测（Stage 0.1，纯 CPU）：

- **嵌套精确性**：单 token + β 任意 → 与 `model.apply_rope` 输出逐位一致（`ρ=1` 时 gain≡1）。
- **Dirichlet 闭式解**：连续 w-token span 的 `ρ_f` 应等于 `|sin(wθ_f/2) / (w·sin(θ_f/2))|`，
  相位等于 `θ_f × span 中心`。这条同时验证了新表示确实包含现有 docstring 的数学。
- **合并精确性与结合律**：任意二叉合并顺序结果一致；与直接对全体成员求平均一致。
- **数值边界**：全相消（`ρ→0`）时 β=0 不产生 NaN/Inf，gain 被 `rho_min` 钳住。

### 5.2 `litgpt/log_kv_cache.py`

- 新增 buffer：每槽 `z_re/z_im`（`(B,G,S,F)` fp32）、每槽 `cluster_id`、每簇 `centroid`
  （`(B,G,K_max,d)`，L2 归一）与 `cluster_size`。
- 键改存 **pre-RoPE**。recent window 同样存 pre-RoPE + 其 z，读出时统一走
  `materialize_slot_key` —— **一条代码路径**，保证嵌套性质在生产里被真实执行而不只是单测里。
  代价是每次读出对窗口做一次逐元素旋转，与已有的 `O(S·d)` matmul 同量级，可忽略。
- `compact()` / `_binary_carry()` 增加 z 的合并（一行 `merge_position_stat`），Σ/Γ 全部改在
  **pre-RoPE 内容空间**统计，物化时对 `σu`、`γa` 施加与 `k̄` 相同的旋转（rank-1 在线性映射下
  仍是 rank-1）。**顺带的好处**：Σ 不再混入位置相位方差，簇内同质时 rank-1 近似精度应系统性
  恢复 —— 这是 D1 自检 width≥8 失真的根因之一。
- 新增路由函数（`@torch.no_grad()`，与 cache 更新同路径，训推自动一致）：

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

- `get_attention_state()` 额外返回 `(B,G,S)` 有效位掩码；`log_kv_slot_attention` 增加一个
  可选的槽有效性掩码参数（在 fp32 分数上填 `-inf`），与现有 `causal_tail` 正交。

### 5.3 `litgpt/model.py`

唯一的接口性改动：cache 现在需要 **pre-RoPE 的 k** 和该 token 的 **cos/sin 切片**，而不是
post-RoPE 的 k。模型本来就同时持有这两样（RoPE 就在这里施加），所以是传参改动而非新计算。
`build_log_kv_cache` / `set_log_kv_cache` / `enable_log_kv_training` 三处按 `importance_pooling`
的先例透传新参数。

### 5.4 矩形张量与 batching

不同 (batch, kv-head) 的簇分配不同，这是 landscape 里点名的第 3 类坑。v1 的处理方式：

- **按 (B, G) 独立聚类** —— 不同头关注不同语义，共享簇是错的。metadata 代价 `K_max × d`
  per head，可接受。
- **预分配 + 掩码**：槽张量固定为 `K_max × L_max × B′`，占用不满时掩码屏蔽。张量始终矩形，
  不引入 ragged 布局。
- ragged / paged 布局留到工程化阶段，不在研究验证期做。

### 5.5 参数与开关

沿用项目惯例：`demo.py` / `eval.py` 走 `run_cli()` 签名自省与 `_o()`，YAML 新字段**默认必须写
`null`**（CLAUDE.md §3.3 的坑），不进 `eval.sh`/`majob.sh` 核心列表，走 `DIAG_ARGS` opt-in。

| 参数 | 默认 | 作用 |
|---|---|---|
| `log_kv_semantic_clusters` | `false` | 总开关，关闭时逐字节复现改动前行为 |
| `log_kv_cluster_k_max` | `16` | 簇数硬顶 = 性能天花板 |
| `log_kv_cluster_tau` | `0.6` | DP-means 阈值，最敏感的超参 |
| `log_kv_position_beta` | `0.0` | 重物化指数，eval-time 可扫 |
| `log_kv_cluster_entries` | `8` | 簇内每级 entry 数 B′ |

## 6. 实验协议

四个阶段，每阶段带一个可证伪的决策门。顺序不能打乱：Stage 0 几乎不花 GPU，却能在写任何生产
代码之前否掉整个方案。

### Stage 0 — 离线可证伪（1 次 GPU dump + 全部 CPU 分析）

只需要一次前向：对几条 32k 的 NIAH prompt dump 每层每头的 **pre-RoPE k** 与 needle 的 token
span（后者直接复用 `log_kv_pin_diag.py` 已有的定位逻辑）。之后所有分析在 CPU 上做。

| 编号 | 测什么 | 为什么 |
|---|---|---|
| S0.1 | §5.1 四条性质单测 | 数学正确性，纯 CPU，不需要 dump |
| S0.2 | `K_eff(n)` 随 τ 的曲线，拟合 log / 幂律 / 线性 | 预算故事成不成立 |
| S0.3 | needle 隔离率：所在簇的 `w` 分布 | §3 的核心机制成不成立 |
| S0.4 | 因子化误差 `‖E[Rk]−E[R]E[k]‖ / ‖E[Rk]‖` | §1.4 的代价有多大 |
| S0.5 | 湮灭度量 `‖E[R(p)k]‖ / ‖E[k]‖` 随位置散布 | 量化 β=1 有多致命，为 β=0 提供证据 |

**Stage 0 决策门**

- **S0.2**：32k 下 τ∈[0.5, 0.8] 时 `K_eff` 应在 10² 量级。若达到 10³ 以上，预算故事垮掉 ——
  回去把聚类粒度调粗（更低的 τ，或先做维度约简）再谈。
- **S0.3**：needle 落在 `w ≤ 4` 的簇里的比例应显著高于随机基线。若 needle 大多并入大簇，
  §3 的机制不成立，方案应当就地停止而不是硬着头皮训。
- **S0.4**：真实簇上的相对误差中位数应 < 0.2。若普遍很大，说明簇不够纯，因子化前提不成立。

### Stage 1 — 实现

按 §5 落地，默认关闭时逐字节复现现状（照 `importance_pooling` 的先例，这条本身就是一条
单测）。额外必测：`K_max=1, β=1` 的退化路径在数值上应逼近现有位置分桶实现。

### Stage 2 — eval-time 探测

诚实的预期：新表示对现有 CPT 权重是分布外的，绝对分数可能不好看。但这一阶段的价值在于
**形状而非绝对值**，且 Config A 是一个硬性正确性闸门。

| 配置 | 参数 | 期望 |
|---|---|---|
| A 正确性闸门 | `K=1, β=1` | ≈ 0.1716 / 0.0827。**对不上就是实现有 bug，不要往下走** |
| B β 隔离 | `K=1, β=0` | 单独看去掉位置衰减的影响 |
| C 主实验 | `K=16, β=0`，扫 τ | niah 应显著高于 A |

```bash
DIAG_ARGS="--log_kv_semantic_clusters true --log_kv_cluster_k_max 16 \
  --log_kv_position_beta 0.0 --log_kv_cluster_tau 0.6" \
    bash majob.sh exp/qwen1.7b-32k/arc_warmup.yaml
```

注意 CLAUDE.md §8.5 的两个既有坑：主 benchmark 参数不要传 `none`（LongBench 来自主列表），
以及多配置串行跑（输出目录不按超参分开，靠 JSON 内的 metadata 区分）。新参数需要一并写进
eval 的 metadata 字段。

### Stage 3 — CPT

Stage 2 有信号后再投入。沿用 `arc_warmup_importance_continue.yaml` 的安全模板（新 `save_path`、
`resume_dir` 只读权重、`max_steps` 是续训步数）先跑 300 步短续训；确认方向后再从 base 全量
1500 步。

**β warmup**：从 `β=1`（≈ 模型熟悉的现有分布）线性退火到 `β=0`，warmup 100 步左右 —— 直接
复用 `second_order_scale` 那套已经验证过的爬坡机制和它的教训（目标值别设太激进、warmup 别只
给 10 步，见 CLAUDE.md §4 的崩溃排查）。

## 7. 消融表

每一行回答一个独立问题，不是穷举网格。所有行都必须附带实际占用槽数（§4.2）。

| 旋钮 | 取值 | 回答的问题 |
|---|---|---|
| `K_max` | 1 / 4 / 16 / 64 | 语义分组本身值多少分？1 是现状锚点 |
| `β` | 0 / 0.25 / 0.5 / 1 | 位置衰减 vs 内容保真的最优点在哪 |
| `τ` | 0.4 – 0.85 | needle 隔离与簇纯度的平衡点 |
| `λ`（mass bias）| 0 / 1 | 大簇的计数质量补偿是否仍然正确 |
| rank-1 Σ/Γ | 开 / 关 | 解耦后二阶修正是否还有边际价值 |
| vanilla memory-matched | B 调大到同槽数 | **排除「只是多用了内存」** |

指标沿用现有四项（ACC / LongBench / LongBench_e / niah@32768），**另加 multi-needle** ——
单 needle 一旦从 0.08 提上去就会迅速失去区分度，而 multi-needle 才是这类分层压缩方法真正
拉开差距、也最被审稿人看重的实验。

## 8. 风险与对策

| 风险 | 后果 | 对策 |
|---|---|---|
| **τ 敏感** | needle 并入大簇，机制失效 | S0.3 提前证伪；τ 进主扫描；v2 可做 τ 反馈控制器（带迟滞防振荡）|
| **β=0 在「撒谎」** | 散布簇声称自己在某个确定相位，可能扰乱信息排序 | β 是连续旋钮；v1.5 可把 `log ρ̄` 作为独立的位置置信项加进分数，让内容不衰减但模型知道位置不可靠 |
| **centroid drift** | 早期成员的归属基于后来会漂移的 centroid | 在线聚类固有近似，接受并记录；周期性重聚类是 v2 选项（摊还 O(S)）|
| **内存不对齐比较** | 论文被质疑「只是多用了内存」| 见 §4.2，从第一个实验起报告槽数 + memory-matched 对照 |
| **存储增加约 40%** | z 是每槽 `rope_n_elem` 个 fp32 | 不改变渐近（槽数仍 O(log n)）；先 fp32 保精度，实测后再评估 bf16 |
| **训练期显存随内容波动** | batch 内簇占用不同 | `K_max` 预分配上界固定，训练期必须开硬顶 |

### 与既有教训的对照

pin 系列的四条死因逐条检查，本方案在设计上都规避了：

- **训推不一致** —— 路由是 k 的确定性函数、跑在同一条 `no_grad` cache 更新路径；
- **分数尺度失配** —— 槽种群完全同质，不存在精确槽混入池化槽；
- **观察窗口假设** —— 路由在 flush 时逐 token 发生，不需要一次性看到整段 prompt，也不依赖
  「prompt 尾部是问题」；
- **剂量效应** —— 不新增槽种类，只改变成员划分。

6.5 的重要性池化建议**保持关闭**：它在 niah 上定向变差的推测原因是高范数 attention-sink
token 挤压了 needle 的份额，而在本方案里 sink 会因为方向离群而自成一簇，被结构性隔离 ——
这正是需要单独验证的一个有趣副作用，但不应与主实验混在一起。

## 9. 相关工作定位

| 工作 | 做了什么 | 与本方案的区别 |
|---|---|---|
| SemantiCache (2603.14303) | 按分隔符切语义块 + 贪心种子聚类合并 | 固定 budget，无簇内分辨率层级，frozen model |
| Multipole Attention (NeurIPS 2025) | key 做 k-means，远处用 centroid 近似打分，越远越粗 | **保留全量 KV（O(n) 显存）**，centroid 只是索引，不是压缩 |
| SeKV (2606.31145) | entropy 引导语义 span，GPU summary + CPU SVD 按需重建 | offload + 检索系统，什么都不丢，非次线性 |
| ClusterAttn (ACL 2025) | 密度聚类自适应簇数 | 稀疏注意力，固定 1024 token 预算 |
| GVote (ICLR 2026) | 蒙特卡洛采样未来 query 免手工设预算 | eviction/selection 框架，预算仍是 O(1) 固定池 |

空着的生态位：**非参先验决定簇数 + 严格次线性空间 + 簇内分辨率层级 + 特征函数式抽象位置**。
CRP 的集中不等式给了这条线一个别人写不出的理论节。
