# SemanticLogKV 开发入口

> 精简版：只保留当前结论、实现边界、下一步和必要的工程约束。旧版本推导、更正记录和
> 反复提醒已移除；需要恢复历史可看 git。

## 文档地图

| 文件 | 内容 |
|---|---|
| `CLAUDE.md` | 当前状态、核心设计、为什么值得做、下一步 |
| [`docs/algorithm-spec.md`](docs/algorithm-spec.md) | 算法规格、实现顺序、复用边界、易错点 |
| [`docs/experiments.md`](docs/experiments.md) | Stage 0-3 实验协议和消融表 |
| [`docs/risks-and-open-questions.md`](docs/risks-and-open-questions.md) | 风险、技术难点、未决问题 |
| [`docs/position.md`](docs/position.md) | RoPE/position 专题 |
| [`docs/glossary.md`](docs/glossary.md) | 术语速查 |

建议阅读顺序：先读本文件，再读 `algorithm-spec.md` 的 §5.18-§5.21 和
`risks-and-open-questions.md` 的 §11。只查符号时看 `glossary.md`。

## 0. 现状速览（2026-08-25）

**问题**：现有 LogKV 按位置做 Fenwick 式 2:1 压缩，宽槽会把语义无关 token 均值到一起。
在 32k NIAH 上，稠密上限约 0.9353，现有 LogKV（二阶、无 pin）约 0.0827，主要损失来自
needle 被宽槽稀释。

**本分支方案**：把“按位置分桶”改成“按 semantic key 分簇”。语义相近 token 共享槽；
稀有事实倾向于自成小簇，并在簇内 ladder 的低层保持精确。位置不再平均 post-RoPE key，
而是存 pre-RoPE 内容均值和真实锚点 `(p_lo, p_mid, p_hi)`，读出时在锚点位置调用标准
`apply_rope`。

**已实现**：Stage 0 的机制 A（pre-RoPE `k_raw`/`v` dump）和 CPU 分析工具链，覆盖
S0.0、S0.2 口径①、S0.3、S0.4、S0.5、S0.6。核心文件：

- `litgpt/semantic_s0.py`
- `unused/semantic_stage0_dump.py`
- `unused/semantic_s0_sweep.py`
- `unused/semantic_s0_needle_isolation.py`
- `unused/semantic_s0_anchor_dedup.py`
- `unused/semantic_s0_analyze.py`

**未实现**：生产路径里的 `log_kv_semantic_clusters`、多簇路由、`CacheAttentionState`、
`op_log` 训练重放、机制 B（post-RoPE attention mass）、S0.7、S0.8 所需的
`cache_serial/cache_batch` 参考 cache。

**当前数据读法**：

- S0.0 支持继续做语义聚类，收益不是单纯来自更好的连续分段。
- S0.3 证明 needle 隔离有真实信号；新主口径是
  `needle_token_exact_entry_rate`，旧“小簇成员数 ≤ B′”只作 proxy。
- S0.6 说明 `lambda_rel=0.875` 召回更高但成本更贵；默认主线仍先看
  `lambda_rel=1.0`。
- `K_max` 的 binding-only 口径给出的是 no-Ward 上界，不是生产默认。32k 下若强行把
  binding 压到 <10%，`lambda_rel=1.0` 需要 `K_max≈280`（`c≈19`）；但
  `K_max=64/128` 的 exact-entry 已接近 unclipped 平台。下一步优先试
  `c≈6-8` 对应的预算，并用 S0.6 成本裁掉过贵配置。
- Stage-0 默认 `B′=128` 是分析口径。生产前必须把 `K_max:B′` 成对定尺；不要复用旧
  `B′=8` 的内存表。

**下一步**：

1. 读完 `stage0_dump/s0_3_kmax_ward.csv` 和 `s0_6_kmax_ward_*.csv`，定下
   `lambda_rel`、`K_max:B′`、`g_max`、`ℓ_block` 的第一组生产候选。
2. 补 S0.2 口径②、S0.7、S0.8 的两份 CPU 参考 cache。
3. 只有 Stage 0 gate 继续成立，才进入生产实现：position 模块、CacheAttentionState、
   多簇路由、segment padding、`op_log` 训练重放。

## 1. 背景

现有 LogKV 的核心瓶颈不是“均值怎么算”，而是“谁和谁被放进同一个槽”。位置分桶会把
needle 和 haystack 按时间邻近关系硬合并；当槽宽为 `w` 时，needle 的 key/value 贡献都
约被稀释到 `1/w`。二阶修正能补一部分分数侧信息，但 value 读出仍然是均值，救不回孤立
事实。

重要性加权池化已经试过，NIAH 定向变差，说明只改池化权重不够。SemanticLogKV 要改的是
成员划分：让高冗余 haystack 被压缩，让稀有事实用小簇保留。

## 2. 核心设计

### 2.1 cluster / segment / ladder

| 层 | 作用 |
|---|---|
| recent window | 最近 token 精确保留 |
| cluster | 语义身份，一个 centroid，只用于路由 |
| segment | 同 cluster 内一段连续访问，只约束低层合并 |
| ladder | 每个 cluster 一条 Fenwick-style entry 层级，真正存储 KV |
| entry | `k_raw_mean`、`v_mean`、`w`、锚点、可选 Σ/Γ |

路由拆成两件事：

- 内容新不新：决定是否开新 cluster。
- 时序是否打断：决定是否在同 cluster 下开新 segment。

这避免了 v3 的预算问题：高频复现实体不会因为多次出现而 fork 出一堆语义相同的 cluster。

### 2.2 位置表示

entry 存 pre-RoPE 内容均值和整数锚点：

```text
p_lo    = 最早真实位置
p_hi    = 最晚真实位置
sum_wp  = Σ w_j · p_j
p_mid   = clamp((2·sum_wp + w) // (2·w), p_lo, p_hi)
```

读出时去重 `[p_lo, p_mid, p_hi]`，每个锚点各调用一次标准 `apply_rope(k_raw_mean, p)`。
这样不平均相位，`ρ`、`β`、`κ` 这套旧机制都不再需要。

### 2.3 attention 接入

一个 entry 展开成 `M∈{1,2,3}` 个 virtual slots：

```text
score = scale · (q · apply_rope(k_raw, p_anchor)) + λ · log(w) - log(M)
value = v_mean
```

`-log(M)` 是正确性项，必须无条件生效，不受 `λ` 门控；它只抵消“一个 entry 被展开成多个
slot”带来的候选质量膨胀。

### 2.4 已知边界

均值池化不能表达 supersession（后文作废前文，例如“100 分，更正为 90 分”）。当前 v1
不引入 gated delta state；S0.7 先测 value 并存/作废比例，再决定是否把 Γ 改成
delta-rule 构造。

### 2.5 根本假设

这里的“语义”实际是 qk-norm 后、RoPE 前的 key 空间。它不保证等于人类语义；有些层/组
可能只是位置头或句法头。所以所有 Stage 0 统计必须按层报告：

- k/v/聚类/路由统计：按 `(layer, KV group)`。
- 依赖 query 的注意力统计：按 `(layer, query head)`。

## 3. 为什么能改善 NIAH

若 needle 成功成为单 token entry：

```text
w = 1, M = 1
score = scale · (q · k_needle)
value = v_needle
position = p_needle 上的标准 RoPE
```

分数、读出和位置都退回稠密行为。失败模式也清楚：

- `λ_rel` 太大：needle 并入 haystack 簇。
- `K_max` 太小：Ward 在预算绑定时合并掉 needle 小簇。
- final ladder 里 needle 不再是单真实成员 entry。

所以 S0.3 的主 gate 必须看 final exact-entry，而不是只看簇大小。

## 4. 预算和兜底

生产实现必须同时报告三笔账：

- **持久 cache entry 数**：用于 memory-matched 对比。
- **读出物理 slot 数**：三锚点固定展开时约为 `entry_count × 3`，这是真实 matmul 宽度。
- **训练 `op_log` 元数据**：只在训练路径分配，约按 `B × G × OP_max × 4 int32` 计。

旧文档里基于 `B′=8` 的 32k 预算表已经不再是当前默认，删掉。生产前按最终
`K_max:B′:L_alloc` 重算；如果只是为了压低 binding 把 `K_max` 抬到几百，次线性压缩的
卖点会被内存账抵消。

最坏情况下内容不可压缩，`K_max` 绑定触发 Ward 合并，结构退化成“少数大簇 + 位置序
ladder”。这是有界内存的质量降级，不是 OOM。

## 9. 相关工作定位

| 工作 | 区别 |
|---|---|
| SemantiCache | 先分段再聚类；本方案先聚类再分段，并在簇内保留层级分辨率 |
| Multipole Attention | centroid 用于索引，仍保留 O(n) KV |
| SeKV | offload/retrieval，不是严格次线性显存 |
| ClusterAttn | 稀疏注意力，固定 token 预算 |
| CompressKV | eviction，不是压缩；NIAH 数字很强，必须用 multi-needle、多轮、非尾部 query 区分 |
| ChunkKV | 支持“连续块/分段边界有价值”这个方向 |
| DeltaKV / Gated DeltaNet | 提供相似 token 距离和 delta-rule 参考，但 v1 不直接采用 |
| ddCRP | join cost 的理论参照 |

单针 NIAH 不足以证明本方案价值。关键实验必须包括 multi-needle、多轮，以及 query 不在
prompt 尾部的设定。

## 10. 工程习惯

- 新增会被 CLI 扫参覆盖的 YAML 字段写 `null`，否则 `_o()` 会让 YAML 非 null 值覆盖 CLI。
- `majob.sh` 遇到已有 `save_path` checkpoint 会跳过训练；新实验用独立目录。
- 本地 Python 环境用 conda env `mineru`；系统 Python 不适合跑仓库测试。
- 任何“生产已实现”的判断以 `litgpt/log_kv_cache.py`、`litgpt/model.py` 实际代码为准；
  当前 SemanticLogKV 生产路径尚未落地。
