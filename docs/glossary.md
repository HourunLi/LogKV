# SemanticLogKV 术语表

> 速查表。设计结论看 `CLAUDE.md`，实现细节看 `algorithm-spec.md`。

Qwen3-1.7B 参考：`n_layer=28`、`n_head=16`、`n_query_groups=8`、`head_dim=128`、
`rotary_percentage=1.0`。

## T1. 结构

| 术语 | 含义 |
|---|---|
| cache | 每个 transformer block 一个 `LogStructuredKVCache` |
| recent window | 最近 token 精确保留，不压缩 |
| flush | token 滑出 recent window，进入压缩结构 |
| cluster | 语义身份，一个 centroid，只用于路由 |
| segment | cluster 内一段连续访问，不分配独立 ladder |
| ladder | 每 cluster 一条多层 entry 结构 |
| entry | 存储单位：`k_raw/v/w/anchors/Σ/Γ` |
| virtual slot | entry 按 anchor 展开后的 attention 单位 |
| block | Fenwick-style carry 的一组 entry |

## T2. 维度和规模

| 符号 | 含义 |
|---|---|
| `B` | 重载：shape 里是 batch size；旧 LogKV 参数里是每层槽数 |
| `B′` | SemanticLogKV 每层 entry 数，Stage 0 默认 128 |
| `G` | KV group 数，Qwen3-1.7B 为 8 |
| `nh` | query head 数，Qwen3-1.7B 为 16 |
| `rf` | `nh / G`，每个 KV group 共享给几个 query head |
| `d` | head dim |
| `N` | `max_seq_length` |
| `K` | 当前实际 cluster 数 |
| `K_max` | cluster 数硬上界 |
| `K_eff` | unclipped 路由下内容需要的簇数，离线分析口径 |
| `L_alloc` | 每簇 ladder 层数，由 `N/K_max/B′` 推导 |
| `w` | entry 覆盖的真实 token 数 |
| `M` | entry 去重后的 anchor 数，1-3 |

## T3. 参数

| 符号 | 参数名 | 含义 |
|---|---|---|
| `λ` | `log_kv_lambda` | 原始 mass bias 系数，只控制 `+λ·log(w)` |
| `λ_new` | - | 新簇阈值，`λ_rel · s_h` |
| `λ_rel` | `log_kv_cluster_lambda_rel` | 新簇阈值系数 |
| `s_h` | - | 每 `(layer, KV group)` 的离线 key 尺度 |
| `η` | `log_kv_seg_eta` | join cost 时序权重 |
| `g0` | `log_kv_seg_g0` | 时序项饱和尺度 |
| `g_max` | `log_kv_seg_gap_max` | 开新 segment 的间隔阈值 |
| `ℓ_block` | `log_kv_seg_block_level` | 段边界保护深度 |
| `γ` | `log_kv_seg_forget` | 跨段 centroid 遗忘 |

## T4. 数学量

| 符号 | 含义 |
|---|---|
| `k_raw` | qk-norm 后、RoPE 前的 key |
| `v` | value |
| `μ_c` | cluster centroid |
| `p_lo/p_hi` | entry 最早/最晚成员位置 |
| `sum_wp` | `Σ w_j·p_j`，int64 |
| `p_mid` | `clamp((2·sum_wp + w)//(2·w), p_lo, p_hi)` |
| `Σ` | key 协方差 rank-1 近似，分数侧二阶修正 |
| `Γ` | key-value 互协方差 rank-1 近似，读出侧修正 |
| Ward cost | `(n_a n_b)/(n_a+n_b) · ||μ_a-μ_b||²` |

## T5. 代码符号

| 名字 | 含义 |
|---|---|
| `compact()` | 按时间序相邻配对合并 entry |
| `_binary_carry()` | 旧 LogKV 的计数进位控制流；语义路径要重写驱动 |
| `log_kv_slot_attention()` | 槽级 attention；语义路径新增 `slot_valid/M_s` |
| `LogKVStreamTrainingAttention` | 自定义 autograd；语义路径 backward 必须回放 `op_log` |
| `CacheAttentionState` | 计划中的具名返回结构，替代裸 tuple 解包 |
| `level0_phase` | segment padding 的独立相位计数器 |
| `op_log` | 训练 forward 记录的路由/结构操作日志，供 backward 重放 |

## T6. 实验

| 名字 | 含义 |
|---|---|
| Stage 0 | 离线证伪，先于生产实现 |
| S0.0 | 语义分组 vs 连续分段 |
| S0.2 | `K_eff(n)` 曲线 |
| S0.3 | needle exact-entry 隔离 |
| S0.4 | key/value 方差 |
| S0.5 | entry 跨度 |
| S0.6 | anchor 去重和 fixed-3 成本 |
| S0.7 | supersession/value 作废 |
| S0.8 | 批量近似 vs 严格串行 |
| memory-matched | 调整 baseline 内存后再比，排除“多用内存” |
