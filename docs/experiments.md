# SemanticLogKV 实验协议与消融表

> 精简版。Stage 0 先证伪，Stage 1 才写生产路径。所有实验都要按层报告；k/v/路由按
> `(layer, KV group)`，依赖 query 的注意力统计按 `(layer, query head)`。

## 6. 实验协议

### Stage 0 - 离线证伪

一次 GPU dump + CPU 分析。当前已完成机制 A：

- dump qk-norm 后、RoPE 前的 `k_raw`
- dump `v`
- batch 固定为 1
- `G` 维保留
- `save_dtype` 默认 fp32
- needle span 使用和 eval 同一套 tokenizer offset 逻辑

机制 B 尚未完成：post-RoPE q/k 的分块 attention mass，不落盘完整 `(T,T)` score，只落盘
`attn_mass_by_dist` 直方图。

当前支持的分析：

| 编号 | 内容 | 状态 |
|---|---|---|
| S0.0 | 语义分组 vs 连续分段，扫 `(g_max, ℓ_block)` | 已有工具 |
| S0.2-① | 纯 DP-means `K_eff(n)` 曲线 | 已有工具 |
| S0.3 | needle final exact-entry 隔离率 | 已有工具，含 `K_max/Ward` clipped 探针 |
| S0.4 | 簇内 key/value 方差 | 已有工具 |
| S0.5 | entry 跨度分布 | 已有工具 |
| S0.6 | 锚点去重和 fixed-3 成本 | 已有工具，含 clipped 探针 |
| S0.7 | supersession/value 作废比例 | 未写 |
| S0.8 | 批量近似 vs 严格串行分歧率 | 缺 `cache_serial/cache_batch` |

### Stage 0 当前结论

- S0.0：语义聚类有边际价值，不直接退回纯分段。
- S0.3：needle 隔离有真实信号；主指标改为 `needle_token_exact_entry_rate`。
- S0.6：不能按最低 `E[M]` 选配置，要同时看 entry 数、fixed-3 slot 数、Ward 合并和
  binding。
- `lambda_rel=1.0` 是默认主线；`0.875` 是更贵的高召回候选。
- `K_max` 不按 binding-only 上界定默认。先裁决 `c≈6-8` 对应预算是否够好。

### Stage 0 常用命令

S0.3 clipped gate：

```bash
python unused/semantic_s0_needle_isolation.py \
  --dump <stage0_manifest.json> \
  --output <out_dir>/s0_3_kmax_ward.json \
  --lambda_rel 1.0,0.875 \
  --g_max inf,8192,4096 \
  --k_max unclipped,15,16,32,64,128 \
  --b_prime 128 \
  --workers 64 \
  --parallel_unit group \
  --log_timing
```

S0.6 clipped gate：

```bash
python unused/semantic_s0_anchor_dedup.py \
  --dump <stage0_manifest.json> \
  --output <out_dir>/s0_6_kmax_ward_rel1.json \
  --lambda_rel 1.0 \
  --g_max inf,8192,4096 \
  --l_block 0,1 \
  --k_max unclipped,15,16,32,64,128 \
  --b_prime 128 \
  --workers 64 \
  --parallel_unit group \
  --log_timing
```

分析输出：

```bash
python unused/semantic_s0_analyze.py '<out_dir>/*.json' --csv <out_dir>/summary.csv
```

### Stage 0 决策门

| 编号 | 通过标准 |
|---|---|
| S0.0 | 语义聚类在 key/value 方差上显著优于 single-cluster baseline |
| S0.2 | `K_eff(n)` 不呈明显幂律；若呈幂律，论文定位改为 `O(n^d log n)` |
| S0.3 | exact-entry needle 隔离显著高于同长度随机 span，且 clipped 后接近 unclipped 平台 |
| S0.4 | value 方差也下降；否则读出侧仍 smear |
| S0.5 | `(g_max, ℓ_block)` 能把 entry 跨度压到可接受区间 |
| S0.6 | fixed-3 物理 slot 数和 entry 数不把 memory story 打穿 |
| S0.7 | 若 value 作废比例高，才把 Γ delta-rule 纳入 v1 |
| S0.8 | `cache_batch` 与 `cache_serial` 的路由/cache/readout 分歧低到可解释；须单独覆盖跨 Phase1/Phase2 边界的构造 case，不能只看整体分歧率 |

### Stage 1 - 实现

按 `algorithm-spec.md` §5.18 顺序落地。默认关闭时必须保持现有数值行为；接口 breaking
迁移（`CacheAttentionState`）要和所有调用点同一改动完成。

### Stage 2 - eval-time 探测

新结构对旧 CPT 权重可能分布外，先看形状，不看绝对分数。

| 配置 | 目的 |
|---|---|
| A: `K_max=1` | 单簇参考点；不是现有 LogKV 的数值正确性闸门 |
| B: 纯语义 | `η=0, g_max=inf`，隔离时序分段贡献 |
| C: 主实验 | 使用 Stage 0 选出的 `lambda_rel/K_max:B′/g_max/ℓ_block` |

示例：

```bash
DIAG_ARGS="--log_kv_semantic_clusters true --log_kv_cluster_k_max 16 \
  --log_kv_cluster_lambda_rel 1.0 --log_kv_seg_gap_max 4096 \
  --log_kv_seg_block_level 1" \
    bash majob.sh exp/qwen1.7b-32k/arc_warmup.yaml
```

### Stage 3 - CPT

Stage 2 有信号后再训练。v1 没有新的连续标量需要 warmup；若纳入 Γ delta-rule，沿用
`second_order_scale` 的谨慎 warmup 思路。

## 7. 消融表

| 旋钮 | 取值 | 问题 |
|---|---|---|
| `(g_max, ℓ_block)` | 纯语义到低层分段保护 | 收益来自语义还是边界 |
| query 位置 | 尾部 / 中部 / 前置 | 区分本方案和 eviction 方法 |
| `K_max` | 1 / 4 / 16 / 64 / 128 | 预算绑定如何影响 needle |
| `K:B′` | 32×64 / 16×128 / 8×256 等 | 语义分辨率 vs 簇内精确容量 |
| `anchor_mode` | `lo_hi_mid` / `lo_hi` / `mid` / `z` | 位置表示贡献 |
| `λ_rel` | 0.5-2.0 | needle 召回和成本平衡 |
| `λ` | 0 / 1 | 原始 mass bias 是否必要；`-log(M)` 始终保留 |
| `γ` | 0 / 0.5 / 1 | 跨段遗忘是否有用 |
| Σ/Γ | 关 / 现有 / delta-rule | 二阶修正和作废信息 |
| vanilla memory-matched | 调大现有 LogKV `B` | 排除“只是多用内存” |

主指标仍用 ACC、LongBench、LongBench_e、niah@32768，并新增 multi-needle。multi-needle
必须同时报告 `K_max_binding_rate`。
