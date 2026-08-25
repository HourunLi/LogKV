# SemanticLogKV Position / RoPE 手册

> 精简版。position 的生产细节以 `algorithm-spec.md` §5.10-§5.11 为准；本文只解释当前
> 方案和验证口径。

## P0. 一句话概览

entry 是存储单位，virtual slot 是读出单位：

```text
entry:
  k_raw_mean, v_mean, w, p_lo, p_hi, sum_wp

readout:
  p_mid  = round_half_up(sum_wp / w)
  anchors = dedup([p_lo, p_mid, p_hi])
  entry -> M 个 virtual slots, M ∈ {1,2,3}

slot:
  key   = apply_rope(k_raw_mean, anchor)
  value = v_mean
  bias  = λ·log(w) - log(M)
```

## P1. 要解决的问题

RoPE 把位置信息写成旋转。不同位置的 post-RoPE key 直接均值，会让高频相位互相抵消：

```text
k_slot = mean_i R(p_i) k_i
```

这个向量通常不再等价于任何一个合法位置上的 token key。语义聚类会让同一 cluster 的
成员跨全文复现，位置跨度更大，所以必须把“内容均值”和“位置表示”拆开。

## P2. 设计目标

1. 读出 key 必须由标准 `apply_rope` 得到。
2. v1 training-free first，不依赖新位置编码训练。
3. 不平均 RoPE 相位。
4. attention 只看扁平 slot 池，不读 cluster/segment。
5. 每个选择能被 Stage 0/2 证伪。

## P3. 当前方案

存 pre-RoPE 内容均值和整数位置锚点：

```text
p_lo   = min member position
p_hi   = max member position
sum_wp = Σ w_j · p_j
p_mid  = clamp((2·sum_wp + w) // (2·w), p_lo, p_hi)
```

`p_mid` 是质心位置，不要求是真实 token。抗相消性只依赖“每个 virtual slot 在一个确定位置
上调用标准 RoPE”，不依赖该位置是否真实出现过。

## P4. 合并

```text
w       = w_A + w_B
k_raw   = weighted_mean(k_A, k_B, w_A, w_B)
v       = weighted_mean(v_A, v_B, w_A, w_B)
p_lo    = min(p_lo_A, p_lo_B)
p_hi    = max(p_hi_A, p_hi_B)
sum_wp  = sum_wp_A + sum_wp_B
```

不存逐层舍入的 `p_mid`，只存 `sum_wp`。这样合并可结合，重放不受树形顺序影响。

## P5. attention 公式

```text
score_{s,a} = scale · (q · apply_rope(k_raw_s, p_a)) + λ·log(w_s) - log(M_s)
value_{s,a} = v_s
```

`-log(M_s)` 必须无条件存在。否则一个 entry 展开成多个 anchor 时，会因为候选数变多而拿到
额外 softmax 质量。

value 不随 anchor 变化；anchor 只影响 key 的位置旋转。

## P6. 三个锚点的作用

| 锚点 | 直觉 |
|---|---|
| `p_lo` | 覆盖最早成员，保留左边界 |
| `p_hi` | 覆盖最晚成员，保留右边界 |
| `p_mid` | 覆盖集中位置和 span 中部 |

三锚点不是恢复原始位置分布，只是用少量确定位置避免相位均值。

## P7. segment 的关系

segment 不存 position；它只通过 level-0 pad 约束低层不要跨段合并。`ℓ_block=0` 关闭保护，
`1/2` 是可负担档，更深会浪费太多 entry。高层仍允许跨 segment，这是预算压力下的有意
降级。

## P8. 验证问题

| 问题 | 对应实验 |
|---|---|
| 三锚点是否有用 | S0.6 + Stage 2 anchor 消融 |
| 是否需要 gather/packed | S0.6 的 `E[M]` 和 fixed-3 成本 |
| 是否需要 learned position | 只有 `lo_hi_mid` 明显不够时再考虑 |
| segment 是否压住跨度 | S0.5 / Stage 2 `(g_max, ℓ_block)` |
| 模型是否接受多位置同内容 | Stage 2 实测 |

## P9. 外部脉络

- RoPE：用标准 `apply_rope`，不发明 frozen model 没见过的位置编码。
- semantic clustering KV cache：多数工作是 eviction 或索引，不是严格次线性压缩。
- decoupled position：支持把内容压缩和位置表达分离。
- learned compaction：可作为失败后的后续方向，不进 v1。

## P10. 当前决策

已定：

- v1 用 `lo/mid/hi` 三锚点。
- 存 pre-RoPE key 内容均值。
- `-log(M)` 独立于 `λ`。
- `p_mid` 由 `sum_wp` 在读出时计算。
- `rope_interleave=True` 先硬失败。

未定：

- 是否保留三锚点还是降为 `lo_hi`。
- 是否值得做 gather/packed。
- 是否需要 learned anchor bias。
