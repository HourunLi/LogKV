# 位置表示与注意力读出

本页对应 [共享算法](algorithm-spec.md) 的 semantic/Alpha 路径，源码为
[`log_kv_position.py`](../litgpt/log_kv_position.py)、
[`log_kv_pack.py`](../litgpt/log_kv_pack.py) 与
[`log_kv_pack_triton.py`](../litgpt/log_kv_pack_triton.py)。当前 Alpha 使用 `mid`；
`multi` 是保留的兼容模式，不是 Alpha 默认。

## 内容与位置分开存储

语义路由使用 qk-norm 后、RoPE 前的 `k_raw`。压缩 entry 存内容均值、value 均值、
真实 token 质量 `w` 和整数位置统计：

```text
w       = w_A + w_B
k_raw   = (w_A k_A + w_B k_B) / w
v       = (w_A v_A + w_B v_B) / w
p_lo    = min(真实成员位置)
p_hi    = max(真实成员位置)
sum_wp  = Σ 真实成员位置
p_mid   = clamp((2 sum_wp + w) // (2 w), p_lo, p_hi)
```

`sum_wp` 使用 int64，位置均值采用 round-half-up，只在读出时计算，避免逐层舍入。
整数位置统计在不溢出的范围内可结合；浮点 K/V 加权均值仍会受归约顺序影响。
`w=0` 的 padding 不贡献质量或边界，读出必须屏蔽。

这种做法避免直接平均不同 RoPE 相位，但不能恢复每个原始 token 的位置与内容关联。
`p_mid` 可以落在没有真实成员的位置。

## mid、multi 与精确槽

| 来源/模式 | 读出位置 | 有效锚点数 M |
|---|---|---|
| 压缩 entry，`mid` | `p_mid` | 1 |
| 压缩 entry，`multi` | 去重后的 `[p_lo, p_mid, p_hi]` | 1–3 |
| Alpha 精确 token | 原始 token 位置 | 1 |

以一阶 attention 为例，对 entry 的每个有效锚点 a：

```text
key_{s,a}   = RoPE(k_raw_s, p_a)
score_{s,a} = scale · q · key_{s,a} + λ log(w_s) − log(M_s)
value_{s,a} = v_s
```

`q` 已按查询位置应用 RoPE。`−log(M)` 独立于 `λ`，消除锚点复制本身带来的候选数放大；
它不使不同位置的得分等价，也不恢复 dense attention。mid 下该项为0。

Alpha 精确 token 的 `w=M=1`，保留原始 K/V、原始位置，质量偏置为0。
这里“精确”指缓存条目；经过前层压缩后，模型隐藏状态不保证等于 dense 模型。
当前 pack 顺序为 `[压缩槽, Alpha精确槽, recent, 当前块]`，历史条目只在当前块 attention
完成后提交，当前块内部仍受因果约束。

## 接口边界

- Alpha 配置关闭二阶修正和 segment gap/padding；多锚点与 Σ/Γ 属于其他兼容路径。
- semantic cache 需要可用的 RoPE 表和原始位置；`rope_interleave=True` 当前不支持。
- 压缩不等于位置外推：更长评测需另行确认模型位置编码和上下文配置。

必要检查在 `tests/test_log_kv_position.py`、`tests/test_log_kv_pack.py`、
`tests/test_alpha_log_kv.py`，覆盖位置、mask、质量偏置以及当前 K/V 梯度。
