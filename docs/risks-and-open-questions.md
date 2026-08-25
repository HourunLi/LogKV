# SemanticLogKV 风险、技术难点与未决问题

> 精简版。已定案的历史争论不再保留，只记录会影响实现或实验解释的风险。

## 8. 风险与对策

| 风险 | 后果 | 对策 |
|---|---|---|
| key 空间未必是语义空间 | 某些层/组聚类无效 | 所有 Stage 0 统计按层、按正确头粒度报告 |
| `λ_rel` 敏感 | needle 并入大簇，或簇数过多 | S0.3/S0.6 联合裁决；`1.0` 主线，`0.875` 只作高召回候选 |
| `K_max` binding + Ward | 预算绑定后合并掉 needle 小簇 | 报告 `K_max_binding_rate`、needle Ward touched/merged 和 final exact-entry |
| key 聚类不保证 value 同质 | 分数侧好，答案读出仍错 | S0.4 同时测 key/value 方差 |
| 早期路由错误不可恢复 | 冷启动错误永久留在 cache | `s_h` 离线标定；冷启动策略另测 |
| 锚点展开增加读出槽 | matmul 宽度可能超过 vanilla | 从第一轮实验报告 fixed-3 slot 数和 memory-matched baseline |
| `-log(M)` 漏掉或被 `λ` 门控 | 展开的 entry 被静默放大 | `log_kv_slot_attention` 单测守恒 |
| pad 字段未清零 | `0 * NaN` 污染真实 entry | pad 全字段 finite zero，读出再 mask |
| 训练 `op_log` 保存错误 | backward 重放错误且不报错 | `op_log` 每 forward 新分配，存 `ctx`，不用原地清零 |
| wall-clock 退化 | Phase 2 串行 orphan 过多 | Stage 1 补 microbenchmark |

## 11. 技术难点清单

### A. 训练重放确定性

现有 `LogKVStreamTrainingAttention` 能重放，是因为结构只依赖计数。语义路由依赖
浮点距离和 `argmin`，不能在 backward 里重算。解决方式：forward 记录 `op_log`，
backward 只回放操作序列。

### B. 路由吞吐

逐 token 串行不可接受。必须用 §5.4 的三阶段批量路由，并用 S0.8 验证它和严格串行参考
的分歧。

### C. 预算定尺

`SEG_max` 不进预分配。每 cluster 一条 ladder，segment 只通过 pad 约束合并。生产前必须
按最终 `K_max:B′:L_alloc` 重算持久 entry、fixed-3 slot 和训练 `op_log` 三笔账。

### D. value smear

按 key 聚类不等于按 value 聚类。若 S0.4 显示 value 方差高，优先考虑 `[k;v]` 联合聚类
或簇内 value 二次分裂；不要先上复杂 state。

### E. flush 调度

flush 必须是 token index 的确定函数，不能依赖训练 chunk 边界或 eval prefill 分块。
`_log_kv_pending` 需要从“留 1 个”推广为“留 `T mod flush_granularity` 个”。

### F. 冷启动

文档开头和 attention sink 可能播种不代表全文的 centroid。候选缓解：窗口播种、前若干
token 更激进更新率、sink 专用通路。未定。

### G. 压缩函数不会被训练

cache commit 仍是 stop-gradient。CPT 只能让模型适应启发式划分，不能学出更好的划分。
`λ_rel/g_max/K_max` 要靠 Stage 0 和 eval 扫参。

### H. `λ·log(w)` 分布变化

语义簇下 `w` 按内容分布，不再是位置层级的代理。`λ` 需要消融；`-log(M)` 不是消融项。

### I. 层/组差异

很可能只有中层、部分 KV group 有效。生产可考虑只启用有效层/组，但这要由 Stage 0 数据
决定。

## 12. 未决问题

| 问题 | 当前默认 |
|---|---|
| decode 阶段是否共享同一套簇 | v1 共享；必须在长 CoT/多轮中测 prompt 事实是否被 decode 内容挤掉 |
| 标定样本数 | 跟 S0.2 一起测稳定性 |
| attention sink 是否占预算 | 未定；sink 专用通路只是候选 |
| multi-needle 与 `K_max` | 未定；multi-needle 必报 binding |
| 冷启动策略 | 未定 |
| eval-time 机制探针 | 未设计；需要记录 needle 最终 entry/簇状态 |
| 跨层共享聚类 | 不进 v1 |
| 模型是否接受“同内容多位置 virtual slots” | Stage 2 才知道 |
| CPT 训练 wall-clock | Stage 1 补合成 benchmark |

## 13. 压缩机制的剩余空间

这些方向不进 v1，除非 Stage 0/2 明确指向它们：

- recent window 与 per-cluster level-0 预算重分。
- 沿 ladder 层级分配不同 `B′_ℓ`。
- Σ/Γ 非对称秩分配。
- gather/packed 只物化有效锚点。
- learned anchor bias 或 learned position branch。
- Γ 改成 delta-rule 构造以处理 supersession。
