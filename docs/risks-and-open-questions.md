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
| wall-clock 退化 | 逐 token host-device 同步（`.item()`/`nonzero()`）使单步前向达到秒级，比 Phase 2 orphan 数量本身更致命 | 先消同步再测；消除后 Phase 2 残余串行步数量级是每文档 `O(K_max)`，不是 `O(T)` |
| Phase1 batch 边界盲区 | 同一 flush batch 内，Phase 2 才新建的簇对该 batch 中 Phase1 已处理完的 token 不可见，可能把本该并入新簇的 token 错分进旧簇 | S0.8 除总体分歧率外，专测"batch 内先后出现的相似 orphan token 跨 Phase1/Phase2 边界"这类构造 case |
| `flush_granularity` 定下后不是自由旋钮 | 训练/推理共享同一 `route_and_flush_batch`，近似质量依赖窗口大小；类比 chunk-wise 训练文献（TNT, arXiv:2511.07343）里训练/推理 chunk size 不匹配会掉分 | 定下 `flush_granularity` 后固定，不当推理期性能旋钮随意调；要调则重新验证 |

## 11. 技术难点清单

### A. 训练重放确定性

现有 `LogKVStreamTrainingAttention` 能重放，是因为结构只依赖计数。语义路由依赖
浮点距离和 `argmin`，不能在 backward 里重算。解决方式：forward 记录 `op_log`，
backward 只回放操作序列。

### B. 路由吞吐

逐 token 串行不可接受。现状（`litgpt/log_kv_cache.py`）：判定阶段已矩阵化——一次性算出
全 batch 的 winner/direct 归属，这部分不再是瓶颈；但 Phase 1 把已判定的 direct token 逐个
写入 ladder（含 `.item()` 同步）还是 per-token 循环，是当前真正的瓶颈；Phase 2 的 orphan
循环设计上就该串行，不是问题。

要分清两类状态更新：数值累积（centroid 更新、ladder carry）满足结合律，可以用批量归并
树处理——Phase 1 的 ladder 写入应该走这条路但目前没有；只有离散决策
（`NEW_CLUSTER`/`NEW_SEGMENT`/Ward 配对选择）改变簇集合本身，不满足结合律，是唯一必须
保持串行的部分，数量级是每文档 `O(K_max)`，不是 `O(T)`。§5.22 point 9 是这个批量化的
具体要求。

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

### J. CPT 是否必须让真实语义聚类天天占关键路径

调研 KV-CAT（`Training Transformers for KV Cache Compressibility`, arXiv:2605.05971）：
训练期只用一个便宜、并行、可微的 router 做自蒸馏（KL loss，dense 前向当 teacher，
`ℒ=λ_mask·ℒ_mask+λ_budget·ℒ_budget+λ_anchor·ℒ_anchor`），router 推理时整个丢弃，真实压缩
（Attention Matching / 梯度优化）是非因果、离线、单独跑的。机制本身和本方案不兼容——簇
路由不可微、必须流式在线，KV-CAT 两段都不满足；论文也没做"代理训练 vs. 真实机制训练"的
消融，不能当作"代理能顶替真实机制"的证据，只能当"训练期用便宜代理可行"的存在性参考。

可借鉴的是策略，不是机制：多数 CPT step 用现有按位置 LogKV（`semantic_clusters=False`，
已验证可用）代替真实语义聚类，只有少数 step 或收尾阶段切到真实语义聚类对真实 LM loss
训练——直接复用 `LogStructuredKVCache` 已有的 `semantic_clusters` 开关按样本/batch 切换，
不需要新的可微 loss。Stage 2（旧 CPT 权重在新结构上 zero-shot 探测）是这个思路的 0% 配比
端点，应先看 Stage 2 结果再决定要不要投入这条 curriculum 路线。

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
| CPT 训练 wall-clock | Stage 1 补合成 benchmark；判定阶段已矩阵化，Phase 1 的 ladder 写入还是逐 token，闸门要在 §5.19/§5.22 point 9 的批量写入做完后跑才有意义 |
| 是否需要 curriculum 混合训练替代全程真实语义聚类 CPT | 待 Stage 2 zero-shot 结果定；结果差则优先修 §11.B 的同步开销和精确 Phase1/2，而不是投入 curriculum |

## 13. 压缩机制的剩余空间

这些方向不进 v1，除非 Stage 0/2 明确指向它们：

- recent window 与 per-cluster level-0 预算重分。
- 沿 ladder 层级分配不同 `B′_ℓ`。
- Σ/Γ 非对称秩分配。
- gather/packed 只物化有效锚点。
- learned anchor bias 或 learned position branch。
- Γ 改成 delta-rule 构造以处理 supersession。
