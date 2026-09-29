# SemanticLogKV 算法规格

本文描述 `semanticLogKV` 分支当前实现。路由内核见
[统一路由](semantic-unified-routing.md)，位置读出见 [Position / RoPE](position.md)。
旧 Stage 0 路线图和未落地的设计不作为现行契约。

## 1. 当前入口与配置

统一路由训练配置是 `exp/qwen1.7b-32k/arc_semantic_unified_cpt100.yaml`，继承
`arc_semantic_fast.yaml`、`base.yaml`。单独使用 `arc_semantic_fast.yaml` 不会开启统一路由。
Python API 为兼容旧路径保留保守默认值，不能把不传参数等同于运行统一路由配置。

| 参数（省略 `log_kv_` 前缀） | Cache API 默认 | 当前统一路由配置 |
|---|---|---|
| `semantic_clusters` / `cluster_k_max` | `false` / `1` | `true` / `12` |
| `semantic_unified_route` | `false` | `true` |
| `B` | `512` | `128` |
| `semantic_flush_granularity` | `2` | `2048` |
| `semantic_anchor_mode` | `multi` | `mid` |
| `semantic_centroid_backend` | `sequential` | `parallel` |
| `semantic_replay_updates` | `false` | `true` |

当前 `recent_size`、`train_block`、`prefill_block` 均为 `2048`，上下文长度 `32768`；
二阶修正关闭，segment 间隔保护和 padding 关闭。短 CPT 从配置中的 Base 权重初始化，
若独立输出目录已有 checkpoint，`auto_resume` 会恢复该实验自己的训练状态。

## 2. 存储对象与数据流

| 名词 | 当前含义 |
|---|---|
| recent | 最近到达、尚未归档的精确 K/V 窗口 |
| flush | recent 溢出后，按固定粒度移交最老 token 的一次批处理 |
| candidate | 当前 flush 内形成的临时语义簇，仅用于决定成员归属 |
| cluster | 一个 `(batch, KV group)` 内的持久语义簇，最多 `K_max` 个 |
| entry / ladder | 簇内压缩条目 / 保存这些条目的多层结构 |
| slot / anchor | attention 读出条目 / 为原始 key 内容施加 RoPE 的位置 |
| mass `w` | entry 代表的真实 token 数；不是 entry 数，也不是重要性分数 |

模型在 Q/K normalization 后、RoPE 前保留 `k_raw`。聚类 centroid 和压缩 key 都在这个
内容空间中计算；当前块和 recent 的 attention 仍使用正常 post-RoPE key。每个层独立建
cache，语义路由按 `(batch, KV group)` 独立执行。

数据流为：当前块进行因果 attention → 写入 recent → 溢出部分 flush → 建簇与归档。
每个离开 recent 的 token 都进入语义层级；一个 token 不同时以 recent 精确副本和压缩
副本计入 attention。

## 3. 统一建簇与合并

当前 `semantic_unified_route=true` 不先把新 token 分配给旧簇：

1. 当前 flush 的每个归档 token 从单独候选开始，质量 `m=1`、半径 `r=0`。
2. 候选仅在本批内部按半径约束合并，**不提前压到 `K_max`**，允许单 token 候选留下。
3. 将候选中心与所有旧簇中心放在一起，按 Ward 代价合并到不超过 `K_max`。
4. 根据最终归属，执行旧簇合并、新簇创建和原 token 的批量写入。

Ward 代价和中心更新为：

```text
cost(a,b) = ma·mb/(ma+mb) · ||μa-μb||²
μnew      = (ma·μa + mb·μb)/(ma+mb)
mnew      = ma+mb
```

旧簇使用真实成员数 `n_total`，新候选使用成员 token 数。候选紧致性用覆盖半径上界约束：

```text
d    = ||μa-μb||
rnew = max(ra + mb/(ma+mb)·d, rb + ma/(ma+mb)·d)
rnew <= sqrt(cluster_lambda_rel · s_h)
```

`s_h` 可从离线标定文件按层、KV group 加载；未提供时实际为 `1`，不是在线估计的方差。
当前统一路由配置未提供标定文件，`cluster_lambda_rel=1`。全局预算合并没有该半径约束，
因此预算压力下最终簇可以变宽，不能把候选半径当成最终信息损失的保证。

每轮选多对互为最近邻、互不重叠的簇，按代价限制合并数量，避免低于目标簇数。
默认使用设备上的增量距离更新和最近邻缓存；轮间只传递小型活动标志，
结束后回传归属记录。它保留每轮更新中心的规则，但浮点舍入与固定索引 tie-break 不保证
与旧版重新压紧编号的实现逐位一致。

有限 hard cap 使用兼容重算路径；当前统一路由配置的容量惩罚与 hard cap 均关闭。

## 4. 簇内压缩与统计量

路由候选只决定成员归属，**不会把候选中心当成一个 summary 写入 cache**。每个归档 token
以 `w=1` 的 entry 进入 level 0；`log_kv_semantic_summary_size` 已移除，旧 YAML 含该键
会报错，应删除，不能继续通过设成 `1` 启用新行为。

每簇每层最多 `B′` 个 entry，层满时把最老内容按时间顺序相邻配对、加权合并后向上进位。
顶层饱和时继续合并最老内容，保持预算和 mass，不因容量直接删除 token。

```text
wnew      = wa+wb
k_raw_new = (wa·ka + wb·kb)/wnew
vnew      = (wa·va + wb·vb)/wnew
p_lo_new  = min(p_lo_a,p_lo_b)
p_hi_new  = max(p_hi_a,p_hi_b)
sum_wp_new = sum_wp_a + sum_wp_b
```

位置和 mass 必须跟随归档；同一簇内按真实位置写入，旧簇合并时重排其现存条目，
不把已有压缩 entry 拆回原 token。

路由 centroid 与 ladder 条目是两套状态。`n_total` 保存真实成员数，`n_eff` 用于 centroid
更新；当前关闭 segment 遗忘后使用所有已归档成员的均值。并行归约可改变浮点末位。
兼容 segment 路径可衰减 `n_eff`、插入零 mass pad；它不改变真实 `n_total`。

二阶路径仍支持 Chan-style 合并后的 rank-1 K 协方差、K/V 互协方差近似，供旧配置使用。
当前统一路由配置不分配这些统计量，也不计算二阶 attention 修正。

## 5. 位置、mass 与 attention

`mid` 读出每个有效 entry 一个 slot，使用
`p_mid=clamp((2·sum_wp+w)//(2·w),p_lo,p_hi)`，读出时才对 `k_raw` 施加 RoPE。
兼容 `multi` 路径对 `lo/mid/hi` 去重，得到 `M∈{1,2,3}` 个 slot。

```text
score(s,a) = scale·q·RoPE(k_raw_s,p_a) + log(w_s) - log(M_s)
value(s,a) = v_s
```

这是当前一阶模型调用的 mass bias（系数为 `1`）；底层通用 attention 函数另支持 `lam`。
`mid` 中 `M=1`；recent、当前块均为 `w=M=1`。空槽必须遮罩，当前块保留因果约束。
`CacheAttentionState` 显式携带 `slot_valid/M_s`，不能把 padding 当作真实 token。

`log_kv_pack.py` 的快速路径直接打包“压缩前缀 + recent + 当前块”，用 key
的附加维度携带 `log(w)` 和无效槽屏蔽值，再调用 Flash SDPA。`auto` 在支持的 CUDA 布局
上使用 Triton pack；不满足条件时走共享数学定义的回退路径。decode 可复用未变化前缀，
flush 后失效。更详细的位置约定见 [Position / RoPE](position.md)。

## 6. 显存边界

语义层数根据配置上限 `N` 分配：

```text
L = max(2, ceil(log2(N/(K_max·B′)+1)) + 2)
persistent entries per batch/group = K_max·L·B′
```

固定 `K_max`、`B′` 和 recent 容量时，**持久推理 KV cache** 随配置上下文上限增长
为 `O(log N)`。不要把 `K_max` 也随 `log N` 增长后仍宣称同一复杂度。该结论不包含模型
权重、RoPE 表、输入/output 张量或训练图，也不表示运行中自动扩展配置上限。

当前 32K、`K_max=12`、`B′=128` 时分配 `L=7`，每个样本、每个 KV group 最多
`12×7×128=10752` 个持久 entry。实际显存还包括 mass、位置、路由元数据和 pack 工作区。
路由临时距离矩阵按 lane 分块，单块为 `O((F+K_max)²)`，其中 `F` 是固定 flush 大小；
这类临时峰值仍须计入实际显存测量。

## 7. 训练与重放

`LogKVStreamTrainingAttention` 以块流式前向：块开始时冻结 prefix，块内对当前 K/V 做
精确因果 attention；写入 cache 的状态 detach。梯度只沿当前块的 Q/K/V 传播，不穿过
历史缓存和路由选择。因此它对应固定 `train_block` 下的流式训练目标，不等于 dense 训练。

反向重置 cache，使用前向记录重放并逐块重算 attention；activation checkpoint 重算也复用
同一次前向的路由。当前 `semantic_replay_updates=true` 还保存 detached 的实际写入 K/V、
小型状态与更新动作，使重放复用同一批量写入结果，避免重新分簇和浮点轨迹漂移。
关闭该选项的兼容路径使用 op-log 与解析后的调度重建状态。

这些记录由各自 autograd 调用持有，不能借用另一个输入的缓存；backward 完成后随图释放。
**训练内存不是 `O(log N)`**：它还包括全序列 Q/K/V、attention plan、更新记录和单块重算图。
推理不保留整个输入历史的训练更新记录。

## 8. 支持边界与真实限制

- 语义路径不支持 interleaved RoPE、MultiheadLatentAttention 或 importance pooling。
- 模型入口要求共享、连续、只追加的 `input_pos`；cache 不是任意位置可覆写的存储器。
- 半径与 Ward 都不能保证任意远程事实无损；质量结论需要真实检索评测，不能从 mass
  守恒或随机路由基准推出。
- flush 之间依赖前一批 cache 状态，仍按顺序推进；GPU 并行覆盖 lane、配对和批量写入，
  当前仍有 host 元数据工作与轮间同步，不能宣称完全无同步。
- 旧三阶段路由、legacy/chunk 路由、`multi` 和二阶修正仅是兼容或对照入口。当前统一路由
  配置不走这些组合；更改路由或位置近似时，应保持训练与评测设置一致。

主要实现：`litgpt/log_kv_cache.py`、`litgpt/log_kv_route_triton.py`、`litgpt/model.py`、
`litgpt/log_kv_pack.py`。维护时优先检查质量/mass、原始位置、预算和 forward/replay 一致性；
性能数值必须注明设备、输入规模和是否包含 profiler 开销。
