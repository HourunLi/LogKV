# SemanticLogKV / AlphaLogKV / BetaLogKV / GammaLogKV 算法规格

本文描述当前分支的共享机制：attach 路由分簇 + Alpha 精确片段 + Beta + Gamma。Alpha 的片段选择与
预算见 [AlphaLogKV](alpha-logkv.md)，Beta 的评分与自适应压缩见 [BetaLogKV](beta-logkv.md)，
Gamma 的延迟归档与顶层合并见 [GammaLogKV](gamma-logkv.md)，
可选的 unified 路由内核见 [统一路由](semantic-unified-routing.md)，位置读出见
[Position / RoPE](position.md)。旧 Stage 0 路线图和未落地的设计不作为现行契约。

## 1. 当前入口与配置

当前训练配置是 `exp/qwen1.7b-32k/arc_gamma_k12_b128_2k.yaml`，继承 `arc_semantic_fast.yaml`、
`base.yaml`；算法开关集中在 `arc_semantic_fast.yaml`。训练/评测入口（`majob.sh`、`eval.sh`、
`demo.py`、`eval.py`）对未写出的键同样默认开启 Alpha/Beta/Gamma、关闭 unified；非语义簇路径
自动关闭 Alpha/Beta，`alpha_exact_tokens=0` 同时关闭 Beta（Gamma 的第 0 层重排随之无作用）。
Cache API 为兼容旧路径保留保守默认值，不能把直接构造 cache 不传参数等同于默认路线。

| 参数（省略 `log_kv_` 前缀） | Cache API 默认 | 入口/当前配置 |
|---|---|---|
| `semantic_clusters` / `cluster_k_max` | `false` / `1` | `true` / `12` |
| `semantic_unified_route` | `false`（attach） | `false`（attach） |
| `B` | `512` | 原始预算 `128`，为精确槽扣减后使用 |
| `semantic_flush_granularity` | `2` | `2048` |
| `semantic_anchor_mode` | `multi` | `mid` |
| `semantic_centroid_backend` | `sequential` | `parallel` |
| `semantic_replay_updates` | `false` | `true` |
| `alpha_exact_tokens` / `alpha_span_max_tokens` | `0` / `64` | `256` / `64` |
| `beta_novelty` / `beta_adaptive_merge` | `false` / `false` | `true` / `true` |
| `gamma_level0_reinsert` / `gamma_top_merge` / `gamma_level_slack` | `false` / `fold` / `2` | `true` / `lightest` / `2` |

当前 `recent_size`、`train_block`、`prefill_block` 均为 `2048`，上下文长度 `32768`；
二阶修正关闭，segment 间隔保护和 padding 关闭，`seg_eta=1`、`seg_g0=2048`、
`cluster_lambda_rel=1`、未提供 `s_h` 标定。CPT 从配置中的 Base 权重初始化，若独立输出目录
已有 checkpoint，`auto_resume` 会恢复该实验自己的训练状态。

更早实验的 YAML 显式写出各自训练时的开关，复评不受新默认值影响：纯 attach / unified 基线
（`arc_semantic_attach_route_*`、`arc_semantic_unified_route_*`、`arc_semantic_unified.yaml`）
关闭 Alpha/Beta；`arc_alpha_*` 为 unified + Alpha；`arc_beta_cpt100.yaml` 为 unified + Alpha + Beta；
`base.yaml` 派生的 Stage-1（multi、二阶）关闭两者；`arc_attach_alpha_beta_*` 为 Gamma 之前的
attach + Alpha + Beta。这些 YAML 都显式写回旧梯子（`gamma_level0_reinsert: false`、`gamma_top_merge: fold`）。

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
Alpha 在建簇前增加精确片段筛选：保留片段进入固定精确池，其余 token（含被替换的旧片段）
按配置的路由（默认 attach）归档。一个 token 不同时以精确副本和压缩副本计入 attention。

## 3. 建簇路由

### 3.1 attach（默认，`semantic_unified_route=false`）

每层、每个样本和 KV group 独立处理一次 flush 的归档 token，flush 内旧簇中心冻结：

1. **直连旧簇**：对 token `x` 与每个存活簇 `c` 计算

   ```text
   gap(x,c)  = max(pos(x) − p_hi(c), 0)
   cost(x,c) = ||x − μ_c||² + seg_eta · gap / (gap + seg_g0)
   ```

   取代价最小的簇；若其语义项 `||x − μ_c||² ≤ cluster_lambda_rel · s_h`，token 直接归入该簇。
   可选容量惩罚和 hard cap 在当前配置中关闭。
2. **orphan 开新簇**：未直连的 token 至多开 `空闲槽数` 个新簇。第一个种子取与其最优旧簇
   语义距离最大的 orphan，其余按最远点依次选取；种子只是本次 flush 的临时原型。
3. **其余 orphan**：按欧氏距离并入最近的存活簇或新种子，不再检查半径；没有空闲槽时只能
   并入旧簇，簇因此可能变宽。整个路由不做 Ward 合并，簇数不超过 `K_max`。
4. **写入**：每簇的 token 按位置排序，原始 K/V 以 `w=1` 批量进入层级，中心按成员均值
   更新；新簇从其最早的 token 开始。Alpha 被替换的旧精确 token 位置可能早于簇的 `p_hi`：
   Gamma 只把这类簇的第 0 层与新 token 一起按位置重排后追加，上层条目不动；旧行为把整簇
   全部条目重新入层级（见 [Gamma](gamma-logkv.md)）。

实现（无有限 hard cap、非 legacy 时）：所有组按位置序在设备上完成上述判定，只回传每个
token 的最终簇号（int8），主机用一次稳定排序得到（组，簇，位置）提交计划；不在存活集合里
的簇即新簇。所有簇都已存活（第一次 flush 之后的常态）时，一个 Triton 核按 64 个 token
一块完成直连代价、argmin、直连判定和 orphan 的精确最近簇；有空闲簇时用 Torch 实现最远点
播种。orphan 距离用精确差平方而非 GEMM 展开，使结果与批形状无关。有限 hard cap 或 legacy
保留两阶段主机规划，二者对同一输入的缓存、镜像和 op-log 逐位一致（有测试）。
Alpha 的不等长归档按样本数量截断，padding 不参与分配也不写入。

可选 `LOGKV_ROUTE_OVERLAP=1`（仅 CUDA、attach）：一次 flush 的决策只依赖本次归档的 K/V
和上一次 flush 提交的簇状态，因此在高优先级侧流上执行（快照、片段选择、归档收集、attach
核与唯一一次回读），只等待上一次 flush 结束的事件，不等待其间排队的注意力；主机在注意力
运行时完成提交规划，精确池重写、梯子追加、中心更新和窗口平移回到主流、排在注意力之后。
只有本次归档的行在上一次 flush 结束时已在窗口中才走侧流；分块与 flush 不对齐、之后才写入的
行留在主流。
默认关闭：先在目标 GPU 上运行 `python unused/check_route_overlap.py`，训练与预填充两条路径
的输出、梯度、缓存与主机镜像逐位一致后再开启。
该脚本仅在反向检查时开启严格确定性算法，排除 FlashAttention 默认反向归约的非确定性；
仍以零容差比较，失败会报告差异数量与最大误差。该设置不改变正常训练的算法开关。

### 3.2 unified（可选，`semantic_unified_route=true`）

unified 不先把新 token 分配给旧簇：

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
当前配置未提供标定文件，`cluster_lambda_rel=1`。全局预算合并没有该半径约束，
因此预算压力下最终簇可以变宽，不能把候选半径当成最终信息损失的保证。

每轮选多对互为最近邻、互不重叠的簇，按代价限制合并数量，避免低于目标簇数。
`merge_passes=1` 默认使用设备上的增量距离更新和最近邻缓存，行扫描只遍历设备端存活节点表；
CUDA 约每 4 轮检查一次小型活动标志与存活数，结束后回传归属记录。它保留每轮更新中心的规则，但浮点舍入与固定索引 tie-break 不保证
与旧版重新压紧编号的实现逐位一致。

`merge_passes=4` 保留冻结中心的近似对照：同一距离计算进行四次配对扫掠，再统一更新中心；
已配对簇的合并结果到下一轮才参与竞争。这个选项会改变全局配对结果，不只是换计算后端。
它只在无限容量的全局预算阶段使用，候选半径阶段不采用该近似。有限 hard cap 使用兼容
重算路径；当前配置的容量惩罚与 hard cap 均关闭。

## 4. 簇内压缩与统计量

路由候选只决定成员归属，**不会把候选中心当成一个 summary 写入 cache**。每个归档 token
以 `w=1` 的 entry 进入 level 0；`log_kv_semantic_summary_size` 已移除，旧 YAML 含该键
会报错，应删除，不能继续通过设成 `1` 启用新行为。

每簇每层最多 `B′` 个 entry，层满时把最老内容按时间顺序相邻配对、加权合并后向上进位。
顶层饱和时继续合并以保持预算和 mass，不因容量直接删除 token：Gamma 每来一条合并质量和最小的
相邻对（并列取最老），旧行为把 slot 0 与 slot 1 左折叠，最老内容会堆进一个越来越重的条目。
Beta 在非顶层批量进位中，把同一 lane 内连续四个待合并 entry 切为两组：
比较 `2|2`、`1|3`、`3|1` 的归一化 K/V 失真，保留代价最低者；不足四个的两条尾部仍二合一。
预算、进位条目数和下列加权统计规则不变；这里的连续指簇内 entry 顺序，不是连续原文 token。

```text
wnew      = wa+wb
k_raw_new = (wa·ka + wb·kb)/wnew
vnew      = (wa·va + wb·vb)/wnew
p_lo_new  = min(p_lo_a,p_lo_b)
p_hi_new  = max(p_hi_a,p_hi_b)
sum_wp_new = sum_wp_a + sum_wp_b
```

位置和 mass 必须跟随归档；Alpha 旧精确片段可能晚于新 token 入层级，所以归档按原始位置
插入。不能把“晚归档”当成“新位置”，也不能降低簇已记录的最大位置。

路由 centroid 与 ladder 条目是两套状态。`n_total` 保存真实成员数，`n_eff` 用于 centroid
更新；当前关闭 segment 遗忘后使用所有已归档成员的均值。并行归约可改变浮点末位。
兼容 segment 路径可衰减 `n_eff`、插入零 mass pad；它不改变真实 `n_total`。

二阶路径仍支持 Chan-style 合并后的 rank-1 K 协方差、K/V 互协方差近似，供旧配置使用。
当前 Alpha 不分配这些统计量，也不计算二阶 attention 修正。

## 5. 位置、mass 与 attention

`mid` 读出每个有效 entry 一个 slot，使用
`p_mid=clamp((2·sum_wp+w)//(2·w),p_lo,p_hi)`，读出时才对 `k_raw` 施加 RoPE。
兼容 `multi` 路径对 `lo/mid/hi` 去重，得到 `M∈{1,2,3}` 个 slot。

```text
score(s,a) = scale·q·RoPE(k_raw_s,p_a) + log(w_s) - log(M_s)
value(s,a) = v_s
```

这是当前一阶模型调用的 mass bias（系数为 `1`）；底层通用 attention 函数另支持 `lam`。
`mid` 中 `M=1`；精确池、recent、当前块均为 `w=M=1`。空槽必须遮罩，当前块保留因果约束。
`CacheAttentionState` 显式携带 `slot_valid/M_s`，不能把 padding 当作真实 token。

`log_kv_pack.py` 的快速路径直接打包“压缩前缀 + Alpha 精确池 + recent + 当前块”，用 key
的附加维度携带 `log(w)` 和无效槽屏蔽值，再调用 Flash SDPA。`auto` 在支持的 CUDA 布局
上使用 Triton pack；不满足条件时走共享数学定义的回退路径。decode 可复用未变化前缀，
flush 后失效。更详细的位置约定见 [Position / RoPE](position.md)。

## 6. 显存边界

语义层数根据配置上限 `N` 分配：

```text
L = max(2, ceil(log2(N/(K_max·B′)+1)) + slack)        # slack = gamma_level_slack，默认 2
persistent entries per batch/group = K_max·L·B′
```

固定 `K_max`、`B′`、recent 和精确池容量时，**持久推理 KV cache** 随配置上下文上限增长
为 `O(log N)`。不要把 `K_max` 也随 `log N` 增长后仍宣称同一复杂度。该结论不包含模型
权重、RoPE 表、输入/output 张量或训练图，也不表示运行中自动扩展配置上限。

`B` 始终表示两层余量的经典梯子的字节。Alpha 从中扣出精确池；`slack≠2` 时在同一字节内取
最大的 `B′`（`slack 0`、K=12、32K、P=256 时 B′=218、4 层），然后重新计算有效 `B′` 与 `L`。比较口径包含持久 KV、
相关元数据和最大复用 pack 工作区，不缩短 recent；具体公式和实际预算见
[AlphaLogKV](alpha-logkv.md)。路由临时距离矩阵按 lane 分块，单块为 `O(F²)`，其中 `F`
是固定 flush 候选规模；这类临时峰值仍须计入实际显存测量。

## 7. 训练与重放

`LogKVStreamTrainingAttention` 以块流式前向：块开始时冻结 prefix，块内对当前 K/V 做
精确因果 attention；写入 cache 的状态 detach。梯度只沿当前块的 Q/K/V 传播，不穿过
历史缓存和路由选择。因此它对应固定 `train_block` 下的流式训练目标，不等于 dense 训练。

反向重置 cache，使用前向记录重放并逐块重算 attention；activation checkpoint 重算也复用
同一次前向的路由。当前 `semantic_replay_updates=true` 还保存 detached 的实际写入 K/V、
小型状态与更新动作，使重放复用同一批量写入结果，避免重新分簇、选择精确片段和浮点轨迹
漂移。关闭该选项的兼容路径使用 op-log 与解析后的调度重建状态。
Beta 还记录每次局部压缩的切分位置；checkpoint 和 backward 直接应用记录，不重新计算切分代价。

这些记录由各自 autograd 调用持有，不能借用另一个输入的缓存；backward 完成后随图释放。
**训练内存不是 `O(log N)`**：它还包括全序列 Q/K/V、attention plan、更新记录和单块重算图。
推理不保留整个输入历史的训练更新记录。Alpha 训练要求 update replay 开启且二阶修正为零。

## 8. 支持边界与真实限制

- 语义路径不支持 interleaved RoPE、MultiheadLatentAttention 或 importance pooling。
  Alpha 还要求 attach 或 unified 路由（不支持 legacy/chunk-tree）、`K_max>1`、`mid`、
  关闭二阶分配和 segment gap/padding。
- 模型入口要求共享、连续、只追加的 `input_pos`；cache 不是任意位置可覆写的存储器。
- 半径、Ward 和固定精确池都不能保证任意远程事实无损；精确片段选择是启发式，质量结论
  需要真实检索评测，不能从 mass 守恒或随机路由基准推出。
- flush 之间依赖前一批 cache 状态，仍按顺序推进；GPU 并行覆盖 lane、配对和批量写入。
  host 元数据经 pinned 内存异步上传，不再隐式同步 stream，回放可与已排队的注意力计算重叠。
  默认 attach 前向每次 flush 只剩一次决策回读（片段分数已在上一次 flush 后预取）；默认
  情况下这次回读仍会等待其前排队的注意力，`LOGKV_ROUTE_OVERLAP=1` 去掉这层等待。
  unified 路由仍有轮询与 trace 回传，不能宣称完全无同步。
- legacy/chunk 路由、`multi` 和二阶修正仅是兼容或对照入口；unified 是保留的对照路由。
  更改路由、位置近似或精确槽策略时，应保持训练与评测设置一致。

主要实现：`litgpt/log_kv_cache.py`、`litgpt/log_kv_route_triton.py`、`litgpt/model.py`、
`litgpt/log_kv_pack.py`。维护时优先检查质量/mass、原始位置、预算和 forward/replay 一致性；
性能数值必须注明设备、输入规模和是否包含 profiler 开销。
