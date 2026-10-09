# BetaLogKV：同预算下减少信息损失

Beta 基于 Alpha，同时开启两项改动：精确片段评分加入旧簇新颖度，层级压缩选择局部失真最小的切分。
保持 `K=12`、精确池 `P=256`、片段上限 `64`、原预算 `B=128`、recent/flush `2048`，
默认与 attach 路由分簇组合；unified 路由（`merge_passes=1`）保留为对照。精确池的预算扣款、
自然边界和整段替换沿用 [Alpha](alpha-logkv.md)，共享机制与两种路由见 [算法规格](algorithm-spec.md)。

## 精确片段评分

对片段 `s`、KV group `g`，沿用 Alpha 的片段均值 `μ_K`、能量 `E_K` 与归一化方差
`R_K、R_V`。以本次 flush 开始前的有效旧簇中心 `c` 计算：

```text
R(s,g) = (R_K(s,g) + R_V(s,g)) / 2
N(s,g) = min_alive_c mean_features (μ_K(s,g) − c)² / max(E_K(s,g), 1e−12)
q(s,g) = R(s,g) + N(s,g) / (1 + N(s,g))
q(s)   = mean Top-min(2,G)_groups q(s,g)
```

无有效旧簇时 `N=0`。旧精确片段与本批新片段用同一套中心重新评分，原有 `1.1` 倍旧片段
保留偏好不变。Top-2 减少少数 KV group 上的信号被全组平均稀释；新颖度让片段内方差为零
的孤立 token 也可能获得非零分。它衡量与已有压缩内容的差异，不保证事实重要性或问题相关性，
随机噪声也可能得高分。没有针对 UUID 的正则或评测标签。

实现把不等长片段批量归约，再与 `[batch,group,cluster,dim]` 的中心做矩阵乘法，
不为每个片段复制所有中心；沿用每次 flush 一次分数回传，不逐片段等待 GPU。
评分索引统一上传，复用 key 均方范数；整段替换仅在所选片段改变时重排淘汰顺序。
CUDA 使用一个 kernel 把所选 token 与归档 token 分别写入精确池和归档缓冲，合并 K/V、
位置搬运及 padding 补零；数值直接复制，不改变选择结果。

## 簇内四进二压缩

仅在非顶层批量进位时，同一 `(batch,group,cluster,level)` 内每连续四个待合并 entry
作为一组，比较三个连续切分 `2|2、1|3、3|1`。每个 entry 携带真实质量 `w_i`，不得假定
同层质量相等。对 `X∈{K_raw,V}`，四条共用能量分母：

```text
E_X = Σ_i w_i ||X_i||² / max(Σ_i w_i, 1e−12)
μ_X(A) = Σ_{i∈A} w_i X_i / max(Σ_{i∈A} w_i, 1e−12)
C(t) = Σ_X Σ_{A∈{[0,t),[t,4)}} Σ_{i∈A} w_i ||X_i−μ_X(A)||² / max(E_X,1e−12)
```

选择最小代价，同代价优先 `2|2`，然后 `1|3`、`3|1`。输出仍是两个加权 entry，保持质量、
位置范围和 `sum_wp`；不跨 lane 合并。每条 lane 的两条尾部及饱和顶层仍按原二合一处理。
既有 entry 无法拆回原 token；局部低失真也不保证未来所有 query 的注意力误差更小。

CUDA 一阶路径用融合 kernel 并行处理四元组，并在同一 launch 读取存活条目；后续统一写回，
避免覆盖尚未读出的源槽。代价计算使用 FP32。CPU/不支持的布局复用 Torch 参考路径。
切分布局与当前层原有索引共用一次上传，左右输出复用读取的元数据。各层调度对全部 lane
整体计算，回放也走同一路径。unified 路由约每 4 轮检查一次结束标志与存活数，行扫描只遍历
存活节点，GPU 仍逐轮更新中心；此调度优化不等同于 `merge_passes=4` 近似配对。
切分会增加少量运算，实际开销需要目标 GPU 测量，不能据此宣称 step 更快。

## 预算与回放

- 两个开关的 Cache API 默认值为 `false`；训练/评测入口和 `arc_semantic_fast.yaml` 默认 `true`。
  关闭两项即恢复 Alpha 行为；`log_kv_alpha_exact_tokens: 0` 会同时关闭两项。
- 不增加持久 KV 张量，不改变有效 `B′` 的计算；精确池、未闭合尾段和复用打包缓冲仍按
  Alpha 口径计入预算，持久推理 KV 保持 `O(log N)`。
- 推理只需要当前批的评分、切分和临时工作区。训练额外记录每组一个 `uint8` 切分值，
  与已有写入记录一起释放；训练总内存本来就不受推理 KV 的 `O(log N)` 约束。
- checkpoint 重算与 backward 复用前向的精确片段选择和切分；没有重新打分、路由或选切分。
  仍要求 Alpha 的路由（attach 或 unified）、mid、一阶、关闭 segment padding，
  训练必须开启 update replay。

## 最小运行入口

[默认训练配置](../exp/qwen1.7b-32k/arc_attach_alpha_beta_k12_b128_2k.yaml)（attach + Alpha + Beta）
从 Base 初始化，使用独立输出目录，训练超参与 `arc_semantic_attach_route_k12_b128_2k` 基线一致；
`_4k` 版本仅把 recent 改为 4096。该目录已有 checkpoint 时 `auto_resume` 恢复它。
`arc_beta_cpt100.yaml` 保留早先 unified + Beta 的 100 步实验。训练与评测均沿用两个开关：

```yaml
log_kv_beta_novelty: true
log_kv_beta_adaptive_merge: true
```

先在训练环境做短 CUDA 检查，覆盖融合更新、质量/位置、精确池和回放：

```bash
python -m pytest tests/test_beta_log_kv.py tests/test_alpha_log_kv.py tests/test_log_kv_orphans.py \
  tests/test_log_kv_route_schedule.py -q
```

再使用相同输入规模测一次短路由，确认额外开销可接受：

```bash
python unused/benchmark_log_kv_unified.py --device cuda --batch 4 --groups 8 --B 128 --iters 3 \
  --route attach --alpha-exact-tokens 256 --alpha-span-max-tokens 64 \
  --beta-novelty --beta-adaptive-merge --profile-dir route_operator_profile_beta > route_beta.jsonl
```

profile 另标注 `exact_select`、`exact_partition`、`ladder_merge_scatter`，用于区分评分、
搬运和压缩更新。已有不含 Alpha/Beta 的旧 profile 不能证明这些路径的开销。

只训练，不触发脚本默认的全套后续评测：

```bash
BENCHMARKS=none NIAH_BENCHMARKS=none \
  bash majob.sh exp/qwen1.7b-32k/arc_attach_alpha_beta_k12_b128_2k.yaml
```

训练后仅评测 32K single2/3：

```bash
DIAG_ARGS='--metadata {"pretrained":"/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/","max_seq_lengths":[32768]}' \
  bash eval.sh exp/qwen1.7b-32k/arc_attach_alpha_beta_k12_b128_2k.yaml niah_single_2,niah_single_3 none
```

`eval.sh` 从 `save_path` 加载训练产物；第三参数 `none` 关闭默认追加的多长度 NIAH。
显式 metadata 是必需的，脚本目前不转发 YAML 中这一字段。
多卡评测各 rank 汇合时每 `LOGKV_EVAL_SYNC_HEARTBEAT_S`（默认 300s）打印仍未到达的 rank，
超过 `LOGKV_EVAL_SYNC_TIMEOUT_S`（默认 7200s）报错；长上下文生成负载差距更大时调高后者。

实现入口：[评分](../litgpt/alpha_log_kv.py)、[压缩参考](../litgpt/beta_log_kv.py)、
[CUDA 更新](../litgpt/log_kv_updates_triton.py)、[缓存与回放](../litgpt/log_kv_cache.py)。
本地 CPU 检查不能确认 CUDA 编译或 A800 性能；NIAH/LongBench 收益尚待实际评测。
