# AlphaLogKV：有限预算的精确片段

Alpha 在 SemanticLogKV 的 recent window 与压缩层之间增加可替换的精确池。
默认关闭；[训练配置](../exp/qwen1.7b-32k/arc_alpha_cpt100.yaml) 开启每层每条样本
`P=256` 个精确 token、片段上限 `64`。同层 KV groups 共用所选位置，各层独立选择。
压缩槽的表示与注意力见 [算法规格](algorithm-spec.md)，归档后的分簇与合并见
[统一路由](semantic-unified-routing.md)；本文只定义 Alpha 增量。

## 片段与分数

分词器启动时解码词表，按 token 末尾的句号、问号、叹号、分号、换行和 EOS 建立边界表；
尾部空格、引号与部分右括号会被忽略，冒号和连字符不是边界。模型每次输入只生成一次
边界标记，各层复用。这是便宜的标点启发式，不能保证句法或 UUID 完整。

每次 flush 把旧精确片段与新 token 一起考虑。未结束尾段跨 flush 拼接，遇自然边界、
64-token 上限或位置不连续便闭合；允许在长度上限处切开。仍未结束的尾段强制保留，
**其长度计入同一个 P 预算**，没有额外的尾段 KV 缓冲。

对片段 `s`、KV group `g`、表示 `X∈{K_raw,V}`，令 `l` 为片段长度、`d_X` 为维度：

```text
μ_X(s,g) = mean_tokens X
E_X(s,g) = mean_tokens,features X²
R_X(s,g) = max(E_X − mean_features μ_X², 0) / max(E_X, 1e−12)
q(s)     = mean_groups [R_K(s,g) + R_V(s,g)]
density(s) = 1.1 × q(s)  （原精确池中已闭合的片段）
             q(s)      （新片段，包括刚拼完整的旧尾段）
```

K、V 两项权重均为 1。打分使用 FP32 批量归约；每次 flush 一次分数回传，CPU 完成
整段预算决策。这是**丢弃片段内差异的风险代理**，不是语义重要性的保证；没有问题、
答案标签或 UUID 正则，单 token 的方差分数为 0。

先为强制尾段腾出空间，再按密度从高到低尝试新片段。空间不足时，按密度从低到高
收集已选闭合片段，直到足够；仅当新片段总分 `density×l` **严格大于被淘汰整段总分之和**
才替换。保留与淘汰都以整段为单位，预算按实际 token 数计算。这是有界贪心，不是最优背包。

## 归档与注意力

- 精确池保留各 token 的原始 K/V 和位置，均值只用于打分。读出按原始位置施加 RoPE，
  质量权重为 1，与压缩槽共同参加一次注意力；不需要额外检索注意力。
- 已提交 token 在 recent、精确池、压缩层之间仅有一个语义归属，不添加重复副本。
  本次未入选的新 token 与被替换的旧片段一起执行统一路由。
- 延迟归档可能早于簇内现存位置：只对受影响簇按位置重排条目，然后重新入层级。
  已压缩条目不能拆回原 token；片段被淘汰后没有恢复原始精确表示的通道。
- padding 不参与归档、质量计数或注意力。精确池收集直接写入预分配缓冲，打包复用
  现有 Triton kernel。flush 内的 `alpha_select` 已包含在 `route` 计时中，不要重复相加；
  flush 之后的打分预取也计入 `alpha_select`，但在 `route` 之外。

默认 `log_kv_semantic_merge_passes: 1` 使用新增量路由；设为 `4` 在增量轮次内加入
冻结中心扫描，作为近似配对对照。Alpha 不另建配对算法，具体路径与有限 hard cap 的
回退规则见统一路由文档。

## 同预算与复杂度

精确池从压缩层预算扣款，**不缩短 recent window**。令 `G` 为 KV groups、`K` 为簇上限、
`N` 为配置最大长度、`e` 为 K/V 元素字节数、`d_p=8⌈max(d_k+1,d_v)/8⌉`。
当前一阶 mid 实现按每层、每条样本计算持久缓存与最大可复用打包缓冲的容量：

```text
L(B) = max(2, ceil(log2(N/(K×B) + 1)) + 2)
C_entry = (d_k + d_v + 2d_p)×e + 41
C_ladder(B) = G×K×L(B)×(B×C_entry + 2)
C_exact = P×[G×(d_k + d_v + 2d_p)×e + 9]
```

`41` 是每个层级条目的元数据字节数，`2` 是每层计数；精确位置和有效标记共 `9` 字节。
实现从原 `B−1` 向下找最大的 `B'≥2`，满足
`C_ladder(B')+C_exact≤C_ladder(B)`；找不到则报错。层数随 `B'` 重新计算，日志显示
`effective_B`。batch 容量乘 batch size，其余未改变的缓冲不计入这笔增量比较。

固定 `P、G、K、B` 时，推理 KV 缓存仍为 `O(log N)`。这不代表总显存或 CUDA 峰值：
输入、RoPE 表、临时距离矩阵、训练激活和回放历史不在上述 KV 容量约束内。

## 速度优化（规则不变）

以下改动只改变实现方式，不改变选择、配对、预算和归档规则：

- `select_spans`：片段切分按边界/位置断点逐段处理，长段按上限切块并计入跨 flush 尾段，
  不再逐 token 循环；打分和贪心不变。
- `_alpha_commit_joins`：去掉逐 token Python 与位置 `.cpu()` 回传。所有延迟簇的旧条目与
  新 token 一次上传，在 GPU 上按 (簇, 位置) 稳定排序（旧条目优先，等价于原逐簇稳定排序），
  每个字段一次 gather + scatter 写到排序后位置。新 `p_hi` 取旧 `p_hi` 与最新新 token
  的较大值：簇内条目都不晚于该簇 `p_hi`，结果与读回旧位置相同。
- `_semantic_route_unified` 跳过无归档 token 的 batch 行；连续 lane 直接切片，否则一次
  gather；padding 行质量为 0，不参与配对。
- `merge_passes>1` 的冻结中心扫描并入增量轮次（见统一路由文档），不再回退全量重算。
- Triton `_scan` 改为 8 行 × 256 列二维 tile，干净 tile 直接退出；`_lance_williams`
  先读目标行当前键，只在更小时才做 64 位 atomic min。
- 每轮路由整轮 CUDA graph 重放、`_plan` 核内排序、距离矩阵单趟融合尾部、层级追加的
  向量化主机规划与异步锁页上传，见统一路由文档；轨迹与原实现逐位一致。
- 可选：`LOGKV_ROUTE_TILE_MB=1024` 让整个 flush 的 32 组合成一个路由 tile（更快但多占
  约 0.5–1 GiB 峰值；默认 256 MiB，见统一路由文档）。
- 打分预取：一次 flush 结束后，下一次 flush 的精确池与待归档块已经确定，立即切分片段、
  打分并把分数异步拷到锁页内存；下一次选择校验输入一致后直接使用，不再在
  `.cpu()` 处等待其间排队的注意力。输入不一致（或回放、CPU）时按原路径现算。
- 层级顶层溢出：原来逐条目合并，每条 3 次 `.item()` 同步和约 80 个小核；现在对所有
  溢出簇批量做左折叠（每步一次成对合并，每步后按缓存 dtype 取整），再一次写回，
  数值与逐条路径相同。
- 注意力 plan 的槽索引改为整数组 NumPy 并异步上传；训练记录/回放的主机状态快照改用
  嵌套列表切片复制（约为 deepcopy 的 1/20）。

与全量重算相比仅有 float32 舍入顺序差异，近似平局时可能选到另一对。以上改动只做了
静态检查；请在训练环境运行：

```bash
python -m pytest --noconftest -q tests/test_alpha_log_kv.py tests/test_log_kv_pack.py tests/test_log_kv_unified.py
```

`tests/test_log_kv_unified.py` 检查增量冻结扫描与全量重算冻结扫描逐对一致；基准预检
`check_incremental_reduce` 覆盖 passes=4 的融合核与 Torch 参照逐位一致，并比较 graph
重放/核内排序与逐核启动/argsort 两种执行方式；`check_pair_distances` 检查融合距离尾部。

## 训练、推理与最小运行入口

Alpha 要求 unified、`K>1`、mid、一阶、关闭 segment gap/padding；训练还要求
`semantic_replay_updates=true`。训练与推理走相同选择/归档规则；checkpoint 重算与反向
使用记录的精确池状态和写入操作，不重新打分或路由。

当前配置从 **Qwen3-1.7B-Base** 开始 100 步 CPT，独立输出到
`qwen1.7b-32k-alpha-cpt100`；该实验若已有 checkpoint，`auto_resume` 会恢复它。
簇上限 12、原预算 B=128；recent、flush、train/prefill block 均为 2048。

训练环境中，只训练、不触发 `majob.sh` 默认的全套后续评测：

```bash
BENCHMARKS=none NIAH_BENCHMARKS=none \
  bash majob.sh exp/qwen1.7b-32k/arc_alpha_cpt100.yaml
```

训练完成后，仅评测 32K single2/3：

```bash
DIAG_ARGS='--metadata {"pretrained":"/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/","max_seq_lengths":[32768]}' \
  bash eval.sh exp/qwen1.7b-32k/eval_alpha.yaml niah_single_2,niah_single_3 none
```

这里显式传 metadata：当前 `eval.sh` 不转发 YAML metadata，且默认会另跑 single1/2/3
的六个长度档；第三参数 `none` 关闭这次额外评测。脚本从实验 `save_path` 加载训练产物，
不是从 `ckpt_dir` 加载初始 Base 权重。
多卡评测各 rank 汇合时每 `LOGKV_EVAL_SYNC_HEARTBEAT_S`（默认 300s）打印仍未到达的 rank，
超过 `LOGKV_EVAL_SYNC_TIMEOUT_S`（默认 7200s）报错；长上下文生成负载差距更大时调高后者。

如需验证 CUDA 路由性能，只跑以下短基准；与已有结果比较时保持 B、batch 和输入规模一致：

```bash
python unused/benchmark_log_kv_unified.py --device cuda --batch 4 --groups 8 --B 128 --iters 3 \
  --alpha-exact-tokens 256 --alpha-span-max-tokens 64 --merge-passes 1 \
  --profile-dir route_operator_profile_alpha > route_alpha.jsonl
```

合成基准的质量守恒/实现一致检查不能证明检索质量。Alpha 的 NIAH 收益、目标 A800
速度和实际 step 耗时仍需真实运行确认；不沿用旧路由版本的 CPU 数字作为当前性能结论。

实现入口：[选择器](../litgpt/alpha_log_kv.py)、[缓存/回放](../litgpt/log_kv_cache.py)、
[边界表](../litgpt/tokenizer.py)、[相关测试](../tests/test_alpha_log_kv.py)。
