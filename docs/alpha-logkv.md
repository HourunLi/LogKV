# AlphaLogKV：有限预算的精确片段

本文保留 Alpha 基线定义。当前默认路线（attach + Alpha + Beta）沿用这里的分段、替换和预算，
评分与簇内压缩的两项改动见 [BetaLogKV](beta-logkv.md)。

Alpha 在 SemanticLogKV 的 recent window 与压缩层之间增加可替换的精确池。
训练/评测入口对语义簇默认开启，[默认配置](../exp/qwen1.7b-32k/arc_semantic_fast.yaml) 显式写出
每层每条样本 `P=256` 个精确 token、片段上限 `64`；`log_kv_alpha_exact_tokens: 0` 关闭
（同时关闭 Beta）。同层 KV groups 共用所选位置，各层独立选择。压缩槽的表示与注意力见
[算法规格](algorithm-spec.md)，归档后的分簇见其中的 attach（默认）与
[unified](semantic-unified-routing.md) 路由；本文只定义 Alpha 增量。

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
  本次未入选的新 token 与被替换的旧片段一起按配置的路由归档（默认 attach）；
  两种路由都接受各样本不等长的归档批，padding 不参与分配。
- 延迟归档可能早于簇内现存位置：只对受影响簇按位置重排条目，然后重新入层级。
  被淘汰片段分散到各簇，这条路径很常见；排序在设备上按（簇，位置）稳定排序完成，
  不回读层级位置。簇的 `p_hi` 不小于其现存条目位置，因此新 `p_hi` 只需新 token 位置。
  已压缩条目不能拆回原 token；片段被淘汰后没有恢复原始精确表示的通道。
- padding 不参与归档、质量计数或注意力。精确池收集直接写入预分配缓冲，打包复用
  现有 Triton kernel。`alpha_select` 已包含在 `route` 计时中，不要重复相加。

attach 直接把归档 token 交给冻结中心的直连/orphan 规则，写入时按位置插入延迟 token。
使用 unified 时，`log_kv_semantic_merge_passes: 1` 为增量路由，`4` 保留冻结中心的近似配对
对照。Alpha 不另建配对算法，两种路由的具体规则见算法规格与统一路由文档。

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

## 训练、推理与最小运行入口

Alpha 要求 attach 或 unified 路由（legacy/chunk-tree 会报错）、`K>1`、mid、一阶、
关闭 segment gap/padding；训练还要求 `semantic_replay_updates=true`。训练与推理走相同选择/归档规则；checkpoint 重算与反向
使用记录的精确池状态和写入操作，不重新打分或路由。没有入层级前的 summary 均值压缩。

当前默认路线的训练与评测命令见 [BetaLogKV](beta-logkv.md) 与 [README](../README.md)。
只开 Alpha、不开 Beta 的既有实验是 `arc_alpha_k12_b128*.yaml`、`arc_alpha_k16_b128.yaml`
（unified 路由，YAML 显式关闭 Beta），簇上限、原预算和 recent 见各文件名。

训练环境中，只训练、不触发 `majob.sh` 默认的全套后续评测：

```bash
BENCHMARKS=none NIAH_BENCHMARKS=none \
  bash majob.sh exp/qwen1.7b-32k/arc_alpha_k12_b128.yaml
```

训练完成后，仅评测 32K single2/3：

```bash
DIAG_ARGS='--metadata {"pretrained":"/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/","max_seq_lengths":[32768]}' \
  bash eval.sh exp/qwen1.7b-32k/eval_alpha.yaml niah_single_2,niah_single_3 none
```

这里显式传 metadata：当前 `eval.sh` 不转发 YAML metadata，且默认会另跑 single1/2/3
的六个长度档；第三参数 `none` 关闭这次额外评测。脚本从实验 `save_path` 加载训练产物，
不是从 `ckpt_dir` 加载初始 Base 权重。

如需验证 CUDA 路由性能，只跑以下短基准；与已有结果比较时保持 B、batch 和输入规模一致：

```bash
python unused/benchmark_log_kv_unified.py --device cuda --batch 4 --groups 8 --B 128 --iters 3 \
  --route unified --alpha-exact-tokens 256 --alpha-span-max-tokens 64 --merge-passes 1 \
  --profile-dir route_operator_profile_alpha > route_alpha.jsonl
```

合成基准的质量守恒/实现一致检查不能证明检索质量。Alpha 的 NIAH 收益、目标 A800
速度和实际 step 耗时仍需真实运行确认；不沿用旧路由版本的 CPU 数字作为当前性能结论。

实现入口：[选择器](../litgpt/alpha_log_kv.py)、[缓存/回放](../litgpt/log_kv_cache.py)、
[边界表](../litgpt/tokenizer.py)、[相关测试](../tests/test_alpha_log_kv.py)。
