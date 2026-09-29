# AlphaLogKV 第一版

`AlphaLogKV` 从 `semanticLogKV` 分出。新增固定精确池，默认关闭；训练配置
`exp/qwen1.7b-32k/arc_alpha_cpt100.yaml` 开启每层 256 token、单片段最多 64 token。
按用户最终决定从 Qwen Base 做 100 步 CPT，输出目录与旧实验分开。

## 保留规则

- 分词器在启动时建立自然结束符表，模型每次输入计算一次边界、各层共用。
  句末标点、分号、换行和 EOS 是边界；冒号、UUID 连字符不是边界。
  这是 token 末尾标点启发式，不是句法分析器。超过 64 token 允许切开。
- 每层独立选择，层内 KV groups 共用片段位置。跨 flush 的未结束片段继续拼接，
  其 token 也计入 256 预算；所以已完成片段的可用预算会随尾段长度变化。
- 批量计算片段内 K/V 的相对方差，两个分量相加、KV groups 取平均：
  `density = mean_groups(var(K)/mean(K²) + var(V)/mean(V²))`。
  这是每 token 的压缩风险代理，不保证“语义重要”，也没有利用题目/答案或 UUID 正则。
  计算均值仅用于打分，精确池中仍保存每个 token 原始 K/V。
- 旧片段打分乘 1.1 以减少抖动；新片段按密度排序，替换时必须覆盖所有被淘汰
  整段的总分 `density × token 数`。这是有界贪心替换，不是最优背包算法。
- 每个已提交 token 只归属 recent、精确池、压缩层之一。新淘汰与旧淘汰 token
  一起执行一次统一路由；旧位置插入时，仅重排受影响簇的现存条目，再进行层级合并。
  已压缩条目不会被逆向拆开，精确片段被淘汰后不再恢复原文。
- 精确 K 使用原始位置 RoPE、质量权重 1，与压缩槽进入同一次注意力。

## 显存与速度约束

新增存储从原 `B` 对应的压缩层预算扣除，自动选择更小的实际 `B`；日志打印
`effective_B`。计算同时计入 K/V、GPU 位置/有效标志、层级元数据和最大 decode
打包工作区。recent、flush、train/prefill block 仍为 2048，推理缓存仍是 O(log N)。
此处约束缓存容量，不承诺含路由临时张量的每次 CUDA 峰值完全相同。

打分为线性批量 GPU 计算，每次 flush 一次分数回传；整段预算决策在 CPU 上完成，
没有逐 token 的 GPU 同步。精确槽已并入现有 Triton pack kernel；训练记录精确池
及归档写入，checkpoint 重算和反向回放均不重选、不重路由。
`alpha_select` 计时包含在 `route` 中，不能再次累加。

Alpha 配置启用 `log_kv_semantic_merge_passes: 4`：全局合并每次计算距离后，
在 GPU 上连续执行四次互为最近邻扫描，已匹配端点退出后续扫描，最后一次性排序、
回传并更新中心。扫描之间复用距离及工作区，没有 CPU 同步；后续扫描只读取尚未
匹配的距离子矩阵。这会改变原配对结果：新合并中心到下一轮才参与竞争。
候选簇半径约束不变，有限 hard cap 仍走原配对。设为 `1` 恢复原规则；
训练、评测共用此参数，checkpoint/反向回放不重新配对。
精确池收集直接写入已有缓冲区，减少三份暂存及复制。

本地 CPU 合成验证（batch=1、groups=1、2048×128、B=128、精确池256、两次flush）：
passes=1→4，全局合并轮数95→26，单次flush中位耗时0.512→0.408秒。
这是CPU数据，不能外推A800速度或NIAH质量；CUDA实现尚待目标机器验证。

第一版使用当前 fast 配置：unified、mid、一阶、无 segment gap/padding；训练要求
`semantic_replay_updates=true`。默认关闭时保留原路径。没有引入额外注意力打分、
新的模型参数或外部依赖。尚未验证 NIAH 收益或 A800 额外开销。

## 训练环境的最小验证

先验证 CUDA 实现与回放，再测短路由；CPU 本地检查不能替代 CUDA 检查。

```bash
python -m pytest --noconftest -q tests/test_alpha_log_kv.py tests/test_log_kv_pack.py
for passes in 1 4; do
  python unused/benchmark_log_kv_unified.py --device cuda --batch 4 --groups 8 --B 128 --iters 3 \
    --alpha-exact-tokens 256 --alpha-span-max-tokens 64 --merge-passes "$passes" \
    --profile-dir "route_operator_profile_alpha_p${passes}" \
    > "route_alpha_p${passes}.jsonl" || break
done
```

两档使用相同输入和预算，基准会先校验当前配对模式的 Triton/Torch 一致性及质量守恒。
`reference_pairs_verified` 指同一模式的实现一致，不表示与旧配对相同。
不要把 B=128 的结果直接与之前 B=256 的结果比较。合成路由不代表检索质量。

通过后启动新实验：

```bash
bash majob.sh exp/qwen1.7b-32k/arc_alpha_cpt100.yaml
```

训练完仅评测 32K single2/3：

```bash
bash eval.sh exp/qwen1.7b-32k/eval_alpha.yaml
```
