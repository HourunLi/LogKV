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

Alpha 配置启用 `log_kv_semantic_merge_passes: 4`：全局合并每轮先按冻结的中心
连续做四次互为最近邻扫描，已匹配端点退出后续扫描，最后按代价统一排序、截断到预算
并一次更新中心。这会改变原配对结果：新合并中心到下一轮才参与竞争。候选簇半径约束
不变，有限 hard cap 仍走原配对。设为 `1` 恢复原规则；训练、评测共用此参数，
checkpoint/反向回放不重新配对。精确池收集直接写入已有缓冲区，减少三份暂存及复制。
（第二版起由增量轮次实现，规则相同，见下节。）

本地 CPU 合成验证（batch=1、groups=1、2048×128、B=128、精确池256、两次flush）：
passes=1→4，全局合并轮数95→26，单次flush中位耗时0.512→0.408秒。
这是CPU数据，不能外推A800速度或NIAH质量；CUDA实现尚待目标机器验证。

第一版使用当前 fast 配置：unified、mid、一阶、无 segment gap/padding；训练要求
`semantic_replay_updates=true`。默认关闭时保留原路径。没有引入额外注意力打分、
新的模型参数或外部依赖。尚未验证 NIAH 收益或 A800 额外开销。

## 第二版：速度优化（算法规则不变）

1. **合并 `semanticLogKV` 的增量轮次**（第六轮及对称修复）：每个 tile 只做一次
   FP32 GEMM 得到距离矩阵，之后用 Lance–Williams 公式只更新被合并簇的行/列；每行
   最近邻常驻显存，只有“上一轮最近邻被合并”的行重扫；形状固定不压缩，主机每轮只
   异步读回每组一个活跃标志。该实现曾在 A800 上把 semanticLogKV 的路由从
   0.617 s/flush 降到 0.0728 s/flush（B=256、无精确池，不是 Alpha 的实测值）。
2. **冻结中心多次扫描并入增量轮次**：第一次扫描读常驻最近邻；之后的扫描只让
   “受限最近邻已被匹配”的未匹配行在未匹配列上重扫（Triton `_scan` 模式 2），再由
   `_select_more` 在 GPU 上接受新的互为最近邻对；真实最近邻表不受影响，合并后照常
   维护。与原 `matching_sweeps` 同一规则：不同轮之间仍无主机同步。
3. **两个热点 kernel**：`_scan` 改为 8 行 × 256 列的二维 tile，脏行标志向量化读取，
   干净 tile 直接退出；`_lance_williams` 先合并读取目标行当前键，只在更小时才做
   64 位 atomic min（大多数行最近邻不变，原来每个存活列都做一次原子操作）。
4. **`_alpha_commit_joins`**：去掉逐 token Python（`min` 生成器、`(b,g,i)` 元组）和
   `.cpu()` 位置回传；延迟簇的旧条目与新 token 在 GPU 上按 (簇, 位置) 一次稳定排序
   （旧条目优先，与原逐簇稳定排序一致），每个字段一次 gather + 两次 scatter 直接写到
   排序后位置，少一份整簇拷贝。新 `p_hi` 用“旧 p_hi 与最新新 token 取大”：簇内条目
   位置都不超过该簇 p_hi，所以与原先读回旧位置的结果相同。
5. **`select_spans`**：片段切分由逐 token 循环改为按段（边界/位置断点）处理，长段按
   64 切块并计入跨 flush 尾段，输出列表、闭合标记与原扫描一致；打分与贪心不变。
   每层每次 flush 都会调用，原先是 32K 训练前向里逐 token 的 Python 热点。
6. **`_semantic_route_unified`** 支持每个 batch 行不同的归档 token 数（`token_counts`），
   连续 lane 直接切片，否则一次 gather；padding 行质量为 0，永不参与配对。

精度与语义：配对规则、预算、紧致度复核、K、精确池规则均不变；与原全量重算相比仅有
float32 舍入顺序不同，近似平局时可能选到另一对（随机输入下逐对一致）。

验证：本次修改只做了静态检查（`python -m py_compile` 与逐行推演），没有在本机运行
测试或基准。请在训练环境运行：

```bash
python -m pytest --noconftest -q tests/test_alpha_log_kv.py tests/test_log_kv_pack.py tests/test_log_kv_unified.py
```

`tests/test_log_kv_unified.py` 新增增量冻结扫描与全量重算冻结扫描逐对一致的检查；
基准预检 `check_incremental_reduce` 额外覆盖 passes=4 的融合核与 Torch 参照逐位一致。

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
