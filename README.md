# SemanticLogKV

基于 LitGPT 的流式 KV 压缩：统一路由让本次 flush 的 token 先形成语义候选，
再与已有簇合并；每个簇维护有界层级缓存。压缩存储 RoPE 前的内容均值与位置统计，
读出时施加位置旋转。在固定簇数、层宽与 recent window 下，推理 KV 存储为 O(log N)。

当前统一路由训练入口是
[`arc_semantic_unified_cpt100.yaml`](exp/qwen1.7b-32k/arc_semantic_unified_cpt100.yaml)：
K=12、B=128、recent/flush=2048、mid anchor、一阶 attention、增量 Ward 路由。
新 token 逐个进入层级，没有预先 summary 均值压缩。
`arc_semantic_fast.yaml` 保留原 fast 路由，二者不要混作同一实验。

## 文档

| 文档 | 内容 |
|---|---|
| [算法规格](docs/algorithm-spec.md) | 缓存结构、两阶段路由、层级压缩、训练回放 |
| [统一路由实现](docs/semantic-unified-routing.md) | Ward/半径公式、增量更新、并行与数值约束 |
| [位置与注意力](docs/position.md) | pre-RoPE 内容、mid/multi 锚点、质量偏置 |
| [全程 SWA 对照](docs/swa-niah-comparison.md) | 同权重、同持久缓存字节预算的比较口径 |

## 训练环境入口

在仓库根目录运行。当前 CPT 配置的实际初始权重是 **Qwen3-1.7B-Base**，训练 100 步；
已有本实验 checkpoint 时，`auto_resume` 恢复训练。运行前核对 YAML 中的训练环境路径。
只训练，关闭 `majob.sh` 默认的后续全套评测：

```bash
BENCHMARKS=none NIAH_BENCHMARKS=none \
  bash majob.sh exp/qwen1.7b-32k/arc_semantic_unified_cpt100.yaml
```

完成后，只评测本次训练产物的 32K single2/3：

```bash
DIAG_ARGS='--metadata {"pretrained":"/home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-1.7B-Base/","max_seq_lengths":[32768]}' \
  bash eval.sh exp/qwen1.7b-32k/arc_semantic_unified_cpt100.yaml niah_single_2,niah_single_3 none
```

`eval.sh` 从 `save_path` 解析实际 checkpoint；它不转发 YAML metadata，故此处显式指定
单档长度。第三参数 `none` 关闭默认追加的 single1/2/3 六档评测。
这里沿用训练 YAML；现有 `eval.yaml` 的 B=256，且未开启 unified/mid，不适用于这组对照。

已完成权重的全程 SWA 配对比较使用专门入口；其语义组是原 fast 路由，详见对照文档：

```bash
bash unused/compare_swa_niah.sh '<已完成checkpoint目录>'
```

LitGPT 通用用法见 [tutorials](tutorials/)。profile 和评测输出是实验产物，必须结合
输入形状与代码版本解读；旧测速不能代表当前实现，合成路由基准不能证明检索收益。
