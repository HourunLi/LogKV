# AlphaLogKV

基于 LitGPT 的流式 KV 压缩：新 token 在 flush 后形成语义候选，与旧簇统一合并；
每个簇维护有界层级缓存。Alpha 在同一缓存预算内保留可替换的短片段精确槽，
尝试减少孤立事实被均值压缩的损失。目标是固定簇数与窗口大小下 O(log N) 的推理 KV 存储。

当前配置：K=12、recent/flush=2048、精确池256 token、片段上限64 token、mid anchor、
一阶 attention、增量 Ward 路由。精确池从原 B=128 的预算中扣除，实际 B 由代码计算。
精确槽的选择是启发式，尚不能宣称改善 NIAH single2/3 或达到某个训练速度目标。

## 文档

| 文档 | 内容 |
|---|---|
| [共享算法规格](docs/algorithm-spec.md) | 缓存结构、两阶段路由、层级压缩、训推边界 |
| [Alpha 精确片段算法](docs/alpha-logkv.md) | 分段、打分、替换、预算，以及训练和评测入口 |
| [统一路由实现](docs/semantic-unified-routing.md) | Ward/半径公式、增量更新、并行和数值约束 |
| [位置与注意力](docs/position.md) | pre-RoPE 内容、mid/multi 锚点、质量偏置 |
| [全程 SWA 对照](docs/swa-niah-comparison.md) | 同权重、同持久缓存字节预算的比较口径 |

训练、32K single2/3 评测和短路由检查的命令统一维护在 [Alpha 指南](docs/alpha-logkv.md)。
在训练环境、仓库根目录运行，并核对配置中的模型和输出路径。当前 CPT 从 Base 初始化；
若实验输出目录已有 checkpoint，`auto_resume` 会恢复它。

LitGPT 通用用法见 [tutorials](tutorials/)。运行产生的 profile 和评测文件是实验产物，
其时间、形状和代码版本须一起解读；它们不定义当前算法。
