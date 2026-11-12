# SemanticLogKV 论文 Baseline 选型记录

核实日期：2026-10-12。目标是论文主实验，证明当前 SemanticLogKV 核心方法贡献；
**CPT 是正式组成部分，以适配后的方法为主，同时检查未适配表现**。
本页记录选型依据和本轮公开源码核查，不是已完成的复现清单或性能排名。
外部链接指向可变分支；正式实验应锁定 commit、依赖与模型版本。

当前实现以 [共享规格](algorithm-spec.md) 和实际加载的 YAML 为准：attach 路由、簇内层级
聚合、Alpha 精确片段、Beta 新颖度与自适应切分。Qwen3-1.7B-Base、32K 是当前实验起点，
不是所有外部方法已有的支持范围。公开源码存在、模型家族支持、目标 checkpoint 可用、
成功运行、复现论文结果是不同状态，本轮只核实前两类证据，不声称后面三项已完成。

## 选择原则

每个 baseline 应回答一个会改变方法判断的问题，不以数量或论文新旧代替覆盖面：

| 要证明的贡献 | 最直接的对照 | 可以回答什么 |
|---|---|---|
| 保留远程历史值得付出压缩开销 | 同预算 SWA、StreamingLLM | 简单扩大窗口或保留 attention sink 是否已经足够 |
| 语义组织有独立价值 | 预算归一化的非语义 temporal LogKV | 相对只按时间层级压缩，semantic 部分是否有增益 |
| 聚合历史优于直接丢弃 | H2O、CaM，必要时 D2O/ZSMerge | 重要性淘汰、合并回收与我们的持续摘要有何差异 |
| 精确片段保护有价值 | ChunkKV、SepLLM-streaming；必要的内部开关对照 | 局部完整性、分隔符规则能解释多少收益 |
| 收益不只是预算分配 | PyramidKV，备选 AdaKV | 层间/头间预算利用率是否足以解释差异 |
| CPT 后的完整方案有竞争力 | DMS，备选 LESS | 学习淘汰或低秩历史状态是否更值得训练成本 |
| 实际内存收益成立 | 真正的 4-bit KV / KIVI | 相同字节预算下，低精度保留更多历史是否更划算 |

存储操作、在线处理方式和训练需求是三条独立分类轴。树形或聚类索引不等于摘要存储，
稀疏读取不等于删掉完整缓存，固定压缩比例也不自动带来 `O(log N)` 存储。

## 基础对照：必须先统一

| 对照 | 现有入口 | 使用边界 |
|---|---|---|
| Full KV | [eval.py](../eval.py) 的 `log_kv_dense_mode=true` | 完整历史质量参照；不同训练权重下不称严格上界 |
| 全程 SWA | `swa_window_size`；[专用比较脚本](../unused/compare_swa_niah.sh) | prefill/decode 均使用滑窗；`0` 匹配参考 LogKV 持久分配字节 |
| 非语义 temporal LogKV | 当前非语义缓存路径；已有 [旧实验 YAML](../exp/qwen1.7b-32k/arc_stage1_nonsemantic_train.yaml) | 旧 YAML 的 B、recent、二阶和训练设置不能直接作为公平对照；先归一化预算与表示设置 |
| 针对性内部开关 | Alpha/Beta/路由已有开关 | 只有结果会改变解释时才增加，不默认做全部组合 |

当前非语义路径会关闭 Alpha/Beta。关闭精确池会释放预算，因此内部比较也要记录实际
`effective_B`、缓存字节和工作区，不能只把开关变化写成“语义分簇消融”。
同一 SemanticLogKV 专用 CPT checkpoint 上切换缓存，只能作为机制诊断，
不等于不同方法各自适配后的公平竞争，详见 [SWA 对照](swa-niah-comparison.md)。

## 公开实现：选择、淘汰与预算分配

“优先”指核心外部候选；“备选”按结果补充；“暂缓”反映代码或成本限制，不等于研究价值低。
以下源码链接均为本轮核实入口，尚未在本项目运行。

| 方法 | 源码状态与入口 | 类别 / 训练 | 选取理由与优先级 |
|---|---|---|---|
| StreamingLLM | [作者仓库](https://github.com/mit-han-lab/streaming-llm)，有实现与示例 | 免训练；sink＋recent；流式淘汰 | **优先**。最简单的有限缓存外部参照，隔离远程摘要价值 |
| H2O | [作者仓库](https://github.com/FMInference/H2O)，有 mask 模拟及 `h2o_hf/utils_real_drop` | 免训练；累计注意力重要性淘汰 | **优先**。比较过去注意力驱动保留与内容驱动聚合；内存结果须使用真实删除路径 |
| SnapKV | [作者仓库](https://github.com/FasterDecoding/SnapKV)，有 attention patch | 免训练；观察窗口评分的 prompt 压缩 | 备选。经典筛选基线，与 PyramidKV 部分重叠，不必同时优先投入 |
| PyramidKV | [作者项目 KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory)，由 PyramidKV 扩展 | 免训练；层间预算分配＋淘汰 | **优先**。检验预算分配收益；这里的层是网络层，不是时间层级 |
| ChunkKV | [论文](https://arxiv.org/html/2502.00299v5)指向 KVPress；有 [ChunkKVPress](https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/chunkkv_press.py) | 免训练；连续 chunk 选择，保留原始 KV | **优先**。与 Alpha 的片段完整性动机直接相关；当前 wrapper 不自动代表论文全部优化 |
| SepLLM | [作者仓库](https://github.com/HKUDS/SepLLM)，分 TrainingFree、Training、Streaming 等入口 | 分隔符表示与局部保留；含不同训练路线 | **优先 streaming 版本**。比较语言边界规则与语义分簇，不能混用不同分支的结果 |
| AdaKV | [作者仓库](https://github.com/FFY0/AdaKV)，有存储与计算实现 | 免训练；头间自适应预算，通常叠加淘汰器 | 备选。检查头间分配是否解释收益；应写清搭配的评分方法 |
| TreeKV | [论文](https://arxiv.org/abs/2501.04987)提供的[作者仓库](https://github.com/ZiweiHe/TreeKV)目前为空 | 免训练；树结构平滑淘汰，固定缓存预算 | **暂缓复现**。与时间层级组织相关，但不能记为已有可运行源码 |

ChunkKV 当前代码按块聚合重要性并选择整块原始 KV；不是把块平均成一个摘要。
该 wrapper 未体现论文全部跨层索引复用等优化。复现时记录实际组件，不能仅以类名作为
完整算法及加速结果的证据。SepLLM 的 streaming 位置处理也需要独立核对。

## 公开实现：合并与聚类压缩

| 方法 | 源码状态与入口 | 类别 / 训练 | 选取理由与优先级 |
|---|---|---|---|
| CaM | [作者仓库](https://github.com/zyxxmu/cam)，有模型与评测入口 | 免训练；将被淘汰项的 value 信息合并回保留项 | **合并类首选**。回答“合并还是丢弃”，但不是与我们相同的 K/V 均值聚类 |
| D2O | [作者仓库](https://github.com/AIoT-MLSys-Lab/D2O)，有模型代码和 LongBench 入口 | 免训练；淘汰与相似性合并结合 | 备选。更接近内容相关合并，可与 ZSMerge 择一补充 |
| ZSMerge | [作者仓库](https://github.com/SusCom-Lab/ZSMerge)，有实现、测试与评测脚本 | 免训练；预算分配与残差信息合并 | 备选。检查另一种回收被删除信息的方式；Qwen2 支持不等于 Qwen3 已验证 |
| KVMerger | [论文](https://arxiv.org/abs/2407.08454)可获取；本次未找到可核实的作者实现 | 免训练；key 相似性驱动的自适应 KV 合并 | 暂缓首批复现；机制高度相关，保留文献比较 |
| SemantiCache | [论文](https://arxiv.org/abs/2603.14303)可获取；本次未找到可核实的作者实现 | 免训练；语义切块、块内聚类合并及簇大小补偿 | **研究相关性高，复现暂缓**。必须解释与我们语义路由和层级摘要的差异 |

[KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory/blob/main/README.md) 中的
“KVMerger-style” 辅助合并逻辑不足以确认完整复现原论文。若使用，必须注明具体变体，
不能直接借用 KVMerger 的名称和论文结论。同名 SemantiCache API 响应缓存项目也不是这篇
KV 压缩论文的代码。“未找到”是本次核查结论，不是证明作者永远没有开源。

SemantiCache 的切块内聚类与我们的跨 flush 持久簇、簇内时间层级、有限精确池和 CPT
是需要逐项区分的机制；不能只凭都使用“语义”二字宣称相同，也不能因为没有代码而略过
新颖性讨论。自行实现应在论文细节足够明确、且会改变实验决策时再安排，并标注为复现。

## 公开实现：需要训练或适配

| 方法 | 源码状态与入口 | 类别 | 选取理由与优先级 |
|---|---|---|---|
| DMS | [NVIDIA 官方实现](https://github.com/NVIDIA/Model-Optimizer/tree/main/experimental/dms)，有 Qwen3、训练、缓存和评测代码 | 学习型按头延迟淘汰，蒸馏适配 | **训练类首选**。比较学习保留策略与我们的启发式聚合＋CPT |
| DMC | [NVIDIA Megatron-LM dmc 分支](https://github.com/NVIDIA/Megatron-LM/tree/dmc)，有训练与推理路线 | 学习追加或合并的动态压缩 | 机制相关，但旧模型/重训练栈迁移成本较高，后置 |
| LESS | [作者仓库](https://github.com/hdong920/LESS)，有模型与训练脚本 | 稀疏原始 KV＋学习型低秩历史状态 | **第二训练类候选**。对照我们的精确项与摘要共存结构 |
| Activation Beacon | [官方代码](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/Long_LLM/activation_beacon)，有训推入口 | 学习型 beacon 摘要 token | 备选。比较显式聚合与学到的摘要；固定比例保留 beacon 不自动满足对数存储 |
| CCM | [作者 Context-Memory 仓库](https://github.com/snu-mllab/Context-Memory)，有训练、流式推理与适配权重入口 | 学习型递归上下文记忆 | 若主张连续交互与历史更新价值，可选；不优先用于单次 prompt 主表 |
| FastKVzip | [作者仓库](https://github.com/Janghyun1230/FastKVzip)，有 gate 训练及推理代码 | 学习轻量 gate 近似 KVzip 重要性 | 备选。检验便宜的学习型评分是否足够，与 KVzap 择一 |
| KVzap | [论文指定项目](https://github.com/NVIDIA/kvpress)及 [kvzap 训练目录](https://github.com/NVIDIA/kvpress/tree/main/kvzap) | 学习代理重要性模型 | 备选。不能因推理可直接调用就称完全免训练 |
| RestoreKV | [作者仓库](https://github.com/cvsp-lab/RestoreKV)，有训练、评测、checkpoint | 学习 restore token / LoRA 表示后再做预算匹配压缩 | 扩展。比较少量修复表示与我们保留精确片段的效果 |

DMS 当前公开版本冻结基础模型、训练淘汰适配参数；原论文配方更新所有模型权重，二者
必须区分。已有 Qwen3-8B 示例不等于有 Qwen3-1.7B-Base 权重。其当前推理入口以单次
prefill 后生成流程为主，多轮交错流程存在限制，见 [DMS README](https://github.com/NVIDIA/Model-Optimizer/blob/main/experimental/dms/README.md)。

KVPress 的 [DMSPress](https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/dms_press.py)
是阈值淘汰封装，不包含原论文训练得到的淘汰器，不能代替完整 DMS baseline。
FastKVzip、KVzap、RestoreKV 的适配权重也都依赖具体底模；训练数据、教师前向和适配成本
应纳入记录。优先 DMS 的依据是接入确定性，不能推断它比 LESS 或 DMC 质量更强。

## 公开实现：重建评分、量化与检索

| 方法 | 源码状态与入口 | 类别 | 用途与优先级 |
|---|---|---|---|
| KVzip | [作者仓库](https://github.com/snu-mllab/KVzip)，有 Qwen3 路线和评测代码 | 默认免训练；上下文重建评分后淘汰 | 若强调未知后续 query，优先扩展；需计额外前向和预处理峰值 |
| KIVI | [作者仓库](https://github.com/jy-yuan/KIVI)，有 2/4-bit、模型与内存速度评测代码 | 免训练；低比特 KV 量化 | **实际内存对照优先**。固定比特下全历史仍线性增长，但有限长度上可能很有竞争力 |
| Quest | [作者仓库](https://github.com/mit-han-lab/Quest)，有实现和示例 | query 驱动的页面检索、稀疏读取 | 速度/带宽扩展，不作为有限总存储的直接替代 |
| SqueezedAttention | [作者仓库](https://github.com/SqueezeAILab/SqueezedAttention)，有聚类、attention 与评测代码 | 聚类辅助稀疏检索 | 解释“索引原始 KV”与“只保存聚合 KV”的差异；代码存在不等于所有高效 kernel 已集成 |
| LycheeCluster | [论文](https://arxiv.org/abs/2603.08453)可获取；本次未找到可核实的公开实现 | 结构感知切块、层级 KV 索引 | 相关工作；对数检索开销不等于对数存储 |
| DuoAttention | [作者仓库](https://github.com/mit-han-lab/duo-attention)，有头模式优化与推理代码 | 校准/优化检索头与流式头划分 | 扩展；只要固定比例头保留完整历史，总历史存储仍线性增长 |

如果使用 Hugging Face 通用量化缓存，应写明实际后端、位宽、分组和高精度残留区，
未经等价性核对不要命名为 KIVI。`bnb.nf4` 等权重量化不能代替 KV 量化。
少读原始 KV 的方法需要报告完整缓存和索引占用；不能只计本次 query 读取的条目。

## 接入与公平比较

### 模型与实现

- 先保留作者算法含义，再考虑复用框架。KVPress 已列 Qwen3 支持，但主要使用 Transformers，
  不等于可直接注入 LitGPT，也不保证每个 press 的所有执行模式已验证。
- 老仓库的 Llama/Mistral/OPT 路径不能直接视为 Qwen3 支持；检查 Q/K normalization、GQA、
  RoPE 与缓存更新。现成大模型 adapter 不能直接用于 1.7B Base。
- 如使用独立 Transformers runner，先在相同 checkpoint、tokenizer、输入和生成规则下核对
  Full KV 质量。跨框架速度差异不能全部归因于压缩方法；记录 kernel、精度与实现后端。
- [KVPress 默认在 prefill 压缩](https://github.com/NVIDIA/kvpress)，其 decode 压缩是另一条路径。
  不将 dense prefill 后压缩的结果包装为全程有限内存 streaming。

### 两种比较目的

**完整方法比较**：SemanticLogKV 使用 CPT，训练型 baseline 使用明确标注的适配配方；
记录底模、训练数据/token 数、更新参数范围、教师计算、GPU 时间和实际预算。
相同训练 token 数不必然等于相同训练成本，不强行把蒸馏、adapter 和全参数 CPT 叫成同一训练。

**机制诊断**：优先比较同一 Base 上有定义的未适配方法，并使用少量训练匹配的内部对照。
需要判断数据适配收益时加入 dense CPT 对照。不要给未训练的随机淘汰器一个无意义的
“零训练 DMS”结果；也不要只在本方法专用权重上测所有策略就宣称公平。

不默认重训所有候选。先验证最小输入、计数、预算与执行模式，只有结果会影响下一步
选型时再投入短适配或扩大评测。

### 内存与任务

- 持久缓存按真实字节对齐，包含元数据、原始 K 副本、量化参数和必要残留区；Alpha 的内部
  预算还计最大 pack 工作区。另报 prefill/生成峰值，不以有效槽位比替代显存比。
- 固定预算淘汰器可以参加有限内存流式比较，不要求所有 baseline 都采用对数层级结构。
  固定比例压缩和全历史量化应另外标明其容量增长规律。
- 全程因果流式压缩与完整 prompt 处理后再压缩可分别比较；若放在同一张表，必须明确
  prefill 是否完整、何时压缩、缓存是否随生成继续增长。
- 压缩时是否可见问题会影响 SnapKV 等方法的评分。将“context 后才到达未知 query”和
  “已知 query 的 prompt 压缩”分开定义，不能让不同方法获得不同的信息。
- 初步流程检查可复用已有 32K NIAH single2/3；它不能单独支持通用长上下文能力结论。
  之后按论文主张选择已有检索、问答/多跳或摘要任务，不默认全量扫所有 benchmark。
- 记录 prompt 截断、生成长度/EOS、评分模板与 prefill/decode 计时边界；具体字段见
  [评测指标](evaluation-metrics.md)。旧 checkpoint、旧字段和旧路由分数不得当作当前默认结果。

## 推荐推进顺序

| 阶段 | 候选 | 决策依据 |
|---|---|---|
| 首批 | Full KV、同字节 SWA、归一化 temporal LogKV；StreamingLLM、H2O、CaM | 用较少机制覆盖完整历史、局部保留、层级摘要、重要性淘汰和信息合并 |
| 补齐核心主张 | ChunkKV、PyramidKV、SepLLM-streaming；实际 4-bit KV | 补充片段、预算分配、边界规则与内存竞争方案 |
| 训练类第一项 | DMS 当前公开适配器版本 | 有 Qwen3 路线和明确训练入口；先验证目标小模型接入再投入训练 |
| 根据结果补充 | LESS；D2O/ZSMerge 二选一；KVzip；FastKVzip/KVzap 二选一 | 每次增加能回答新问题的候选，避免同类重复 |
| 暂缓首批复现 | TreeKV、KVMerger、SemantiCache、LycheeCluster、DMC | 前四项代码可得性不足；DMC 迁移与训练成本较高 |

下一项最划算的工作是把本地基础对照的模型、任务、预算与测量口径对齐，再接入首批三种
外部方法。SemantiCache 的机制比较应同步用于论文定位，但目前不阻塞可复现路线。
最终主表根据实际接入与评测证据收敛，不要求把全部候选都复现一遍。
