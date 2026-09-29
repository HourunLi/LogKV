# 整批 flush 候选分簇与新旧簇统一合并

开启 `log_kv_semantic_unified_route: true` 后，每次 flush 的全部新 token 先独立形成候选，
再与旧簇统一合并到 `log_kv_cluster_k_max` 以内。默认仍为原 fast 路由，已有 checkpoint
无需转换；训练、prefill、decode 共用新路由。

## 本版算法

1. 新 token 从单成员候选开始，按轮合并互为最低 Ward 代价且满足紧凑度约束的配对。
   半径上界采用三角不等式递推，限制为 `sqrt(cluster_lambda_rel * s_h)`，防止相似关系
   链式连接成宽簇。候选数自适应，允许单 token，不提前压到 K。
2. 将全部新候选与全部旧簇放入同一集合，新—新、新—旧、旧—旧统一竞争。
   每轮并行选择互不重叠的相互最近配对；不足以合并到预算时更新中心、权重和代价，继续下一轮。
   每轮最多合并 `当前簇数-K` 对，避免超过需要的合并数。代价相同使用确定的对称规则破平局。
3. 保留规划产生的旧簇合并顺序，先执行旧簇结构变更，再按每簇真实位置顺序批量写入新 token。
   候选只是统计量与本批索引，不预建额外 KV 层级；没有入层级前的 KV 平均。
4. 操作日志和更新回放复用同一实际写入序列；checkpoint 重算与反向传播不重新分簇。

Ward 使用 `n_a*n_b/(n_a+n_b) * ||mu_a-mu_b||²`，是 K 空间平方误差的代理指标，
不等于真实注意力或检索损失。并行配对也不保证与逐次全局最小 Ward 相同。
规划阶段不插入新 token，因此规划中心不能解释成实际层级每次中间压缩的中心。

持久缓存仍为固定 K 套层级，注册 buffer 的布局和字节数与原路由相同，保持 O(log N)。
临时距离矩阵为 O((F+K)²)，F 为固定的 flush 大小；不创建 `[F,F,head_dim]` 张量。
每个 tile 的 batch/KV group 数按 FP32 距离矩阵预算均衡选取（CUDA+Triton 256 MiB，
其余 64 MiB；F=2048 时仍为 16/4 组）。自第六轮起，距离矩阵只在 tile 开始时算一次，
之后逐轮增量更新，主机每轮只异步读回每组一个活跃标志（见第六轮）。
Python 簇成员关系在合并结束后统一重建，旧簇物理重建仍按原顺序执行。
生产路由的中心始终保持整块 Tensor，候选到全局合并之间不再逐 token 拆分中心并重建
Tensor 对象；无合并的组直接复用候选和中心。逐节点中心仅由兼容接口按需物化。
CUDA 融合核只读取一份 Gram 矩阵，在行内计算距离、Ward 代价、半径/容量筛选及 XOR
最近邻，不再物化多份两两矩阵。2048 个候选、16 组的一份 FP32 Gram 约 256 MiB；
这不是总峰值保证。持久 KV buffer 布局不变；GPU 总峰值和路由耗时需要实测。
CPT 更新回放还会保存旧簇合并的写入块，
训练峰值不能由推理缓存预算推断。

大候选集的平方距离改用 FP32 矩阵乘法，小集合保留直接距离计算。关闭路由距离的
autocast/TF32 并恢复调用方设置；先减去公共原点，入选候选配对再直接复核半径，避免
数值相消造成错误的宽簇。距离计算的舍入顺序变化可能改变接近相等的配对，不承诺与旧版
逐位一致。候选不提前限制到 K、按轮合并互不重叠配对、最终 K 预算及实际 KV 写入规则不变。

## 路由性能检查

训练日志中的 `logKV_host route 2074s/430` 是 430 次调用累计的 CPU 墙钟时间，平均
约 4.82s/次，包含 GPU 等待。不能将它与 `attn_fwd` 的异步提交耗时直接比较为 GPU
计算占比。CUDA event 计时也包含流等待，不等同于单个 kernel 的纯计算时间。

优化前，CPU 合成测试（2048 token、128 维、单组随机 K）中 48 轮距离计算占约 87%：
原代码每轮用显式禁用矩阵乘法的 `cdist` 重算全部距离，再逐组回传配对。
这是已复现的瓶颈；真实训练数据下旧簇重建的占比仍需单独观察。
同输入 CPU 对比（2048 token、128 维、4 组，预热一次、三次测量取中位数），
单次 flush 从约 13.11s 降至 1.99s；该结果不能外推为 GPU 或整步训练加速比。

第一轮优化后的真实训练仍为 route 1122.945s/420、整步 1212s，平均 route 2.67s。
A800 合成输入（batch=4、groups=8、flush=2048）的 route 为 2.278s/flush、额外峰值
396 MiB；独立插桩中统一规划占已统计阶段约 75%，每个四组批次仍需 49–55 轮。
第二轮优化因此融合每轮最近邻计算并扩大组批量，保留每轮互近邻、不重叠配对的规则；
不减少算法所需的轮数，也不把候选提前压到 K。第二轮的 CUDA 加速与峰值尚待实测。

在训练环境先运行这个短基准，不加载模型、不启动 CPT：

```bash
python unused/benchmark_log_kv_unified.py --device cuda --batch 4 --groups 8 --iters 2 > route_profile.jsonl
```

CUDA/Triton 下脚本先对少量输入核对融合路径与 Torch 参照的配对一致性，覆盖平局、
无有效配对、padding、半径筛选及整组容量回退；失败即报错，不开始计时。首次编译发生在
预检查和预热阶段，不计入最终中位数。第一行的 `route_backend` 应为 `triton`，
`reference_pairs_verified` 应为 `true`；无 Triton 时明确报告 `torch` 并使用原参照路径。

第一行报告同步完成后的总 route 时间/flush 和额外峰值显存；第二行单独插桩，报告
候选建簇、统一合并规划、旧 KV 合并、新簇写入、批量写入耗时及每次搜索提出的配对数
（已经过精确半径复核，尚未经过最终预算截断）。
阶段插桩会增加同步，不能拿它替代第一行总耗时。默认随机输入和两次 flush；这是性能诊断，
不代表真实 NIAH 质量、整步训练吞吐或完整训练峰值。现有新路由 YAML 无需加新开关。

第二轮 A800 实测为 2.110s/flush、330 MiB，配对检查通过；相比上一轮耗时仅下降
7.4%。下一步使用算子 profiler 区分距离计算、GPU 启动/等待与 Python 调度开销：

```bash
python unused/benchmark_log_kv_unified.py --device cuda --batch 4 --groups 8 --iters 1 \
  --profile-dir route_operator_profile > route_operator_profile.jsonl
```

该模式替代原有阶段计时，先保留未插桩基准，然后分别采集一次 Torch CPU/CUDA profiler
和一次 cProfile。每次仅记录最后一个 flush，前面的缓存构建、reset、编译预热不纳入采集；
采集不额外逐阶段同步。`route_summary.txt` 包含按自身耗时排序的 CPU/CUDA 算子与
Python 调用排名，`route_trace.json` 是带 `logkv/` 阶段标记的 Chrome 时间线，
`route_python.prof` 可供后续离线分析。若未采集到 CUDA device events，摘要明确提示
无法归因到 GPU kernel，不能把空 CUDA 表解释成 GPU 没有耗时。profiler 本身有开销，
速度对比仍看未插桩 JSON；嵌套阶段的累计时间不能相加。

第三轮优化由实测 CPU 调用统计驱动：`unbind` 约 1.221s、展平索引列表推导累计
0.388s（包含约 87 万次重复 `size` 查询）。改为整块中心传递、无合并时直接复用，
并将 `mu.size(1)` 移出逐索引循环。配对规则、矩阵运算和 KV 写入顺序不变。
CPU 上随机、全相同、聚簇三种输入连续三次 flush，与修改前全部缓存 buffer 和宿主状态
逐位一致；真实 GPU 加速需复测，不能用上述插桩时间相减推算。

第三轮 A800 实测进一步降至 1.435s/flush、330 MiB。第四轮按剩余热点优化每轮数据流：
配对回传后直接使用 NumPy 数组，批量筛选有效配对、生成存活索引和保存原始节点 ID，
不再逐轮构造嵌套 Python 列表。每轮只上传一份来源/合并对象映射。全局阶段的 CUDA
`merge_pack` 在一个 kernel 中读取旧中心和权重，输出合并且压紧的新缓冲区；未合并行
直接复制，完成组通过前缀切片退出，避免额外索引上传和三次 `index_select`。
候选半径仍用原 Torch 运算顺序，配对规则、组内合并顺序、K 预算和持久 KV 布局不变。
每轮仍有一次回传，算法相依的轮次没有被声称完全并行。

基准现在先检查融合更新与 Torch 的逐位一致性，覆盖不等权重、复制行、padding、非二次幂
维度和输入不可变性；第一行应同时有 `reference_pairs_verified: true` 和
`reference_updates_verified: true`。第四轮 CUDA 性能尚待实测；CPU 三类输入连续三次
flush 与第三轮全部缓存 buffer、宿主状态逐位一致。新 profiler 范围 `logkv/merge_pack`
用于区分更新压紧与配对计算，继续用同一短基准验证，不需要重跑完整训练。

### 第五轮：数组归属和跨组批量写入

第四轮 A800 已测到 0.61708s/flush；同配置真实训练为 route 363.07s/420、
449s/step。150s/step 是下一目标，尚未达到或验证。

本轮生产路由直接处理权重、根编号和合并边数组，不再为每个新 token 和中间簇构建
`_SemanticTreeCluster`。候选归属通过并行指针跳转还原；旧簇之间的合并顺序仍按
原始轮次和代价顺序提取。节点接口保留给诊断和参考检查。

各 batch/group 的旧 KV 合并按依赖轮次批量执行：一轮每组最多一对，同组顺序不变，
跨组共用排序元数据回传、gather、清空和 ladder append。新簇的首 token 同样批量
初始化；后续写入和回放按片段生成数组索引，不再遍历所有 token 构建 Python 列表。
整个 unified flush 延迟写回标量镜像和训练 op-log，末尾集中上传。推理持久 KV
预算、候选紧致度判据、互为最近邻规则及 K=12 均未调整；每个配对轮次仍需一次主机回传。

本地 CPU 同输入检查：随机、相同、聚簇三类输入，分段开/关，连续三次 flush 的
缓存及宿主状态与修改前逐位一致；补充旧节点路由对照、op-log、更新回放和梯度检查。
CPU 大样本的 Python 调用数从约 116 万降到 24 万，但总时长仅从 16.26s 降到
15.76s，CPU 距离计算占主导，不能用这个总时长推算 A800 加速比。CUDA 性能和
150s/step 目标需要在训练环境复测。

短基准现增加批量写入与逐组写入的状态预检，失败直接报错；首行应额外出现
`reference_state_verified: true`。用新目录避免与第四轮 profile 混淆：

```bash
python unused/benchmark_log_kv_unified.py --device cuda --batch 4 --groups 8 --iters 1 \
  --profile-dir route_operator_profile_v5 > route_operator_profile_v5.jsonl
```

预检通过后用原配置跑一个真实训练 step，比较 route、replay 和总耗时；不需要重跑下游评测。

### 第六轮：增量 Lance–Williams 轮次、异步终止与设备端旧簇排序

v1.6 算子 profile（最后一次 flush，含 profiler 开销）为 252 ms，而 GPU 自身只忙
57.8 ms：瓶颈在主机侧。`global_merge` 占 72%（182 ms），每轮都要：整块 FP32 Gram
重算（`ampere_sgemm` 98 次、平均 268 µs）、最近邻核、`select_pairs` 的十余个小算子、
一次 `.cpu()` 回传（114 次，约 25 ms 等待）、逐组 NumPy 循环（33 ms self）和
`merge_pack` 映射上传。随机输入全局阶段每 tile 约 55 轮，每轮只合并约 7% 的簇，
所以这些逐轮开销被放大。

本轮在不改变配对规则的前提下改写轮次的求值方式：

1. **增量距离**：每个 tile 只做一次居中 FP32 GEMM，以 `baddbmm_(beta=1)` 直接累加到
   范数和上得到 D²（只分配一个 M×M，少四次整矩阵逐元素遍历）。合并 (a,b)→a′ 后用 Lance–Williams 恒等式
   `D²(k,a′)=α·D²(k,a)+β·D²(k,b)−αβ·D²(a,b)`（α、β 为质量占比）更新 a′ 的行与列；
   同轮两个都被合并的簇之间再套一次该恒等式。逐轮 O(M²·head_dim) 的 GEMM 变为
   O(合并数·M) 的访存。
2. **常驻行最小值**：每行保存打包键 `(cost 位模式 << 32) | (row ^ col)`，一次 int64
   最小值即等价于原 XOR 平局规则。未合并簇之间的代价不变，因此只有“上一轮最近邻被
   合并”的行需要整行重扫；合并后的行把新代价用 atomic min 推给其余行。随机输入早期
   约一半行需要重扫（热点邻居），但重扫只读 D² 行，不含 head_dim 因子。
3. **不压缩**：行号始终是输入节点号，形状整轮固定，不再逐轮回传尺寸或上传映射；
   已合并的行以质量 0 屏蔽。
4. **每轮 4 个 Triton 核 + 1 次 argsort**：`_select`（互为最近邻、精确半径复核、
   被拒配对写入精确距离）、`_plan`（每组预算、质量/存活/trace 提交）、
   `_lance_williams`（中心、行列更新、行最小值与推送）、`_scan`（脏行重扫）。
   主机把每组活跃标志异步拷到 pinned 内存，并在下一轮排队后才读上一轮的标志：
   GPU 不等待 Python，最多多排一个空操作轮。合并结束后一次性读回 trace 和存活行。
5. **旧簇合并排序上设备**：两段稳定排序 + 段内前缀最大值，与原双指针归并等价
   （双指针归并等于按两侧前缀最大值的稳定归并），去掉每一步的 `.cpu().tolist()` 同步和
   Python 排序；3000 组随机用例（含平局、无质量 pad、非单调键）与原顺序逐项一致。
6. **批量写入与 op-log**：无分段（`seg_gap=None`）时整块 NumPy 构建暂存行；op-log 直接以
   NumPy 行记录与上传，不再逐 token 构造 tuple 再转回数组（训练每次 flush 约 B·G·F 行）。
   span→索引改为一次 `repeat/arange`。

语义与精度：

- 配对规则不变：XOR 平局、互为最近邻、不重叠、精确半径复核、每组 `count−K` 预算按代价
  取前缀、旧簇合并顺序按轮次和代价。容量硬上限（默认关闭）需要两套行最小值，仍走原
  全量重算实现 `_semantic_unified_reduce_rounds`。
- 牺牲的精度：LW 递推代替逐轮重算 Gram，舍入顺序不同，float32 近似平局时可能选到
  不同配对。与 float64 全量重算对照（8 个紧簇种子），新旧实现首次偏离 float64 的轮次
  完全相同，说明差异来自 float32 本身的近平局；随机输入下与原实现逐轮逐对一致。
  完全相同或整数坐标的精确平局下，索引不再压缩，XOR 规则可能选出另一组同样合法的配对。
- 候选阶段被精确复核拒绝的配对，会把这对的 D² 改写为精确值并重扫两行，之后这两行
  可以与其它簇配对；原实现会因同一 GEMM 误差反复提议并冻结它们。紧致度约束不变。
- 持久 KV buffer 布局和字节数不变；临时 D² 仍为 `[tile, F+K, F+K]` FP32，另有
  O(tile·F) 的状态数组。`_SEMANTIC_ROUTE_TILE_BYTES` 调大可减少 tile 数和总轮数
  （以显存换速度）。

本地验证（CPU；未在 GPU 上测速）：

- `tests/test_log_kv_unified.py` 34 项通过（CUDA 项跳过），新增增量轮次与全量重算
  逐对一致、设备端旧簇排序、融合核对照三项。
- Triton 核在解释器模式下与 Torch 参照逐位一致：trace、存活行、中心，覆盖随机、紧簇、
  整数平局、全相同、半径约束、GEMM 抵消导致的拒绝和非 2 的幂维度。
- 与修改前代码（git HEAD）同输入连续 6 次 flush（2×3 组、K=5、分段开/关、op-log 开、
  非单调位置）：全部 buffer、宿主镜像和 op-log 逐位一致。

A800 上需要复测。第一行应额外出现 `reference_incremental_verified: true`（融合核与
Torch 参照对照）：

```bash
python unused/benchmark_log_kv_unified.py --device cuda --batch 4 --groups 8 --iters 2 > route_v6.jsonl
python unused/benchmark_log_kv_unified.py --device cuda --batch 4 --groups 8 --iters 1 \
  --profile-dir route_operator_profile_v6 > route_operator_profile_v6.jsonl
```

profile 标签：`logkv/candidates`、`logkv/global_merge`、`logkv/merge_round`（每轮）、
`logkv/pair_matrix`（每 tile 一次的 GEMM）、`old_kv_merge`、`new_cluster_write`、
`batched_write`；`pair_search`/`merge_pack` 只在容量硬上限回退路径出现。诊断模式的
`merge_rounds` 记录每轮各组实际合并的配对数。

### 第六轮补丁：距离矩阵保持严格对称（修复 “no finite merge cost”）

A800 实测第六轮：`route_median_s_per_flush` 0.0728 s（第四轮 0.617 s），
`reference_incremental_verified: true`。但实际运行报错
`RuntimeError: unified routing has no finite merge cost to satisfy the cluster budget`。

原因：同一轮里两个簇都被合并时，它们之间的新距离由两个 Triton 程序各自用二次
Lance–Williams 公式计算，展开顺序相反（p 先 q 后、q 先 p 后），float32 舍入不同，
D 出现 ulp 级不对称。D 对称时，全局最小键的那一对必然互为最近邻；不对称时，近似平局
可能形成 p→q→r→p 的偏好环，没有任何互为最近邻的配对，于是簇数仍大于 K 时报错。
原全量重算实现每轮都从对称的 Gram 出发，没有这个问题。CPU 上用格点加 ulp 级噪声构造
的输入，400 次全部出现不对称（最大绝对差 6.0，距离量级 1e6）。

修复：

1. 两个都被合并的簇之间，固定按“保留下标小的一侧先展开”计算。借助旧 D 的对称性，
   两侧程序读到相同的四个旧距离、按同一运算顺序求值，写入逐位相同的值；其余条目本来
   就由同一程序同时写入行和列。修复后同样的 400 次输入不对称为 0。
2. 初始矩阵显式把上三角镜像到下三角（CUDA 用一个 Triton 核，其余用 Torch 分块复制），
   不再依赖 GEMM 恰好逐位对称。
3. 兜底：全局阶段若仍有组找不到互近邻配对，只把这些组交给原全量重算重跑，并警告一次；
   真的没有有限代价时仍抛出原错误。
4. 顺带修复：生产路由之前绕过了容量硬上限；硬上限非零时现在走全量重算（原行为）。
5. CPU 上 `from_numpy(...).to('cpu')` 与调用方数组共享内存，质量原地更新会改写调用方
   权重；现改为复制。

验证（CPU）：`tests/test_log_kv_unified.py` 38 项通过，新增逐轮对称性、卡住回退、
硬上限路由三项；Triton 核在解释器中与 Torch 参照逐位一致，且两者的 D 始终严格对称；
与修改前代码的缓存状态对照仍逐位一致。基准预检 `check_incremental_reduce` 新增格点
近平局用例，并断言 D 严格对称。

v1.7 profile 读法（下一步）：最后一次 flush 带 profiler 为 109 ms，GPU 自身 50 ms；
轮次循环已转为 GPU 受限（`cudaEventSynchronize` 11 ms 是主机在等 GPU）。GPU 热点是
`_scan`（16.2 ms/120 次）和 `_lance_williams`（16.1 ms/116 次）：前者逐行、逐程序串行
检查脏行，后者对每个存活列都做一次 64 位 atomic min。可先读当前键、只在更小时推送，
并把 `_scan` 改为多行二维 tile。主机侧 `batched_write` 仍有约 15 ms Python。

## 第一轮评测

使用已完成训练的同一个具体 `step_*` checkpoint。新配置
`exp/qwen1.7b-32k/compare_niah_unified.yaml` 继承 semantic 对照，保持
K=12、B=256、recent=2048、逐 token 条目、mid anchor、二阶 scale=0，只更换路由。

在仓库根目录执行，替换 `<已有checkpoint的step目录>`；单机卡数可调整：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 eval.py \
  --config exp/qwen1.7b-32k/compare_niah_unified.yaml \
  --checkpoint_dir '<已有checkpoint的step目录>' \
  --output_path out/niah-unified-32k --limit 4
```

确认运行正常后，去掉 `--limit 4`，更换输出目录跑正式评测。比较时分别看 single1/2/3，
并核对结果 JSON 的 checkpoint、`log_kv_semantic_unified_route`、anchor 与
`cache_budget`。已有 semantic 结果只有在其余配置一致时才能直接复用；否则将上面的配置
替换为 `compare_niah_semantic.yaml`，输出到独立目录补跑对照。SWA 对照可继续独立进行。

## CPT 与配置限制

`demo.py`、`majob.sh` 和 `eval.sh` 已透传新开关。后续 CPT 使用新实验配置，从现有
checkpoint 加载并设置独立的 `save_path/expid`；不要把继承的 base checkpoint 或原实验
保存目录当作续训来源/目的地。本轮先评测，不启动训练。

summary 提前平均及其配置入口已删除。旧 YAML 中的 `log_kv_semantic_summary_size`
需要移除；模型权重无需转换。新路由不能同时启用 legacy 或 chunk-tree。容量软惩罚 beta 必须为 0；
如启用有限时间间隔分段，要求 `seg_forget=1`，确保中心与 Ward 的累计计数一致。
默认 gap=None 无需修改。可选容量上限沿用既有溢出原则：没有任何可行配对时，优先保留所有
token 和簇数预算，允许超过容量上限。段落文本边界、精确槽和 V 感知评分未在本轮加入。

## 验证

```bash
python -m pytest -q tests/test_log_kv_unified.py tests/test_semantic_eval_flags.py
```

覆盖紧凑候选和 singleton、超过 K 的候选、配对互斥与平局、旧簇合并顺序、精确条目、
KV 加权和与位置统计守恒、持久缓存字节数、操作日志/更新回放和 checkpoint 梯度。CUDA 可用时
额外运行 CUDA 配对及 checkpoint 梯度检查。真实 32K 分数、吞吐与峰值显存仍需 GPU 评测。
