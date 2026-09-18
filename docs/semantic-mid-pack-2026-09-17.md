# SemanticLogKV：单 mid anchor 与融合输入准备

目标环境：A800、CUDA 12.8、PyTorch 2.11。当前 fast YAML 已启用：

```yaml
log_kv_second_order_scale: 0.0
log_kv_semantic_anchor_mode: mid
log_kv_semantic_pack_backend: auto
```

`mid` 将每个有效 entry 物化为一个加权中心位置的 slot，质量偏置为 `log(w)`。
`multi` 保留原来的 lo/mid/hi 去重与 `log(w)-log(M)`。未指定参数的旧配置/API
默认仍为 `multi`，避免已有实验静默改变数学定义；本次使用的 fast 配置明确选 `mid`。
两种模式共用缓存写入与 replay 接口。底层 lo/hi 元数据仍保留供旧模式使用；mid 读出
不读取端点、不调用 anchor 去重、不生成三份候选。不要混用 mid 训练与 multi 评测而
误以为是同一组实验。单 mid 本身是精度换速度，融合打包则以相同 mid 定义为参照。

## 实现

1. mid 计划按已有 CPU count 镜像枚举占用区间，一次上传索引，使用 `sum_wp/w`
   的整数 round-half-up 中心。无全量 flip、sort、`max().item()`。正权重 entry 的
   中心天然位于 lo/hi 内，无需读取端点 clamp。segment 的零权重对齐 pad 仍被 mask；
   不把 pad 计入真实 token。默认 fast 配置没有 segment pad。
2. `log_kv_pack_triton.py` 的 kernel 直接写最终 `[pooled, recent, current]` K/V：
   gather、RoPE、质量偏置、无效位置清零、特征维度补齐在一次写出中完成。
   无中间的 pooled+recent/current `cat` 和整个 K/V 的后续 `pad`。
   Q 仍经原有扩维变换；最终 attention 和其 backward 仍使用 PyTorch Flash SDPA。
   动态 prefix/recent 长度不作为编译期常量，避免每个 chunk 的长度触发一套编译。
3. 打包的自定义 autograd backward 只截取 current K/V 的梯度，符合现有 streaming
   stop-gradient 目标。历史缓存不带梯度；当前 chunk 不经过 detached cache。
   不改变低显存 routing replay，也不关闭 Block checkpoint。
4. 训练、checkpoint 重算、backward replay、prefill、decode 共享
   `log_kv_chunk_attention`。decode 的 pending 单 token 仍保持两 token 提交语义。
5. 无梯度 decode 可复用最终输入 workspace：pooled 未变时不再 gather/RoPE，已有
   recent 也不重写，只写新增 recent/current。flush/ladder append/clear/reset、dtype/
   device 转换或可追踪的 RoPE 变更会失效。训练不持有这些可变 workspace。
   若用户在 inference mode 中原地改写一个没有 version counter 的 RoPE tensor，必须
   reset cache；常规模型接口不会这样更新 RoPE。

`auto` 在支持的 CUDA/Flash 形状下优先 Triton，Triton 未安装时警告一次并回退到 torch
直接打包。`torch` 强制 torch packer，`triton` 在进入快路径时要求可用 Triton。
非 CUDA、非受支持 dtype/形状、二阶非零或 `multi` 均保留原 attention 路径。
Triton 编译/执行异常不会被静默吞掉。K/Q/V 的特征补齐与原 Flash 变换一致，包含
质量偏置坐标；目前并未消除 128→136 的特征维度扩展。

## 验证和目标机命令

```bash
python -m pytest -q tests/test_log_kv_pack.py tests/test_log_kv_flash.py tests/test_log_kv_speed.py tests/test_log_kv_cache.py tests/test_log_kv_position.py tests/test_semantic_eval_flags.py
python unused/benchmark_log_kv_pack.py --device cuda
torchrun --nproc_per_node=8 demo.py --config exp/qwen1.7b-32k/arc_semantic_fast.yaml
```

GPU 测试要求实际 Triton 和 Flash forward/backward，检查 bf16/fp16、部分/完整 RoPE、
非连续输入、空 lane、段间 pad、输出与 dQ/dK/dV、streaming replay 和 decode workspace。
CPU 测试以旧物化路径为独立参照，另外检查完整 GPT 的 Block checkpoint 和奇数 prefill
后的 pending token。`tests/test_semantic_eval_flags.py` 实际执行两个 shell 脚本的参数
提取部分，防止训练使用 mid 而 eval 丢失参数。

初次本地检查：Python 3.9 / torch 2.6，使用临时源码加载器延迟类型注解并绕开不可用的
checkpoint I/O 依赖；284 passed / 16 skipped。其中 CUDA/Triton 测试未执行。
这些 CPU 结果不能替代目标 A800、torch 2.11、多卡 FSDP 的验证。

基准脚本在同一份缓存上对比 `multi_reference`、`mid_reference`、`mid_torch`、
`mid_triton`，先校验相同 mid 定义下的输出/梯度，再预热并测量。报告计划生成、输入
准备、一次 attention forward+backward 的 wall time 和增量峰值显存；同时报告 pooled
和完整 attention 的 padding 比例。该脚本不包含 routing、MLP、LM-head、FSDP 通信，
不能作为 optimizer step 的加速倍数。

本地小规模 CPU smoke（N=8192、chunk=128、G=2、D=16、K=8、B=64）中：
pooled 宽度 5079→2049，完整宽度 5335→2305。它验证了 anchor 数下降，不代表真实
模型的 token 分布或 GPU 性能。

## 本轮边界

仍按最长 lane 补齐。基准脚本已提供浪费比例；没有额外引入分桶/变长 attention 内核，
因为该项需要先用目标模型的实际长度差异判断收益。固定 ladder 容量、LM-head loss
checkpoint 和多卡通信也未改变。减少 anchor 主要降低物化结果和 attention 的开销，
不直接缩小底层固定 ladder buffer。

## 完整训练 step 计时

`demo.py` 每个 optimizer step 默认输出细分 `logKV_host`，汇总该步所有梯度累积的
micro-batch。`route` 为执行路由决策的语义 routing + flush；`replay` 为按 op-log 重建
的 flush（包括下述优化启用后的 Block checkpoint 重算）。`plan` 为 anchor 索引计划；`pack` 为输入物化；`attn_fwd`
包括初次前向、checkpoint 重算及 backward 内部重新求 attention；`attn_bwd` 是当前
chunk 的 attention/打包梯度计算。plan 的计时不再混入 pack，也不会在 replay 中重算。
`route/replay` 只统计 semantic flush，单 ladder 的普通 cache 更新不在这两项内；K=1
无 op-log 的直接写入仍计入 route。旧 fallback 的 attention 输入扩维仍计入 attn_fwd。

另一行 `step_host (inclusive)` 汇总 data、整个模型 forward、整个模型 backward、
optimizer。这些父级范围包含 LogKV 子项，不可把两行相加。forward 包含 loss.item()
带来的等待；backward 包含 checkpoint 重算及 FSDP 等待。剩余时间不能直接解释成
某一个 GPU 算子的耗时。`Time` 采用单调时钟，覆盖完整梯度累积和 optimizer 提交，
排除步间日志/checkpoint 保存；普通 step 不为计时额外同步 CUDA，因此不是严格的
GPU 完成时间，不能直接与此前含 checkpoint 保存的 Time 混用。

默认 `log_kv_profile_steps: null` 不创建 CUDA events。需要采样时，在实际使用的 YAML
设为下面的值，或保留 null 并在启动命令追加 `--log_kv_profile_steps '[3,4,5]'`：

```yaml
log_kv_profile_steps: [3, 4, 5]
```

编号从 1 开始，指日志中的全局 optimizer Step；恢复训练时也使用全局编号。选中 step
额外输出 `logKV_cuda` 和 `step_cuda (inclusive)`，只在 step 结束统一 synchronize，
不逐 chunk/阶段 synchronize。这是当前流上的 CUDA event 时间跨度，可能包含 GPU
空闲、CPU 提交间隙和依赖/通信等待，不是 profiler 的纯 kernel 执行时间；其他 CUDA
流的独立工作也不能由它精确归因。不能将 host 与 CUDA 时间相加。采样会增加 event
记录、内存及同步开销，速度对照应同时保留相邻未采样 step。

所有 rank 都采样，控制台与新增 TensorBoard 指标只记录 rank 0，不做跨 rank 求和。
TensorBoard 使用 `train/logkv_host_*_s`、`train/logkv_cuda_*_s`、
`train/step_host_*_s`、`train/step_cuda_*_s`，并记录调用次数及 `logkv_cuda_profiled`。
启用 TensorBoard 的方式沿用现有训练配置。首次 CUDA 编译可能污染首步，建议先看预热
后的连续 step。测试入口为 `python -m pytest -q tests/test_log_kv_timing.py`。

## Block checkpoint 复用首次 routing 记录

`demo.py` 在 Fabric 完成模块包装后，为现有的 Block checkpoint 安装
`log_kv_checkpoint.py` 中的 `context_fn`。在 FSDP、activation checkpoint 开启、
semantic K>1 的训练配置下自动启用，无需修改当前 fast YAML。启动日志应出现：

```text
LogKV checkpoint routing replay: enabled for 28 Blocks (strict, no rerouting fallback)
```

每一次 checkpoint 调用的两个上下文共享自己的记录列表，独立于其他层和 micro-batch。
首次 forward 正常 routing，将生成的 op-log 引用交给这次 checkpoint；重算通过该日志
replay 并重建 ladder，不重新选择 cluster、不生成第二份日志。随后的 LogKV backward
继续使用同一日志。保存的只有已有 op-log 的引用和调用元数据，没有额外保留 Q/K/V 或
cache 快照；生命周期由 checkpoint 计算图管理，丢弃图即可释放记录，也支持 retain_graph
后的再次 backward。当前是 non-reentrant checkpoint，原始 autograd ctx 仍保留，
重算的作用是恢复 saved tensors，并非必须替换成一个新 ctx。

这是严格路径：缺失日志、cache/输入形状/配置不匹配会报错，不会静默重新 routing。
安装时检查每个 Block 都已包装，并拒绝 reentrant、自定义 checkpoint 函数/上下文及
debug 模式。没有 activation checkpoint 时不需要这项复用；普通单 ladder/K=1 不启用。
该接口针对当前 Fabric/PyTorch 包装方式，不支持通过 torch.compile 编译整个训练图。

按用户此前同配置的调用次数，预计变化为：

| 计数 | 原路径 | 启用复用 |
|---|---:|---:|
| route_n | 13440 | 6720 |
| replay_n | 6720 | 13440 |
| plan_n | 14336 | 14336 |
| attn_fwd_n | 21504 | 21504 |
| attn_bwd_n | 7168 | 7168 |

重算仍需写入/合并 ladder、生成 plan、计算 attention，所以不会消除全部 checkpoint
开销。此前约 240 秒/step 的收益估算尚未实测，需用稳定训练 step 验证。

CPU 回归（本机 Python 3.9/torch 2.6，临时加载器兼容类型注解并跳过 checkpoint I/O
依赖）为 296 passed、21 skipped；新增 CPU 测试包含实际 Block checkpoint、两次梯度
累积、多份未反向的图、retain_graph、日志释放、严格错误处理，以及输出/梯度/cache
一致性和 routing 次数减半。CUDA、双卡 FSDP 本机未执行。目标机验证：

```bash
python -m pytest -q tests/test_log_kv_checkpoint.py
torchrun --standalone --nproc_per_node=2 -m pytest -q tests/test_log_kv_checkpoint.py -k fsdp
```

第一条在 A800 上包含 bf16/Triton/Flash 前后向检查，双卡项在未使用 torchrun 时跳过；
第二条实际运行 SHARD_GRAD_OP、Block checkpoint 和两次 micro-batch 梯度累积，
核对各 rank 的损失、梯度及调用次数。之后使用原来的 8 卡训练命令测量完整 step。

## 跨 lane 批量分配 orphan（2026-09-18）

`_semantic_route_orphans_fast` 现在只收集有 orphan 的 lane，批量执行最远点选种和
最近簇计算。种子与分配结果合并为一次 `.cpu().tolist()`，不再每个 lane 分别回传。
临时输入按本次 flush 的最大 orphan 数补齐，距离张量为 `[活跃 lane, orphan, K]`，
不生成额外带 head_dim 维度的四维距离张量。原有中心在整个 flush 内保持不变；
临时种子不会提前修改真实 centroid。容量预留、满容量时保留 token、按时间提交和
segment/pad/op-log 规则沿用原实现。

第一轮 checkpoint 严格复用机制不变，无需修改 YAML 或训练命令；正在运行的进程
继续使用已加载的代码，下次启动使用第二轮实现。route/replay 的调用次数不会因
本轮再次减少，应比较相同配置下预热后的耗时。

CPU 分配微测（K=8，所有簇已满，每 lane 47 个 orphan，D=8，排除 ladder 写入）：

| lane 数 | 修改前 ATen 调用 | 修改后 ATen 调用 | 修改前/后结果回传次数 |
|---|---:|---:|---:|
| 8 | 152 | 27 | 8 / 1 |
| 32 | 608 | 27 | 32 / 1 |

这些是操作和 `.cpu()` 调用计数，不是 GPU 耗时或整步加速倍数。Phase 1 的结果回传、
实际新建簇和 ladder 写入仍存在；本轮没有消除全部 routing 同步和写入开销。
与修改前函数的 8 组混合 lane CPU 差分（fp32/bf16、容量开关、8/32 lane）分配一致。
本地相关回归 302 passed、25 skipped；新增测试覆盖不等长 lane、空/满/部分占用的簇、
相同 key、非连续输入、两条 replay 路径及分配操作数不随 lane 数增长。
第二轮 CUDA 尚未在本机执行，在 A800 上运行：

```bash
python -m pytest -q tests/test_log_kv_orphans.py tests/test_log_kv_checkpoint.py
torchrun --standalone --nproc_per_node=2 -m pytest -q tests/test_log_kv_checkpoint.py -k fsdp
```

第一轮的上述单卡和双卡检查此前已由用户在目标机报告通过；第二轮需要重新验证。

## routing/replay 共用 staging 减少复制（2026-09-18）

本轮仅实现两项已确认的优化：

- 两个提交入口保留原始 K/V 和 positions 视图，将逻辑 token 索引在 CPU 转为
  `(batch, group, token)` 坐标后一次上传，直接 gather。不再先 reshape 整块非连续
  K/V；位置视图也无需先展平复制。坐标索引比原方案每个选中 token 多一个 int64，
  用来避免 K/V payload 的整块副本，不增加跨 chunk 持久缓存。
- 无 pad 时 gather 结果直接作为 staging；有 pad 时在新分配的零 buffer 上原地写入。
  初始 p_lo/p_hi/sum_wp 相同，复用同一份只读位置张量，写入缓存时仍进入独立字段。
  staging 在 ladder 写入/合并时只读，centroid 的求和输入和求和顺序不变。

没有引入执行计划缓存，也未修改 op-log 解析、新建簇/pad 调度或合并算法。前两轮优化
及严格 checkpoint 复用保留，训练命令和 YAML 不变。

CPU 无 pad staging 微测（batch=4、groups=8、chunk=128、D=128，非连续 K/V 及
展开的 positions；不含 ladder 写入和 centroid 更新）：

| 指标 | 修改前 | 修改后 |
|---|---:|---:|
| ATen 调用 | 33 | 16 |
| clone | 3 | 0 |
| 非原地 index_copy | 2 | 0 |
| 原地 index_copy_ | 3 | 0 |

这不是完整 replay 或训练 step 的加速倍数。与修改前实现的实际两层 Block checkpoint
差分中，mid/无 segment、multi/有 segment、二阶 0.2/有 segment 三个配置，在两次
micro-batch 梯度累积后损失、全部参数梯度和 cache buffers 逐位一致。
本地完整相关回归为 308 passed、29 skipped；新增测试还覆盖 fp32/bf16、非连续 K/V、
共享位置视图、pad 零填充、staging 不被写坏、输入不被修改，以及输出/梯度一致性。
本轮 CUDA 测试本机未执行，目标 A800 上运行：

```bash
python -m pytest -q tests/test_log_kv_staging.py tests/test_log_kv_orphans.py tests/test_log_kv_checkpoint.py
torchrun --standalone --nproc_per_node=2 -m pytest -q tests/test_log_kv_checkpoint.py -k fsdp
```

然后用原配置观察稳定 step 的 `route` / `replay` 时间和 `Time`。本轮仅减少复制，
预期调用次数与第一轮后相同，不应把调用次数不变误判为没有启用。
