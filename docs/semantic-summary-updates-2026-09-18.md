# SemanticLogKV 更新路径：并行 centroid、小组摘要、训练更新复用

本轮只实现这三项；32K、K=8、B=64、global batch=128、micro batch=1 和 Block
activation checkpoint 保持原配置。没有接入 hash，没有新增或移除原有快路由的建簇规则。

## 配置与兼容边界

`exp/qwen1.7b-32k/arc_semantic_fast.yaml` 已开启：

```yaml
log_kv_semantic_centroid_backend: parallel
log_kv_semantic_summary_size: 8
log_kv_semantic_replay_updates: true
```

三个字段的代码默认值分别是 `sequential`、`1`、`false`，旧配置沿用旧路径。
训练、直接 eval、`eval.sh`、`majob.sh` 都透传这些字段，启动日志打印实际取值。
做旧路径对照时，在独立 YAML 中显式覆盖这三个值；本项目 YAML 的非 null 值会覆盖 CLI。

- `parallel`：CUDA centroid 每次读取 32 个 token，沿 token 维归约，再处理下一块。
  不使用 atomic；同一簇不同 ordinal 的更新仍按原顺序执行。继续使用独立输出和单独
  commit kernel，避免读写同一份旧 centroid/n_eff。并行加法允许舍入差异，路由边界上的
  token 可能改变归属。CPU 使用原生 segment_reduce，不具有这个 Triton 加速。
- `summary_size=8`：JOIN 内每个簇的连续至多 8 个 token 生成一个加权条目，尾组立即
  提交；不跨簇、不跨 flush 合并。新簇的首个 token 仍走原来的创建路径。保留质量、
  首末位置和 int64 位置和，centroid/n_total 仍根据原始 token 更新。摘要进入原有
  ladder 后按质量加权合并，mid attention 仍使用真实质量和位置统计。
- `replay_updates=true`：首次训练 forward 记录每次 flush 的 clear/append 动作、
  append 输入及更新后的路由状态。checkpoint 重算和 backward 直接执行这些 append，
  恢复 centroid/n_eff 等状态；跳过 op-log 解析、原始 K/V gather、摘要和 centroid 重算。
  ladder 仍需重建，attention 仍需计算，不是取消 backward。

摘要仅支持一阶、无 segment 边界的 fast three-phase 路径；遇到 second_order 非零、
有限 seg_gap_max、legacy 或 chunk-tree 组合会明确报错。`summary_size=1` 恢复原始条目，
并可继续使用 segment/pad。更新复用限 fast three-phase、K>1，其他路由明确拒绝。

摘要改变压缩近似，不要求与 summary_size=1 的输出一致；训练和推理应使用相同的摘要大小。
验证要求是同一新配置的普通 replay、保存更新 replay、checkpoint 的缓存、输出和梯度一致。

## 生命周期、因果性与设备同步

- 原 CPU-only `_SemanticReplayPlans` 保持无 Tensor；设备更新由独立的
  `_SemanticReplayUpdates` 持有。CPU 保存位置、动作和状态镜像；K/V 摘要、centroid 等
  payload 始终在原设备，记录时 detach+clone，避免 recent/cache 被覆盖后读到新数据。
- 每个 autograd/checkpoint 调用独立拥有记录。autograd 通过 save_for_backward 保存
  payload；ctx 的执行描述不含 payload Tensor。checkpoint 帧也持有原记录以支持重算。
  retain_graph 可重复使用，计算图释放后 payload 释放。cache 仅在提交调用内临时绑定
  记录，finally 清除，不把不同 micro-batch 的记录混在一起。
- CUDA 记录完成事件。重算/反向清空可变 cache 前先等待生产事件；消费每个 flush 时也
  做设备侧事件等待，并 record_stream 保护跨流读取的存储生命周期。不做 CPU synchronize
  或 payload CPU offload。原路由自身已有的决策读回仍然存在。
- 更新记录校验 shape/dtype/device、配置、CPU positions 和 op-log 游标；缺失、重复或
  顺序错误直接报错，不回退 rerouting。
- 更新始终在当前 chunk attention 之后提交。历史摘要仍 stop-gradient；当前 chunk 的
  K/V 梯度通过原 attention backward 计算。
- 推理从不绑定训练记录，因此不保留线性的摘要历史，继续使用 recent + K 条有界 ladder。
  训练为了复用摘要会增加 O(T / summary_size) 的 payload；这不是 O(log T) 的训练记录。

生产 mbs=1、G=8、D=128、bf16、summary=8 时，仅 K/V 摘要的理想大小约为每层 16 MiB
（按完整 32K 上界估计），28 层约 448 MiB，另有尾组、初始 token 和路由状态等开销。
摘要=1 且启用更新复用会明显增加训练显存；应结合基准输出和完整 step 峰值决定。

## 基准与验证

新基准分开报告四种变体：原顺序累加、并行 centroid、增加摘要、再增加更新复用。
每种变体按自己的压缩定义生成参考状态，每个 warmup/计时迭代都在计时结束后校验。
报告包含 route/replay 中位时间、峰值额外显存、单 flush 保存 payload 大小和实际 entry 数。
缓存复制、编译预热和结果比较不计入测量。CPU 的 peak 字段为占位 0，不代表零内存。

```bash
python -m pytest -q tests/test_log_kv_summary_updates.py tests/test_log_kv_checkpoint.py tests/test_log_kv_replay_plan.py
python unused/benchmark_log_kv_summary_updates.py --iters 10
# 覆盖新旧配置的双卡 FSDP + bf16 + Flash backward
torchrun --standalone --nproc_per_node=2 -m pytest -q tests/test_log_kv_checkpoint.py -k fsdp
```

本机没有 CUDA；Triton 新分支、A800 显存和完整训练 step 时间待目标机验证。
CPU/torch 2.6 合成基准，T=32768、chunk=2048、batch=1、G=8、D=128、3 warmup + 3 计时：

| 变体 | route ms | replay ms | lane 0 entry 数 |
| --- | ---: | ---: | ---: |
| sequential | 82.254 | 48.396 | 3061 |
| summary=8 | 63.309 | 29.934 | 1595 |
| summary=8 + 更新复用 | 62.913 | 11.904 | 1595 |

这些是单次末尾 flush 的 CPU 合成结果，不能当作 A800 tokens/s 或训练减半的证据。
CPU parallel 仍执行相同的 segment_reduce，因此其计时波动不代表并行 kernel 收益。
本次基准也捕获并修复了预填充 cache 未显式 reset 时的 replay 游标初始化问题。

本机回归：349 passed、68 skipped、1 warning。覆盖缓存、更新算子、replay、checkpoint、
输入打包、位置、Flash 接口、计时及启动脚本；环境为 Python 3.9/torch 2.6，使用已有临时
bootstrap 加载实际 model/cache，跳过项包含 CUDA/Triton/FSDP 路径。补改 null 参数透传后，
脚本参数测试另跑 16 passed。语法编译、shell 语法和 git diff --check 通过。
