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
