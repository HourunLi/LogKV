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
路由现在每次批处理最多 4 个 batch/KV group，距离矩阵、互近邻配对和中心更新一起计算，
每轮统一回传配对。Python 簇成员关系在合并结束后统一重建，旧簇物理重建仍按原顺序执行。
批处理会增加临时工作区，持久 KV buffer 布局不变；GPU 峰值和路由耗时需要实测。
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

在训练环境先运行这个短基准，不加载模型、不启动 CPT：

```bash
python unused/benchmark_log_kv_unified.py --device cuda --batch 4 --groups 8 --iters 2 > route_profile.jsonl
```

第一行报告同步完成后的总 route 时间/flush 和额外峰值显存；第二行单独插桩，报告
候选建簇、统一合并规划、旧 KV 合并、新簇写入、批量写入耗时及每次搜索提出的配对数
（尚未经过精确半径复核和最终预算截断）。
阶段插桩会增加同步，不能拿它替代第一行总耗时。默认随机输入和两次 flush；这是性能诊断，
不代表真实 NIAH 质量、整步训练吞吐或完整训练峰值。现有新路由 YAML 无需加新开关。

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
