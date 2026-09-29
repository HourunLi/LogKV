# Semantic LogKV / 全程 SWA：32K NIAH 配对实验

本轮只做缓存方式对照，不追加训练，不修改语义簇算法。两组使用同一份已经完成
CPT 的权重，因此这是该 checkpoint 上的推理消融，不能推论 SWA 独立训练后的上限。

## 运行

在原有 GPU 评测环境、仓库根目录执行（单机 8 卡；可改 `GPUS_PER_NODE`）：

```bash
bash unused/compare_swa_niah.sh \
  /home/ma-user/work/bucket-pangu-green/lihourun/ckpt-logKV/qwen1.7b-32k-cpt-logKV-semantic-fast
```

脚本只解析一次最新 `step_*`，随后把同一具体 checkpoint 路径传给两组。请使用已经
结束训练的 checkpoint，避免权重文件在评测中被覆写。脚本使用当前 Python 环境，
不安装依赖、不训练。默认全部样本，先检查环境可在命令末尾加 `--limit 4`；这只适合
冒烟检查，不能作为正式结果。两组均测 single1/2/3，仅使用 32768 长度。

结果分别写入该 checkpoint 下 `swa_compare_<时间>/semantic` 和 `swa`；可通过
`SWA_COMPARE_OUTPUT_DIR` 指定输出目录。`GPUS_PER_NODE=1` 可做单卡检查。
直接运行 `eval.py --config compare_niah_*.yaml` 也可；不要经 `eval.sh` 转发，那个
旧脚本尚未透传 `swa_window_size`。

## 对齐内容

- `compare_niah_semantic.yaml` 继承当前 `arc_semantic_fast.yaml`，包括 mid anchor、
  逐 token 条目、parallel centroid，避免误用缺少这些开关的旧 eval 配置。
- SWA 每层、每个 query 仅能看见位置 `[query-W+1, query]`，包含自身。prefill 从
  第一个 token 就遵守该限制，没有完整 dense prefill。环形缓存提交前先完成本块
  attention，防止提前覆盖本块早期 query 需要的历史。保留原始 RoPE 位置。
- `swa_window_size: 0` 先按同样配置构建 LogKV 缓存，计算已分配的注册缓存 buffer
  字节数，再换算能够容纳的最大 SWA 窗口，且不超过模型上下文上限。统计包含预留
  空槽、recent 双缓冲、原始 K 副本和注册元数据；不包含共享 RoPE、模型权重、CPU
  宿主副本和 attention/打包临时工作区。这是持久缓存分配预算对齐，不是峰值显存对齐。
- JSON 的 `cache_budget` 记录参考 LogKV 字节数、SWA 实际字节数和最终窗口；两组
  均保存逐样本结果。空槽也占显存，所以不能用有效 slot 数代替本次对齐口径。
  本机按 Qwen3-1.7B、bf16、32K 和当前 K=12/B=256/recent=2048 配置实算：28 层
  注册缓存 buffer 共 2,990,374,912 字节（约 2.79 GiB），对应 SWA 窗口 26,023。
  正式运行以 checkpoint 实际架构和日志中的预算为准。
- 两组采用相同 tokenizer、任务生成默认种子、截断及 greedy 生成逻辑。沿用现有
  为输出预留上下文空间的行为；发生 prompt 截断时日志会提示。

## 先看什么

先核对两组 checkpoint 路径、样本数、实际窗口和缓存字节数，再分别比较 single1/2/3；
不要只看三者均值。通过保存的样本核对 prompt/答案一致，区分两者都失败、仅 LogKV
失败、仅 SWA 失败。尤其注意 needle 是否落在 SWA 最终窗口内，但不要把“窗口外”
直接等同于必然失败：多层网络可能通过后续 token 传递信息。

当前 SWA 使用有界分块和显式局部 mask，先保证语义正确。现有评测的
`decode_ms_per_token` 包含 prefill，不是独立 decode 延迟；本轮不据此宣布速度优势。
无 CUDA 的本机只验证短序列数值与缓存预算，真实 32K 得分需在 GPU 环境运行。

后续约束：推理 KV 必须维持 O(log N)；可以从同一预算划出少量精确槽；允许从已有
权重 CPT，并加入与评测隔离的合成检索数据。具体机制等本轮对照结果后再决定。
