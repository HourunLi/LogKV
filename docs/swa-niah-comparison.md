# 全程 SWA 对照：范围与预算

此对照使用同一已完成 checkpoint，比的是缓存机制；不能据此推断 SWA 独立训练的上限。
SWA 在 prefill 和 decode 都使用滑窗，查询位置 q 只能看到 `[q−W+1, q]`，包含自身。
位置仍使用原始 RoPE 坐标。

## 现成脚本比较什么

[`compare_swa_niah.sh`](../unused/compare_swa_niah.sh) 将 checkpoint 固定到一次解析出的
实际权重目录，比较 `compare_niah_semantic.yaml` 与 `compare_niah_swa.yaml`。
前者继承 `arc_semantic_fast.yaml`，因此是原 fast semantic 路由，**不是 Alpha 或统一路由**。
当前该配置为 K=12、B=128、recent=2048、mid、一阶、逐 token 入层级。

```bash
bash unused/compare_swa_niah.sh '<已完成checkpoint目录>'
```

在 GPU 环境、仓库根目录执行。默认8卡、32K single1/2/3；`GPUS_PER_NODE=1` 可改为单卡，
附加 `--limit 4` 仅用于检查流程。两组输出分开保存在 `swa_compare_<时间>` 下，
可由 `SWA_COMPARE_OUTPUT_DIR` 指定目录。该脚本不训练、不安装依赖。
当前 `eval.sh` 不透传 `swa_window_size`，SWA 使用上述专门脚本。

## 预算口径

`swa_window_size=0` 先构建参考 LogKV，统计已分配的注册缓存 buffer 字节数，
再换算最大 SWA 窗口，并受模型上下文长度限制。包含空槽、recent 双缓冲、原始 K 副本
及注册元数据；排除共享 RoPE、权重、CPU 副本和打包/attention 临时工作区。
因此这是**持久缓存分配预算**对齐，不是总峰值显存对齐。

以结果 JSON 的 `cache_budget`、实际窗口和实际加载配置为准，不沿用旧 B=256 的窗口数值。
Alpha 从“压缩存储＋最大打包工作区”扣精确槽的内部预算口径与此不同，见
[Alpha 预算说明](alpha-logkv.md)，不能只比较两个 B 参数就宣称同内存。

## 结果判断

- 核对 checkpoint、tokenizer、任务长度、样本数、生成设置和预算，分别看 single1/2/3。
- 记录 prompt 是否被截断。needle 在 SWA 最终窗口外不等于必然失败，信息可能经后续 token 传递。
- 现有 `decode_ms_per_token` 包含 prefill，不能作为独立 decode 延迟。
- 上述脚本不自动构成 Alpha 的消融。Alpha 的 CPT/评测入口见 [Alpha 指南](alpha-logkv.md)。
