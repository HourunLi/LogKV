# 同内存基线：64K / 128K

外部基线都是免训练方法，跑在原始 Qwen3-1.7B-Base 上（与 `eval_long{64k,128k}_dense.yaml` 同一权重）。
每个基线的持久 KV 缓存字节数，对齐同一长度下 LogKV 缓存的实际分配。预填充按块流式进行，整个请求里缓存都
不超过这个预算，不会出现先整段 dense 预填充、再压缩的 O(N) 阶段。实现见 `litgpt/budget_kv_cache.py`
（SWA 走 `litgpt/model.py` 原有的滑窗路径）。

## 方法

| 配置后缀 | 方法 | 每层每个 KV 头保留的条目（共 C 条，SWA 为窗口 W） |
|---|---|---|
| `swa` | 全程滑窗注意力 | 每个 query 只看最近 W 个 token |
| `streamingllm` | StreamingLLM | 前 4 个位置（attention sink），加最近的 C−4 个 |
| `h2o` | H2O | 最近的 C/2 个；其余按至今收到的注意力累计值（自进入缓存起所有 query 的 softmax 概率之和，对同组 query 头取平均） |
| `snapkv` | SnapKV（逐块） | 最近的 64 个；其余按本块最后 64 个 query 的注意力打分（对这些 query 与同组头取平均，再沿保留条目的位置顺序做宽 5 的平均池化） |

- 输入按 1024 个 token 一块处理：块内 query 读 [保留条目 + 本块（因果）]，块结束后按策略留下 C 条。
  解码一步就是一块（SnapKV 的观察窗口这时为 1）。
- 每个 KV 头各自选择。K 存 RoPE 之后的值，留在原位置，不重新旋转。
- SnapKV 每块重新打分，不跨块累计。原版 SnapKV 是整段预填充后只压一次，需要 O(N) 的预填充，不在此口径内。
- 参数可用 YAML 的 `kv_baseline_options` 修改：`sinks`、`recent`、`window`、`kernel`、`chunk`。
  `kv_baseline_budget` 为 0 时按字节对齐，大于 0 时直接给出 C。

## 预算口径

先用 YAML 里同一组 `log_kv_*` 设置，在该长度的 `max_seq_length` 下建 LogKV 缓存，统计已分配的注册 buffer
字节数 R（与 [全程 SWA 对照](swa-niah-comparison.md) 的口径相同：含空槽、recent 双缓冲和元数据；不含共享 RoPE 表和
临时工作区）。然后：

- 驱逐基线：C = ⌊R / (n_layer · G · (2 · head_size · 元素字节 + 8 + s))⌋。其中 G 是 KV 头数，8 是 int64 位置；
  H2O 的累计分数为 float32，s = 4，其余方法 s = 0。
- SWA：W 由同一个 R 换算（`swa_window_size: 0`）。

结果 JSON 的 `cache_budget` 记录 R（`reference_logkv_bytes`）、基线实际的 `persistent_cache_bytes` 和 C；
`kv_compression` 记录 `kv_compression_factor`（已处理 token 数 / 保留条目数）。比较时看各方法在报告的压缩比下的分数。

## 显存口径

- 持久 KV 缓存在任何时刻都不超过预算。
- 临时工作区是一块的 K/V 与分数，为 O(C + 块长)。
- 模型激活不在缓存口径内：当前评测把整段输入一次送进模型，隐状态和 MLP 激活是 O(N)。dense、LogKV 和各基线都是如此，
  与缓存方法无关。

## 运行

```bash
bash eval.sh exp/qwen1.7b-32k/eval_long64k_snapkv.yaml
```

配置为 `eval_long{64k,128k}_{swa,streamingllm,h2o,snapkv}.yaml`，都继承 `eval_long{64k,128k}_baseline.yaml`：任务、样本数、
YaRN 与 `block_size` 相同，它的 `log_kv_*` 设置只用来给出预算。结果写到 `dense_niah_base_eval/long{64k,128k}_<方法>`。

## 限制

- LogKV 用的是 CPT checkpoint，基线与 dense 用的是原始权重；两者的差距里包含 CPT 的影响。
- H2O 与 SnapKV 在 SDPA 之外再用 float32 算一遍打分用的注意力概率（H2O 每块全部 query，SnapKV 只算观察窗口）。
