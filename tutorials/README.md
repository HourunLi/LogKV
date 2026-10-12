# 教程索引与适用范围

本目录主要保留通用 LitGPT 教程。SemanticLogKV 的当前路线、CPT 与评测使用
[项目 README](../README.md) 和根目录 `majob.sh` / `eval.sh`，通用 `litgpt pretrain`、
`generate`、`evaluate`、`LLM` API 不自动启用 attach + Alpha + Beta。

## SemanticLogKV 研究入口

- [共享算法规格](../docs/algorithm-spec.md)：默认配置、缓存与回放、复杂度边界。
- [Alpha](../docs/alpha-logkv.md)、[Beta](../docs/beta-logkv.md)、
  [unified 对照](../docs/semantic-unified-routing.md)、[位置表示](../docs/position.md)。
- [评测指标](../docs/evaluation-metrics.md)、[SWA 同预算对照](../docs/swa-niah-comparison.md)、
  [Baseline 选型与公开源码](../docs/baselines.md)。

## 通用 LitGPT 教程

| 用途 | 文档 |
|---|---|
| 入门、模型与权重 | [完整入门](0_to_litgpt.md)、[下载](download_model_weights.md)、[HF 转 LitGPT](convert_hf_checkpoint.md)、[LitGPT 转 HF](convert_lit_models.md) |
| 训练与数据 | [预训练](pretrain.md)、[TinyLlama](pretrain_tinyllama.md)、[数据准备](prepare_dataset.md)、[微调总览](finetune.md) |
| 微调方法 | [全参数](finetune_full.md)、[LoRA](finetune_lora.md)、[Adapter](finetune_adapter.md) |
| 推理与评测 | [推理](inference.md)、[Python API](python-api.md)、[通用评测](evaluation.md)、[部署](deploy.md) |
| 资源与性能 | [权重量化](quantize.md)、[OOM](oom.md)、[历史资源表](resource-tables.md) |
| 开发与示例 | [开发文档](developer-docs/README.md)、[Lightning Trainer 示例](examples/ptl-trainer/README.md) |

命令以 [当前 CLI](../litgpt/__main__.py) 为准：`finetune` 是 LoRA 别名；全参数微调使用
`finetune_full`，Adapter 使用 `finetune_adapter` / `finetune_adapter_v2`。不使用旧的
`finetune lora` 或 `finetune adapter_v2` 子命令写法。

旧教程中的示例输出、GPU 资源表、下载列表与云端成本不是当前模型和环境的实测承诺；
当前支持模型以 [配置源码](../litgpt/config.py) 为准。量化教程针对权重，不代表 KV 量化。
[Thunder](../extensions/thunder/README.md) 与 [TPU/XLA](../extensions/xla/README.md) 是通用扩展，
其示例不构成当前 SemanticLogKV CUDA 路由、graph、流与回放已验证的证据。
