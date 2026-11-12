# LitGPT Python API 开发入口

面向使用者的例子统一维护在 [Python API 教程](../python-api.md)，实现为
[`litgpt/api.py`](../../litgpt/api.py)。这里不再维护与源码不一致的 API 设计草稿；旧提案可由 Git 查看。

当前 `LLM.load` 参数为 `model`、`init`、`tokenizer_dir`、`access_token`、`distribute`；
设备、精度和量化等分发设置通过 `LLM.distribute` 配置。`LLM.save` 保存 LitGPT checkpoint，
不接受旧草稿中的 `format="lightning | ollama | hf"`。
HF 转换使用 [转换入口](../convert_lit_models.md)。

`LLM` 当前有 `forward`、`trainer_setup`、`save`、`load`、`distribute`、`generate` 和
`benchmark` 等方法，没有旧草稿中的 `prepare_dataset`、`instruction_finetune`、`pretrain`
或 `serve` 实例方法。训练与服务分别见 [训练教程](../pretrain.md)、[微调教程](../finetune.md)
和 [部署教程](../deploy.md)。

修改接口时同步检查公开签名、[API 测试](../../tests/test_api.py) 和使用者教程。
SemanticLogKV 使用独立的 [训练与评测入口](../../README.md#运行入口)，不能从通用 API
支持某模型推断它已通过压缩缓存和重放验证。
