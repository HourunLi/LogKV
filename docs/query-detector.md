# GLiClass 动态任务片段与固定双池（仅评测，显式开启）

本功能在 LM prefill 前扫描当前可见的完整输入，不再固定取开头或结尾 64 token。
句子/换行切分与通用语言规则先筛候选，GLiClass 对候选及附近上下文做零样本分类。
选中片段通过 LM tokenizer 的原始字符偏移映射回 cue token id，复用 Gamma 的准入证据。
没有阳性片段时不注入 query cue，也不强制选 top-1。

GLiClass 每个评测 rank 懒加载一次，每个生成请求检测一次；各层共用 cue。
`niah_turns` 只检测缓存建立时可见的 context，不读取后续轮的问题。
loglikelihood 和 PPL 路径不扫描待打分文本。原有配置的行为保持不变。

## 1. 只下载必要模型文件

在联网机的仓库根目录运行：

```bash
python unused/prepare_query_detector.py
```

默认目录为：

```text
/workspace/data/hf_cache/gliclass-instruct-edge
```

仅下载四个运行文件：`model.safetensors`、`config.json`、`tokenizer.json`、`tokenizer_config.json`，
并生成记录 SHA256 的 `bundle.json`。权重已有且与上游 SHA256 一致时直接复用。
下载固定为 `knowledgator/gliclass-instruct-edge-v1.0` 的 commit
`727be8a417f6a7718e591b025e07054c146d8139`；`--revision <commit>` 可显式覆盖。

脚本只使用 Python 标准库，不调用 pip、不安装包、不下载 wheel、不改变任何依赖版本。
旧的 `--install-deps`、`--wheels` 参数已移除，误用时会在下载之前报错。
所有文件直接写入普通目录，不使用 Hub snapshots/blob 软链接；通过复制而非 rename 发布。
默认跳过 HTTPS 证书校验，`--verify-ssl` 恢复；SHA256 文件校验始终执行。
下载默认使用 `https://hf-mirror.com`，元数据和模型文件请求均走镜像，不自动回退官方站点。
`--endpoint <镜像地址>` 可显式更换镜像，`--out <目录>` 可覆盖保存位置。

将模型目录按普通文件复制到离线机即可。已下载的模型目录可以继续复用，其中旧的 wheel 文件
不会被加载或安装，也不会被脚本自动删除。

## 2. 复用现有依赖与离线检查

本功能不需要安装外部 `gliclass` 包。`litgpt/gliclass_local.py` 提供仅推理适配层：
使用现有 Transformers 的 ModernBERT 编码器，以及项目已有的 PyTorch、tokenizers、safetensors，
读取本地权重并执行 GLiClass 的 segment embedding、投影、标签池化和分类头。
模型参数严格加载；不匹配时明确报错，不静默改用随机参数。

仓库原有 `transformers>=4.51.3,<4.57` 约束保持不变，不增加版本锁定的 detector extra。
适配层固定使用 SDPA、关闭自动 compile，不要求额外的 Flash Attention 包。
使用本地 JSON 构造 encoder 和 tokenizer，不访问远程基础编码器或 remote code。

在现有评测环境中直接运行：

```bash
python unused/prepare_query_detector.py --verify
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python unused/query_detector_smoke.py
```

下载器、smoke 工具和两份评测 YAML 使用同一默认目录。其他位置可显式指定：

```bash
python unused/query_detector_smoke.py --model-dir /actual/model/directory \
  --text-file /actual/sample_prompt.txt --device cuda:0
```

输出片段起止字符、原文、分数和累计检测耗时。`--no-rules` 扫描所有片段以检查规则漏检。

## 3. 评测入口

两个新增配置都继承现有相同模型大小的 `eval_full_word_first_query.yaml`，使用同一 checkpoint：

```bash
export LOGKV_QUERY_DETECTOR_DIR=/workspace/data/hf_cache/gliclass-instruct-edge
GPUS_PER_NODE=1 LIMIT=2 EVAL_LENGTHS=32k EVAL_SUITES=ruler,babilong,turns \
  bash eval.sh exp/qwen1.7b-32k/eval_full_word_first_query_detected.yaml
```

8B 对应 `exp/qwen8b-32k/eval_full_word_first_query_detected.yaml`。
去掉试跑限制即沿用原有 32K–256K 全量流程。结果使用独立的 `eval_tag`。
`eval_main`、`eval_lengths` 和任务选择遵循原入口约定。

主要参数如下：

| 参数 | 新入口值 | 含义 |
|---|---|---|
| `log_kv_gamma_query_detector` | 本地模型目录 | 未设置时仍使用旧 query 提取逻辑 |
| `log_kv_gamma_cue_tokens` | `0` | 动态检测模式要求关闭固定开头 cue |
| `log_kv_gamma_query_cue` | `true` | 接入请求级 cue |
| `log_kv_gamma_query_pool_tokens` | `512` | query 池固定额度；`0` 是旧单池模式 |
| `log_kv_gamma_surprisal` | `word_first` | 固定双池的另一条保留策略 |

`log_kv_gamma_query_detector_options` 支持 `device`、`threshold`、`batch_size`、`max_chars`、
`context_chars`、`max_spans`、`use_rules`。新入口的 `device: model` 使用本 rank 的 LM 设备；
也可写 `cpu`。默认阈值 0.8 是未针对 benchmark 调整的起始值，不代表校准后的正确概率。
512 字符候选与前后各 128 字符是计算分块长度，不是输入位置限制。默认最多保留 8 个候选，
相邻阳性片段合并。规则属于通用问句/任务结构筛选，可以关闭，未覆盖的表达可能漏检。

## 4. 固定双池与重复 attention

总精确槽仍为 `log_kv_alpha_exact_tokens`。query 池固定占 `query_pool_tokens`，word_first 池
占其余槽；两者分别排序填充，不借用额度。每池额度必须至少覆盖一个最大未闭合片段。
两路共享一次设备打分和分数读回，并保留原有的分数预取路径。

两个池可保留同一真实片段的两份 K/V，保留原始 RoPE 位置，进入同一次 attention。
相同副本在 softmax 中合起来相当于该项 logit 加 `ln(2)`。重复槽计入物理预算；
只有两池都不保留某个原始 token 时，它才归档一次。梯子质量与原始位置统计不重复计数。
缓存 reset 和回放状态包含两池归属；推理 KV 预算的 `O(log N)` 阶数保持不变。

`semantic_cluster_stats` 增加每行 `alpha_unique_tokens`，区别真实 token 数与物理副本槽数。
结果 JSON 增加 detector 配置与 `query_detector_stats`（当前 rank 的加载/扫描耗时、候选数、
阳性片段数、模型 revision）。检测耗时单列，原 LM prefill 吞吐不包含它，评估总体收益时须合计。

## 5. 已验证与限制

已在 CPU 上完成普通文件模型下载、SHA256 校验和禁网加载/推理。
本地适配层在隔离的 Transformers 4.51.3、4.56.2 环境验证，并与参考 GLiClass 实现
比较真实权重的预测分数；这些兼容性检查不修改现有环境。软件测试覆盖全文中部定位、拒绝全部候选、原始 token 映射、
双池重复保留、只归档一次、批次内不等长状态、分数预取和缓存 reset。

该小模型未针对本项目训练。实际小样例中能识别直接问题和祈使任务，但也将记者引用的问题
误判为当前任务；少量示例并未稳定解决此问题。因此它是可复评的实验入口，不宣称已达到
可靠的任务角色识别，更不宣称提高 BABILong/RULER 准确率。英文小模型的中文能力也未经验证。
目前没有 Qwen checkpoint/GPU 的端到端性能结果。

全局任务角色可能依赖候选附近以外的上下文，有限窗口不能解决所有远距离引用。
检测器只能读取缓存建立时可见的文本，不会恢复已经压缩丢失的信息。
