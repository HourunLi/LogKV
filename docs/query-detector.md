# GLiClass 动态任务片段与固定双池（仅评测，显式开启）

本功能在 LM prefill 前扫描当前可见的完整输入，不再固定取开头或结尾 64 token。
句子/换行切分与通用语言规则先筛候选，GLiClass 对候选及附近上下文做零样本分类。
选中片段通过 LM tokenizer 的原始字符偏移映射回 cue token id，复用 Gamma 的准入证据。
没有阳性片段时不注入 query cue，也不强制选 top-1。

GLiClass 每个评测 rank 懒加载一次，每个生成请求检测一次；各层共用 cue。
`niah_turns` 只检测缓存建立时可见的 context，不读取后续轮的问题。
loglikelihood 和 PPL 路径不扫描待打分文本。原有配置的行为保持不变。

## 1. 联网机准备普通文件和依赖

联网机应与离线机使用相同 Python 版本、操作系统和 CPU 架构，以准备兼容的 wheel。
以下命令在仓库根目录执行。已有合适的 PyTorch 时会直接复用它。

```bash
python unused/prepare_query_detector.py --install-deps --wheels
```

模型为 `knowledgator/gliclass-instruct-edge-v1.0`（32.7M）。脚本通过 HTTPS 直接把模型、
tokenizer 和配置写成普通文件，不使用 Hub 的 snapshots/blob 软链接结构。
默认固定到已验证的 commit `727be8a417f6a7718e591b025e07054c146d8139`，
检查权重的上游 SHA256，并为所有交付文件生成 `bundle.json`。
`--revision <commit>` 可指定固定版本；`--endpoint https://hf-mirror.com` 可更换访问入口。
按当前环境要求，模型 API、权重下载和脚本内 pip 操作默认跳过 HTTPS 证书校验；
`--verify-ssl` 恢复模型下载的证书校验，pip 恢复自身正常的证书配置。
pip 会自动信任 PyPI、文件 CDN 和环境变量/pip 配置里的镜像主机；额外下载主机可加
`--pip-trusted-host <hostname>`。这些设置只作用于本脚本的请求和子进程，不修改全局 SSL/pip 配置。
文件的 SHA256 校验继续执行。
文件通过复制而非 rename 发布，适用于不支持 rename 的普通对象存储挂载。

`--wheels` 根据已安装的固定版本导出完整运行依赖闭包，并下载普通 wheel 文件。
**不打包 PyTorch 及其 CUDA 依赖**，离线机继续使用已有的 `2.7 <= torch < 3` 环境。
不使用 `--wheels` 时，模型下载和校验脚本仅需 Python 标准库。

默认模型目录为：

```text
/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/gliclass-instruct-edge
```

下载器、smoke 工具和两份评测 YAML 使用同一默认位置，避免写到仓库的 `data/`。
`--out <实际目录>` 仍可覆盖下载位置。目录包含 `model.safetensors`、`config.json`、`tokenizer.json`、tokenizer 配置、
`bundle.json`、`detector-requirements.txt` 和 `wheels/`。将整个目录通过普通文件复制传到离线机。
也可以打包后传输：

```bash
CACHE_DIR=/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache
tar -C "$CACHE_DIR" -czf "$CACHE_DIR/gliclass-instruct-edge.tar.gz" gliclass-instruct-edge
```

GLiClass 0.1.15 支持该权重需要的 segment embeddings；更新版 GLiClass 的依赖可能要求
Transformers 5。这里固定到 Transformers 4.57.3，避免为此切换到 5.x。项目 extra 的兼容上限
相应扩展到 `<4.58`；只开启本功能时才需要安装 `query-detector` extra。

## 2. 离线机安装和检查

解包或复制到实际路径后，运行：

```bash
MODEL_DIR=/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/gliclass-instruct-edge
python unused/prepare_query_detector.py --verify
python -m pip install --no-index --findlinks "$MODEL_DIR/wheels" \
  -r "$MODEL_DIR/detector-requirements.txt"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python unused/query_detector_smoke.py
```

加载器要求真实本地目录，拒绝 Hub ID、目录中的软链接和缺失文件；模型和 tokenizer 都使用
`local_files_only=True`，不启用 remote code。模型内嵌 encoder config，无需另外下载基础编码器。
加载缺失/额外/形状不匹配的权重时直接报错，不用随机初始化的头继续评测。

可用自己的普通文本检查输出：

```bash
python unused/query_detector_smoke.py \
  --text-file /data/sample_prompt.txt --device cuda:0
```

输出为片段起止字符、原文、分数和累计检测耗时。`--no-rules` 扫描所有片段，适合观察规则筛选的漏检。

## 3. 评测入口

两个新增配置都继承现有相同模型大小的 `eval_full_word_first_query.yaml`，使用同一 checkpoint：

```bash
export LOGKV_QUERY_DETECTOR_DIR=/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/gliclass-instruct-edge
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

已在 CPU 上完成普通文件模型下载、依赖打包、SHA256 校验，并在阻止 socket 连接的进程中
验证本地模型加载和推理。软件测试覆盖全文中部定位、拒绝全部候选、原始 token 映射、
双池重复保留、只归档一次、批次内不等长状态、分数预取和缓存 reset。

该小模型未针对本项目训练。实际小样例中能识别直接问题和祈使任务，但也将记者引用的问题
误判为当前任务；少量示例并未稳定解决此问题。因此它是可复评的实验入口，不宣称已达到
可靠的任务角色识别，更不宣称提高 BABILong/RULER 准确率。英文小模型的中文能力也未经验证。
目前没有 Qwen checkpoint/GPU 的端到端性能结果。

全局任务角色可能依赖候选附近以外的上下文，有限窗口不能解决所有远距离引用。
检测器只能读取缓存建立时可见的文本，不会恢复已经压缩丢失的信息。
