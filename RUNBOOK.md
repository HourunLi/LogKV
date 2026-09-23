# 32K Dense / SinkWindow / SemanticLogKV 对照实验 · 运行手册

这份手册假设三个分支（基准分支 + `.../dense` + `.../sinkwindow`）的改动都已经
可用（无论是分别 checkout 到同一个工作区合并使用，还是已经合并回一处）。本轮
交付只包含代码和工具，**所有步骤都没有在交付这份手册的会话里执行过**——这个
容器没有 GPU、没装 torch；请在你自己的训练/评测环境里按顺序跑一遍，出现任何
和本手册描述不一致的地方，先按§〇的态度处理，不要绕过去。

## §〇 先读这个：怎么对待这份手册和代码里的说明

审计这个仓库时反复发现同一类问题：文件名、注释、`expid`、旧文档描述的行为，
和实际生效的字段值/代码对不上（例子：`base.yaml` 的 `expid` 写 k1，路径写
k64，实际字段是 k_max=16；`arc_semantic_stage1_k16.yaml` 文件名叫 k16，其实
既没有走 `config:` 继承，`global_batch_size`/`max_steps` 也和 base.yaml 不同）。
这份手册和新增的所有工具都尽量只依赖"实际读出来的字段值"（`demo.py` 落盘的
`resolved_config.yaml`、`cache_accounting.py` 反射出的真实 buffer），而不是
硬编码的清单或者名字——但手册本身的文字描述仍然可能有没发现的类似问题。
按§九跑一遍验证，任何一步结果和这里写的对不上，先记录差异，不要静默选一个
近似值继续跑大规模训练/评测。

## §一 训练前必要验证（对应任务说明 §九，8 条最小检查）

这些测试文件已经写好（见"交付的测试文件"一节），但**从未在任何环境里跑过**。
在你的 GPU 环境里，按顺序跑：

```bash
# 1. 短序列等价性 + mask 正确性 + prefill/decode 一致性（Dense、SinkWindow 各一份）
pytest tests/test_dense_bypass.py -v
pytest tests/test_sinkwindow_cache.py -v

# 2. 字节统计正确性（含去重、mid-decode 瞬时态、op_log 排除）
pytest tests/test_cache_accounting.py -v

# 3. NIAH 导出的 needle/answer span 标注正确性
pytest tests/test_needle_spans_answer_exact.py -v

# 4. resolved-config dump 的快照/序列化正确性
pytest tests/test_resolved_config_dump.py -v

# 5. W 校准脚本的前提（单调性）和边界情形
pytest tests/test_calibrate_sinkwindow_w.py -v
```

全部通过之后，再做一次真正跨越窗口边界的手工检查（`test_sinkwindow_cache.py`
里已经有这个用例，但这是全实验里风险最高的一处，值得你自己再读一遍断言在
验证什么）：分块 prefill 和逐 token decode，在 token 数超过
`sink_size + window_size` 之后，两条路径必须给出完全一致的 cache 状态和
logits。这是设计阶段发现的一个真实 bug（环形缓冲取模写入配合"按列下标切
mask"，在多 token 一次性提交且跨越绕回边界时会读到本该看不到的未来 token）
修好之后留下的回归测试，如果这个测试失败，不要绕过去，SinkWindow 的正确性
不成立。

## §二 测算 SemanticLogKV 的持久缓存字节数

`exp/qwen1.7b-32k/semantic_stage1_frozen.yaml` 已确认指向 `arc_semantic_fast.yaml`
（文件内注释记录了继承链和关键超参数差异）。直接测：

```bash
python scripts/measure_cache_bytes.py exp/qwen1.7b-32k/semantic_stage1_frozen.yaml \
    --context_length 32768 --per_buffer
```

不需要 GPU、不需要真实权重（在 `torch.device("meta")` 上建模型，只看
shape/dtype）。记下 `structural_bytes`——这是 W 校准和最终预算表都要用的号，
不要用 `live_extra_bytes`（mid-decode 瞬时态，只做参考）也不要用
`torch.cuda.max_memory_allocated()` 之类的 allocator 数字。

如果 SemanticLogKV 是按内容动态分配、不同校准样本测出来的数字有波动，按任务
说明 §5.2 的要求，在校准集上多测几条取一个预先约定好的统计量（比如均值），
同时报告范围，不要在正式测试集上挑数字。

## §三 校准 SinkWindow 的 W

```bash
python scripts/calibrate_sinkwindow_w.py \
    --budget_config exp/qwen1.7b-32k/semantic_stage1_frozen.yaml \
    --context_length 32768 \
    --sink_size 4 \
    --output sinkwindow_w_calibration.json
```

`sink_size` 按任务说明 §4.1 的建议默认取 4；这个脚本对 `window_size` 做二分
搜索，找到不超过 §二 测出的 SemanticLogKV 结构性字节预算的最大整数 W。如果
算子要求窗口按某个粒度对齐，脚本会向下取整并在输出里报告剩余预算比例——按
任务说明 §5.2 的要求，这个比例要写进最终报告，不能只留一个整数 W。

把这一步定下来的 `window_size` 填进
`exp/qwen1.7b-32k/sinkwindow_stage1_train.yaml`/`sinkwindow_stage1_eval.yaml`
的 `log_kv_sink_window_window_size` 字段（两份文件都要改，值必须一致；当前是
故意写成 `-1` 的占位值，训练前必须替换——忘了替换会在训练/评测一启动就报错，
不会静默用错的窗口跑掉），然后**冻结**：S、W、SemanticLogKV 配置三者都不应该
在三组训练启动之后再变。

## §四 导出固定的 NIAH 样本集

```bash
python scripts/export_niah_samples.py \
    --tokenizer_dir <base checkpoint 目录，含 tokenizer.json> \
    --tasks niah_single_1,niah_single_2,niah_single_3 \
    --max_seq_lengths 32768 \
    --output niah_samples_32k.jsonl
```

只跑一次，跑完把 `niah_samples_32k.jsonl` 当成三个分支共用的固定输入——不要
让任何一个分支的评测重新调用生成器。校准阶段如果需要样本，另外单独跑一次
到不同的输出文件，不要和正式测试集混用（任务说明 §7.1 明确要求两者分离）。

脚本跑完会打印每个 span 来源（`explicit_needle`/`ruler_magic_regex`/
`sf_needle_regex`/`answer_sentence`/`answer_exact`）的命中次数——如果某个
来源命中数是 0，很可能是 RULER 的 prompt 模板变了、`litgpt/needle_spans.py`
里对应的正则/启发式失效了，不是这个任务真的没有 needle，先查再往下走。

**已知未完成的一环（读一遍再往下走）**：这一步导出的样本，目前还没有被
`eval.py` 的真实评测路径直接消费——`eval.py` 的 niah_* 评测仍然是通过
lm-eval-harness 的正常任务机制、调用 RULER 自己的生成器。让 `eval.py`
真正回放这份固定导出文件，需要接管 RULER task 的 `eval_docs` 来源，这需要
一个真实的 `lm_eval` 环境去开发和验证，这次交付的环境里没有，所以没有做，
明确留在这里而不是假装解决了。缓解手段是下面 §六 的 `cross-check`：每次
真实评测跑完后，比对这份导出文件和评测的逐样本输出，如果同一个
`(task_name, doc_id)` 在两边的标准答案对不上，说明 RULER 两次生成的内容
不一样，正式对比不能用这批结果。如果你验证过 RULER 在这两次调用之间实际上
是内容确定的，欢迎跳过这个顾虑，但仍然建议先跑一次 cross-check 确认。

## §五 跑三组训练 + 评测

沿用现成的 `majob.sh`/`eval.sh`，不新增编排层：

```bash
# Dense
bash majob.sh exp/qwen1.7b-32k/dense_stage1_train.yaml

# SinkWindow
bash majob.sh exp/qwen1.7b-32k/sinkwindow_stage1_train.yaml

# SemanticLogKV（先确认好 §二 提到的 config: 指向哪份候选）
bash majob.sh exp/qwen1.7b-32k/semantic_stage1_frozen.yaml
```

`majob.sh` 训练完会自动跑评测；如果单独跑 niah 评测（比如换了固定样本集后
重新评一遍），确保 eval YAML 里带上 `log_samples: true`，否则
`eval.py` 不会落盘逐样本记录（`*_niah_samples.jsonl`），后面 §六的分组分析
和 cross-check 都需要这份文件：

```bash
bash eval.sh exp/qwen1.7b-32k/dense_stage1_eval.yaml niah_single_1,niah_single_2,niah_single_3 none
bash eval.sh exp/qwen1.7b-32k/sinkwindow_stage1_eval.yaml niah_single_1,niah_single_2,niah_single_3 none
bash eval.sh exp/qwen1.7b-32k/semantic_stage1_frozen.yaml niah_single_1,niah_single_2,niah_single_3 none
```

训练开始后 `resolved_config.yaml` 会自动落到每个分支的 `save_path` 下（不需要
额外操作）——这是"最终生效配置"，不是启动用的 YAML；核对配置时读这份文件，
不要读 YAML 或它的注释。

## §六 位置分组分析

对每个分支的评测输出，先跑一次一致性检查（见 §四 的"已知未完成的一环"）：

```bash
python scripts/analyze_niah_visibility.py cross-check \
    --export niah_samples_32k.jsonl \
    --run <该分支的>_niah_samples.jsonl
```

`DIVERGED` 就先停下来查为什么，不要带着不一致的数据继续。确认 `CONSISTENT`
后，做分组分析（示例：SinkWindow vs SemanticLogKV，配对差值 bootstrap CI）：

```bash
python scripts/analyze_niah_visibility.py analyze \
    --export niah_samples_32k.jsonl \
    --run_a sinkwindow_niah_samples.jsonl --label_a SinkWindow \
    --run_b semantic_niah_samples.jsonl --label_b SemanticLogKV \
    --sink_size 4 --window_size <§三定下的W> \
    --score_key <从一条真实样本记录里确认实际的 metric key 名，脚本默认猜
                 exact_match，不一定对> \
    --output visibility_report.json
```

`--score_key` 必须从一条真实的 `*_niah_samples.jsonl` 记录里确认——这个脚本
是在没有 `lm_eval` 的环境里写的，没法验证 RULER niah 任务实际用的 metric 键名，
默认值只是一个合理猜测。

## §七 汇总实验清单

```bash
python scripts/write_experiment_manifest.py \
    --label SinkWindow \
    --checkpoint_dir <该分支 save_path> \
    --niah_export niah_samples_32k.jsonl \
    --niah_samples sinkwindow_niah_samples.jsonl \
    --output manifests/sinkwindow_manifest.json
```

三个分支各跑一次。清单里的 `git.commit`/`git.dirty` 记录的是**跑这个脚本时**
的仓库状态——如果训练用的是更早的一次 commit，请额外记录训练当时的 commit
hash（比如训练日志里打印过，或者你自己单独记的）。

## 交付的代码与工具（这次改动新增/修改的文件）

| 文件 | 作用 |
|---|---|
| `litgpt/needle_spans.py` | 加了 `answer_exact` span（原来只留句子级范围，精确匹配算完就丢了） |
| `litgpt/log_kv_cache.py` | `LogStructuredKVCache.extra_live_tensors()`，暴露 mid-decode 瞬时 workspace 供字节统计工具读 |
| `litgpt/cache_accounting.py` | 通用字节统计（`structural_bytes`/`live_extra_bytes`，按 storage 去重） |
| `litgpt/sinkwindow_cache.py` | SinkWindow cache 实现（sinkwindow 分支） |
| `litgpt/model.py` | Dense/SinkWindow 的训练+推理接入 |
| `demo.py` | resolved-config dump（基准分支）；Dense 训练路由 `log_kv_dense_mode`（dense 分支）；SinkWindow 训练路由 `log_kv_sink_window_mode`（sinkwindow 分支） |
| `eval.py` | `log_samples` 开关 + niah 逐样本 JSONL 落盘（基准分支）；SinkWindow eval 接入 `log_kv_sink_window_mode`（sinkwindow 分支） |
| `scripts/yaml_resolve.py` | 轻量 YAML `config:` 继承解析，供下面几个脚本共用 |
| `scripts/measure_cache_bytes.py` | 见 §二 |
| `scripts/calibrate_sinkwindow_w.py` | 见 §三（sinkwindow 分支） |
| `scripts/export_niah_samples.py` | 见 §四 |
| `scripts/analyze_niah_visibility.py` | 见 §六 |
| `scripts/write_experiment_manifest.py` | 见 §七 |
| `exp/qwen1.7b-32k/dense_stage1_{train,eval}.yaml` | Dense 分支冻结配置（dense 分支） |
| `exp/qwen1.7b-32k/sinkwindow_stage1_{train,eval}.yaml` | SinkWindow 分支冻结配置（sinkwindow 分支） |
| `exp/qwen1.7b-32k/semantic_stage1_frozen.yaml` | SemanticLogKV 分支配置，已确认指向 `arc_semantic_fast.yaml` |
| `tests/test_dense_bypass.py`、`tests/test_sinkwindow_cache.py`、`tests/test_cache_accounting.py`、`tests/test_needle_spans_answer_exact.py`、`tests/test_resolved_config_dump.py`、`tests/test_niah_sample_rows.py`、`tests/test_calibrate_sinkwindow_w.py` | §一的验证脚本，写出来但没跑过 |

## 明确没有解决的事情（不要假装它们已经解决）

1. **`eval.py` 还没有真正回放固定导出的 NIAH 样本**，见 §四。`cross-check` 是
   检测手段，不是预防手段。
2. **0.946/0.204/0.172 这三个历史成绩**在这个仓库里查不到出处（结果目录被
   gitignore、这个容器的 git 历史最早只到 2026-08-27）；按你的要求，本轮
   三组都独立重新训练，这三个数字只作历史背景，不代入正式对比表。
3. 这次交付的所有代码都**只做过静态自查，没有执行过**（这个容器没有 GPU、
   没装 torch）；§一的验证是训练前的强制关卡，不是可选项。
