# niahLogKV：学习到的保留打分（不可压缩度估计器）

niahLogKV 建立在 attach + Alpha + Beta + Gamma 之上（见 [Gamma](gamma-logkv.md)），不改变路由、梯子
与预算。它只替换精确池选片段时的打分：由每层一个小打分头读片段的 K/V，估计"这段内容被梯子平均后
无法恢复、而后文需要它"的程度。打分不看开头指令（cue）、不看问题（query），也不用手写的字面量规则；
后三者只作为可叠加的附加技巧另行报告。

| 参数 | 默认 | 作用 |
|---|---|---|
| `log_kv_retention_heads` | `null`（关闭） | `train_retention.py` 训出的打分头文件；评测与 CPT 都读它 |
| `synthetic_retrieval_prob` | `0.0` | CPT 时训练块插入合成事实与问答的概率（只在 `demo.py`） |
| `synthetic_facts` / `synthetic_questions` / `synthetic_kinds` | `2,8` / `1,3` / 全部 | 每块事实数、问题数与值的种类 |

关 Alpha（`log_kv_alpha_exact_tokens: 0`）时打分头一并关闭。`arc_semantic_fast.yaml` 显式写出两项默认值，
已有实验不受影响。

## 1. 打分头

- 输入：候选片段每个 token 的 pre-RoPE K 与 V（所有 KV group 拼接，各自做 RMS 归一化），即精确池会保留的
  那份 K/V；与位置无关，逐层整段 prefill 与逐 token decode 看到同样的输入。
- 结构：每层一个 `RMSNorm[K | V] → Linear(2·G·D, 64) → GELU → Linear(64, 1)`，输出 token 的 logit
  （Qwen3-1.7B：每层约 13 万参数）。片段的 logit 取其 token 的最大值。
- 价值：`max(exp(min(z, 12)) − exp(0), 0) / 片段长度`，与 cue、字面量同形，加到片段的 K/V 分数上，进入
  原有的整段贪心（含旧片段 1.1 倍）。logit 低于 0 的片段只用 K/V 分数；新建的头偏置为 −4，选择与不用头相同。
- 成本：打分在选择已有的那次设备计算里完成（CUDA 融合核路径与 PyTorch 路径都在设备上相加），每次 flush
  仍只读回每个片段一个分数；不增加 KV，推理 KV 仍为 `O(log N)`。分数预取路径同样带头。
- 打分头不是 LM 的子模块：不进 LM 的 state dict、不随 FSDP 分片、CPT 时冻结。文件
  （`litgpt.retention.RetentionHeads`）记录形状、权重和元数据，`tag()` 是权重摘要；开关与 tag 都写入
  回放签名，反向与 checkpoint 重算只回放前向的选择，不重新打分。

## 2. 合成检索数据（训练标签）

`litgpt/synthetic_retrieval.py` 在一块训练 token 中随机插入事实，并在末尾追加对其中一部分的问答：

- 事实 = key + value。key 是两个词（连字符或空格连接）或"人名's 词"；value 的种类为 number（3–10 位）、
  decimal、code（字母-数字或十六进制串）、uuid、word（1–2 个词）、name、date。词与人名取自 tokenizer
  自己的单 token 词表。
- 14 种事实句式、5 种问答句式，均不含 RULER 的措辞（special magic、One of the、mentioned in the provided
  text），NIAH 评测的句子训练中未见过。
- 事实插在随机位置之后最近的句末（256 个 token 内），块长不变：去掉干草堆自身的尾部腾出位置。
  返回的掩码标出事实句的 token，作为打分头的正样本。

## 3. 训练打分头

训练环境用 `retention.sh` 启动（环境激活、HF/NCCL 变量、GPU 与多机检测同 `majob.sh`）。同一份评测 YAML
先训练头、再评测；头写到 YAML 的 `log_kv_retention_heads`（或 `retention_out`）：

```bash
bash retention.sh exp/qwen1.7b-32k/eval_niah_heads_p1024.yaml
NIAH_ONLY=1 bash eval.sh exp/qwen1.7b-32k/eval_niah_heads_p1024.yaml
```

- 启动前检查：YAML 的 `save_path`（或 `retention_checkpoint`）下要有 `lit_model.pth` 或
  `step_*/lit_model.pth`，否则提示先用 `majob.sh` 训练该 checkpoint；要有输出路径，且开着 Alpha。
- 选项都是 YAML 键 `retention_*`：`steps`（2000）、`lr`（1e-3）、`hidden`（64）、`length`（32768）、
  `facts`（`2,8`）、`questions`（`1,3`）、`kinds`（全部）、`val_docs`（16）、`val_every`（200）、
  `save_every`（50）、`seed`（1234）、`init`、`checkpoint`、`out`、`resume`（true）；`RETENTION_ARGS='--steps 40'` 之类在命令行覆盖。
- 断点续训：每 `retention_save_every` 步（以及每次验证和最后一步）rank 0 写两个文件：`<out>` 只有头（评测与 CPT
  读它），`<out>.state` 是完整训练状态（头、AdamW 的矩估计、正样本权重的累计计数）。中断后用同一条命令重跑，
  从步数最大的那个继续（同步数优先 `.state`；只有 `<out>` 时只恢复头）；学习率取当前配置，可在续训时调整。
  续训的训练块换一个打乱种子（种子 + 已训步数）重新抽取，不重放中断前的顺序；插入的事实只由种子、rank 与步数
  决定。已训到 `retention_steps` 的头直接跳过，调大 `retention_steps` 即接着训。启动时打印从哪个文件的第几步继续，
  或从 0 开始的原因；`<out>` 或 `<out>.state` 存在却读不出时停下报错，不会从 0 重训并覆盖它。
- 每个文件都先写 `.tmp` 再替换；OBS 桶挂载不允许重命名时，把写完的 `.tmp` 复制过去后再删除，续训也读能完整
  读出的 `.tmp`。
- tokenizer 与 `eval.sh` 一样依次找 `tokenizer_dir`、`resume_dir`、`ckpt_dir`、`checkpoints/<arch>`。
- 开发环境也可直接 `torchrun --nproc_per_node=N train_retention.py --config <YAML>`。

- 按 YAML 构建与 `eval.py` 相同的模型与缓存（LM 冻结），按 `demo.py` 的方式读 YAML 的训练数据；每步取一块
  （默认 32768 token），插入合成事实，带着正在训练的头整段 prefill。
- 每次 flush 的选择把候选片段的 K/V 交给训练器（`litgpt.retention_train.HeadTrainer`），只取本次新归档的
  片段（每个 token 每层只计一次），含事实 token 的片段为正样本。损失是带正样本权重（负/正样本数之比，
  上限 200）的二元交叉熵，只更新头（K/V 截断梯度）；各 rank 取不同的块，平均头的梯度。
- 选择一直使用正在训练的头，深层看到的是推理时会看到的 K/V。
- 每 `retention_val_every` 步 rank 0 在固定的留出块上报告每层把事实 token 留在精确（精确池或 recent）的比例
  `kept_per_layer`。第 0 步的数字就是只用 K/V 分数时的比例。留出块与训练块取自同一份语料
  （不同的打乱种子），插入的事实每块都是新生成的。
- `retention_kinds` 限定值的种类（例如去掉 `uuid`，再看 niah_single_3 是否泛化）；`retention_init` 从已有的头开始。

## 4. 评测与 CPT

- 只评测：[`eval_niah_heads_p1024.yaml`](../exp/qwen1.7b-32k/eval_niah_heads_p1024.yaml) 在不带 cue 的
  P=1024 梯子 checkpoint（`arc_gamma_k12_b128_p1024_2k`）上打开上面训出的头，与该 checkpoint 自身的评测
  只差这一项。`eval.py`、`eval.sh`、`majob.sh` 只在 YAML 设置路径时传 `--log_kv_retention_heads`；结果 JSON
  记录路径与 tag。文件不存在时 `eval.sh`、`majob.sh` 在加载模型前停下，并给出写出该路径的
  `eval_niah_*.yaml` 的 `retention.sh` 命令（每个 `eval_niah_*.yaml` 训练、评测的都是它自己路径上的头）。
- CPT：[`arc_niah_heads_k12_b128_p1024_slack0_2k.yaml`](../exp/qwen1.7b-32k/arc_niah_heads_k12_b128_p1024_slack0_2k.yaml)
  为 P=1024 + `level_slack: 0`（B=128 预算下 B′=203、4 层），不开 cue、query、字面量；精确池选择用冻结的头，
  训练块以 0.5 的概率插入合成事实与问答。对照
  [`arc_niah_synth_k12_b128_p1024_slack0_2k.yaml`](../exp/qwen1.7b-32k/arc_niah_synth_k12_b128_p1024_slack0_2k.yaml)
  数据与结构相同，只是不用头。`demo.py` 把头文件复制到 `save_path/retention_heads.pth`。
- CPT 之后 K/V 分布变了，在新 checkpoint 上从 CPT 用的头出发重训 500 步（YAML 里的 `retention_init`、
  `retention_steps`），再评测：

```bash
BENCHMARKS=none NIAH_BENCHMARKS=none bash majob.sh exp/qwen1.7b-32k/arc_niah_heads_k12_b128_p1024_slack0_2k.yaml
bash retention.sh exp/qwen1.7b-32k/eval_niah_heads_refit_p1024_slack0.yaml
NIAH_ONLY=1 bash eval.sh exp/qwen1.7b-32k/eval_niah_heads_refit_p1024_slack0.yaml
```

报告口径：基准是同一梯子、同一 checkpoint 有无打分头；cue、query（保护 + 问题）在评测 YAML 里另行打开，
作为附加技巧单列。不带 cue 的 P=1024 checkpoint（`arc_gamma_k12_b128_p1024_2k`）上的只评测 YAML，都与该
checkpoint 自身的评测只差所列开关：

| YAML | 开关 | 作用 |
|---|---|---|
| `arc_gamma_k12_b128_p1024_2k.yaml` | 无 | 基准：只用 K/V 分数 |
| `eval_niah_heads_p1024.yaml` | 打分头 | 学到的估计器 |
| `eval_niah_heads_heldout_p1024.yaml` | 打分头（训练时不含 uuid、word 值） | 泛化：niah_single_3 与 words 针 |
| `eval_gamma_literal_p1024.yaml` | 字面量 | 手写估计器（只认数字串） |
| `eval_gamma_query_p1024.yaml` | query + 保护 | 技巧：问题；与"打分头 + query"同 checkpoint 对照 |

cue 与 cue + query 的技巧结果取自带 cue 训练的 checkpoint（`eval_gamma_cue_guard_p1024_slack0.yaml`、
`eval_gamma_query_p1024_slack0.yaml` 等，见 [Gamma](gamma-logkv.md)），报告时注明 checkpoint 不同。
上限是 dense CPT 的模型（`cpt_dense.yaml`），不是压缩 checkpoint 的 dense 模式。

**泛化任务。** `eval_tasks/niah_cue` 另有三个不含数字的针：`niah_single_2_words`（RULER 模板）、
`_words_qend`、`_words_paraphrase`，值是一个随机英文词，手写字面量规则对它们不起作用。本分支的
`eval.sh` 与 `majob.sh` 默认跑它们，NIAH 阶段共 20 个任务；`unused/parse_lmeval_table.py` 汇总为
`AVG_NIAH_words`（32K，不计入 `AVG_all`）。

## 5. 限制

- 监督来自合成事实：只有 14 个固定句式、3 种键形式，事实整句标为正样本，头可以靠句式与值的格式认出事实；
  合成验证块与训练块同分布，`kept_mean` 衡量不了泛化。RULER 的针（连字符键、"is: 值"）与部分模板、键形式相近，
  在 RULER 上的结果也不能作为泛化证据。
- 不看问题时，精确池容量决定上限：全是同格式针的干草堆（`niah_multikey_2/3`）里几百根针同样需要保留，
  P=1024 装不下，也分不出被问的是哪一根。
- 头依赖它训练时的 checkpoint 的 K/V；换 checkpoint（包括 CPT 之后）需要重训，文件 tag 用来核对。

## 6. 诊断：惊异度与自然复用

两个只读诊断脚本，不训练、不改变任何运行，用 `diag.sh`（`majob.sh` 的环境，单进程单卡）启动：

```bash
bash diag.sh unused/niah_surprisal.py --config exp/qwen1.7b-32k/arc_gamma_k12_b128_p1024_2k.yaml \
    --tasks niah_single_2,niah_single_2_words,niah_single_2_words_qend --samples 32 --out <输出>.jsonl
bash diag.sh unused/reuse_label_stats.py --config exp/qwen1.7b-32k/arc_gamma_k12_b128_p1024_2k.yaml \
    --blocks 16 --out <输出>.json
```

- token 的惊异度 s[t] = −log2 p(ids[t] | ids[:t])（比特），是模型对它的编码长度。两个脚本都按 YAML 构建与 `eval.py`
  相同的模型与缓存，整段 prefill 后从最终隐状态分块算出（不保留整段 logits）。
- `niah_surprisal.py`：lm-eval 自己的 NIAH 样本（RULER 模板与 `eval_tasks/niah_cue` 的变体），按 Alpha 的切法
  切片段，以 max、top4、mean（惊异度）与 uni_max（文档内一元编码长度，不用 LM）为片段分数，报告针所在片段在
  离开 recent 窗口的片段中的名次、装下它需要的精确池 token 数（`tokens_needed`，最终池的离线估计，不含 1.1 倍）、
  在 P=256/1024/2048 下装得下的比例，以及分数比针高的片段。
- `reuse_label_stats.py`：YAML 的训练数据（按 `demo.py` 读）。同一 n-gram 的相邻两次出现 a < b 相距不小于 gap
  （默认 recent 窗口），且该 n-gram 首次出现时的平均惊异度不低于阈值，则 a 处的 n 个 token 为正。按 (n, 阈值)
  报告正 token 占比、含正 token 的片段数与 token 数、能否装进精确池、首次与复用处的惊异度、LogKV 缓存自己的
  选择留住正 token 的比例（对比所有已归档 token），以及正 n-gram 的种类（数字、大写词、代码符号、小写词），并打印例子。
  标签不涉及任何句式或任务。

测试：`tests/test_retention.py`（启动脚本的配置读取与 checkpoint 检查、选项解析与续训、片段价值公式与设备相加、新头不改变选择、prefill 与逐 token 一致、
能识别针的头把针留在精确池、校验与保存/加载、回放签名、合成事实的版式与种类、训练器能分开正负样本、
`demo.py` 的混入），`tests/test_exp_configs.py`（niah 配置只差所述开关），
`tests/test_semantic_eval_flags.py`（启动脚本只在设置时传头文件），`tests/test_niah_cue_tasks.py`（words 针），
`tests/test_surprisal_diag.py`（惊异度与逐位 log-softmax 一致、recent 窗口内缓存与不缓存一致、片段切分、针的名次、
复用对与阈值、`diag.sh`）。
