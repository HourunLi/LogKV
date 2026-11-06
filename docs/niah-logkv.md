# niahLogKV：学习到的保留打分（不可压缩度估计器）

niahLogKV 建立在 attach + Alpha + Beta + Gamma 之上（见 [Gamma](gamma-logkv.md)），不改变路由、梯子
与预算。它只替换精确池选片段时的打分：由每层一个小打分头读片段的 K/V，估计"这段内容被梯子平均后
无法恢复、而后文需要它"的程度。打分不看开头指令（cue）、不看问题（query），也不用手写的字面量规则；
后三者只作为可叠加的附加技巧另行报告。

| 参数 | 默认 | 作用 |
|---|---|---|
| `log_kv_retention_heads` | `null`（关闭） | `train_retention.py` 训出的打分头文件；评测与 CPT 都读它 |
| `log_kv_gamma_surprisal` | `null`（关闭） | 精确池按 token 惊异度给片段打分（`mean`、`top4`、`mix`、`multi` 或词分数 `word_max`、`word_top2`、`word_first`、`multi_word`，见第 7 节；只评测） |
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
- `niah_surprisal.py`：lm-eval 自己的 NIAH 样本（RULER 模板与 `eval_tasks/niah_cue` 的变体，以及多针的
  `niah_multikey_1`、`niah_multivalue`、`niah_multiquery`；多针时按出现顺序逐根报告），按 Alpha 的切法
  切片段，以 max、top4、mean（惊异度）与 uni_max（文档内一元编码长度，不用 LM）为片段分数，报告针所在片段在
  离开 recent 窗口的片段中的名次、装下它需要的精确池 token 数（`tokens_needed`，最终池的离线估计，不含 1.1 倍）、
  在 P=256/1024/2048 下装得下的比例，以及分数比针高的片段。
- `reuse_label_stats.py`：YAML 的训练数据（按 `demo.py` 读）。同一 n-gram 的相邻两次出现 a < b 相距不小于 gap
  （默认 recent 窗口），且该 n-gram 首次出现时的平均惊异度不低于阈值，则 a 处的 n 个 token 为正。按 (n, 阈值)
  报告正 token 占比、含正 token 的片段数与 token 数、能否装进精确池、首次与复用处的惊异度、LogKV 缓存自己的
  选择留住正 token 的比例（对比所有已归档 token），以及正 n-gram 的种类（数字、大写词、代码符号、小写词），并打印例子。
  标签不涉及任何句式或任务。
- 每个脚本都把可读的汇总写到输出文件旁的 `<输出名>.summary.txt`（不依赖终端输出）；`--report <输出文件>` 不加载模型、
  不重放，直接从已有的输出文件重建这份汇总。
- `niah_surprisal.py` 在 LogKV 缓存上还从真实缓存读出：prefill 结束时，被问的每个值在多少比例的层里全部精确（精确池或
  recent 窗口），记为 `exact_layers`，汇总里是 "real cache" 一行；YAML 开了惊异度准入时，惊异度直接取缓存自己用的值。
- 两个脚本加 `--save-surprisal <文件>.npz` 时另存每段序列的 token id 与惊异度（NIAH 还存针与值的位置），并记下
  recent 窗口、flush 粒度、片段上限、精确池与片段边界 token，供下面的离线重放使用。

**精确池策略的离线重放**（`unused/pool_policy_sim.py`，只用 CPU、不加载模型）读上面的 .npz，按缓存的节奏重放：
token 逐个进入，攒满 recent + flush 后最早的 flush 个离开 recent 窗口，完全离开的候选片段与池中片段一起交给缓存自己的
`choose_spans`（按价值密度整段换入，池中片段乘 1.1）装满 P，没装下的即并入梯子、不再回来。价值密度

  v(T) = D(I + β·M(T)) · g(T − last(T))

D 与缓存相同：token 分数与 z 分数为 16^分数，sum 为 2^比特，词分数为整段价值 2^比特 再除以片段长度（`per_span: false`
时不除，即此前缓存按每 token 计的做法）。

- I：准入时的信息量，片段的惊异度分数（mean、top4、topq 即最高四分之一的均值、max、mix、sum），词分数（`word_max`：
  片段中编码长度最大的词的编码长度，词与编码长度见第 7 节；`word_top2`：最大两个的均值；`word_first`：同 word_max，但每个词按它在本序列首次出现时的
  编码长度计，再次提到罕见实体的句子保留其分量），多维 z 分数
  （`multi`：mean 与 top4；`multi_word`：mean 与 word_max），或恒为 1（只看新近的基准）。
- g：遗忘，none、幂律 (1 + age/τ)^(−α) 或指数 exp(−age/τ)。
- last(T)：片段末尾；打开复用时取 T 之前它的"有信息的 n-gram"（片段内惊异度不低于 `--reuse-bits` 的 n-gram）
  最近一次再出现的位置（刷新）。M(T)：复用加分时这些再出现的惊异度之和。
- 全部因果：T 时的价值只用 T 之前的 token。默认网格为 3 种分数 × 6 种遗忘 × 3 种复用（无、刷新、加分）再加新近基准
  与 mix、multi、sum、word_max、word_top2、word_first、multi_word 七种分数，共 62 种策略。
- 词分数要用 tokenizer 判断词首（默认按 .npz 里记下的
  配置找，或用 `--tokenizer` 指定）；`--space-words` 改回只在空白处分词的旧规则以作对照。
- `--explain K` 另写 `<输出名>.explain.txt`：每种策略、每个任务前 K 个丢针的样本，列出针所在片段与最终池中分数最高的
  片段（文本、长度、各分数、在已归档片段中的名次）。汇总首行记下代码的 commit。

报告每种策略：NIAH 提示结束时针的值仍为精确的比例（按任务，以及针在前三分之一的样本）；自然块上远距离复用的
命中率（`reuse_label_stats.py` 的标签：b 读入时 a 处的那份是否仍为精确），单块与 `--concat` 块拼接成的长流；
池中 token 所在片段在保留期间被复用过的比例。

```bash
python unused/pool_policy_sim.py --niah <niah>.npz --natural <natural>.npz --concat 4 --out <输出>.json
```

## 7. 惊异度准入（只评测）

`log_kv_gamma_surprisal: mean | top4 | mix | multi | word_max | word_top2 | word_first | multi_word` 让精确池按片段 token 的惊异度排序：token t 的惊异度是模型自己对它的编码长度
−log2 p(ids[t] | ids[:t])（比特），片段分数取其均值（`mean`）、最大 4 个的均值（`top4`）、两者的平均（`mix`），或多维度（`multi`：两者各自相对本条
序列已归档片段的均值与标准差换算成 z 分数，标准差下限 0.25 比特，片段按更突出的一维计分），价值密度为
`SURPRISAL_SCALE`（1e4）× `SURPRISAL_BASE`（16）^ min(分数, 32)，加到 K/V 分数上进入原有的整段贪心（含旧片段 1.1 倍）。
1e4 让惊异度主导排序，K/V 分数只区分惊异度相同的片段；指数让整段换入跟随惊异度的顺序：换入要求"密度 × 长度"
超过被换出片段之和，线性价值下 20 token 的针换不掉一个惊异度一半、64 token 的片段，每比特 16 倍时惊异度高 1.5 比特
的片段即可抵过 64 倍长度。不看开头、不看问题、不训练，不增加 KV。

- 惊异度来自模型自己的 logits：开启时 `GPT.forward` 把带缓存的输入按不超过 recent 窗口与 flush 粒度的块（默认 2048）
  逐块前向，每块之后把该块各 token 的惊异度交给 `SurprisalStream`（块首 token 用上一块或上一步的末位 logits）。
  flush 只归档更早块里的 token，所以被归档的 token 的惊异度总已在流里。逐 token 解码同样逐步喂入。
- 代价：每块对所有位置算一次 lm_head（Qwen3-1.7B 上约为预填充计算的两成），主机上每个 token 存一个浮点数（词分数再加 token id，
  `word_first` 再加一张"词 → 首次编码长度"的表）；KV 不变。
- 词分数（`word_max`、`word_top2`、`word_first`、`multi_word`）按片段里信息量最大的词计分。词按词法切分（`word_starts`，由
  tokenizer 各 token 单独解码的文本得到每个 token 首末字符的类别）：字母与数字连成一个词，词内可含连字符、撇号、下划线
  （"mid-century"、"don't"、UUID），数字之间的 "." "," 也算词内（"3.14"）；空白、其余标点、每个 CJK 字符以及片段首都开始
  新词。所以 URL、路径、标记是若干个词，不会因为不含空格就成了一个越长分越高的"词"（只按空白切分时，一条链接所在的
  64 token 句子以 85–240 比特排在所有针之前，十几个这样的句子就占满 P=1024）。词的编码长度是其
  token 惊异度之和 −log2 p(词 | 上文)，与分词方式无关：数字逐位切分，每位最多 log2 10 ≈ 3.3 比特，按 token 平均或峰值
  会输给单个罕见词，按词算一个 7 位随机数约 23 比特、一个 UUID 上百比特，而普通词只有几比特。`word_max` 取片段中最长的词，
  `word_top2` 取最长两个词的均值（例如键与值），`word_first` 同 `word_max`，但每个词按它在本序列首次出现（相同 token id）
  时的编码长度计：罕见的名字读过一次后模型就能照抄，但再次提到它并带来新值的句子（一个键的多个值、表格的一行行记录）
  与第一次同样关乎它；查找是因果的，按片段离开 recent 窗口的先后登记，各层读到相同的值。整段价值为 `SURPRISAL_SCALE` × 2^min(比特, 256)，即该词的逆概率，价值密度再除以片段长度（按每 token 计时，64 token 的片段
  算作 64 倍，22 token 的针要长约 1.7 比特才换得掉它）；`multi_word`
  与 `multi` 相同，只是两维为 mean 与 word_max。片段仍按句切，值与问它的词留在同一片段。模型从 token 文本建词首表
  （`set_alpha_span_boundary_ids(..., literal_texts=...)`，`eval.py` 开启词分数时自动传入），每块前向后连同惊异度交给流。
  GPU（P=1024，空白分词）：`word_first` 单针 1 / 1 / 0.968，multikey_1 / multivalue / multiquery 0.866 / 0.871 / 0.801，words
  0.957（dense CPT 0.966 / 0.883 / 0.885，words 0.885）；`word_max` 的 multivalue 只有 0.40：键已可照抄，第 2–4 个值的句子只剩
  22–26 比特的数值。真实缓存中被问的值是否精确（`niah_surprisal.py` 的 real cache）与 GPU 准确率一致，用真实运行存下的惊异度
  重放也一致；用 K/V 准入缓存上算的惊异度重放则高估（0.97），因为精确留下第一条记录后，同格式的后续记录变得可预测——
  多针的离线判断要用对应准入真实运行的 .npz。挤掉这些针的是带链接的句子，这促成了上面的词法分词：用该次运行的惊异度重放，
  multikey_1 / multivalue / multiquery 由空白分词的 `word_max` 0.75 / 0.41 / 0.88、`word_first` 0.75 / 1.00 / 0.88 变为词法分词的
  0.97 / 0.54 / 0.98 与 0.91 / 1.00 / 0.96，单针仍全为 1.00。
  把值单独切成片段的做法（信息片段）已试过并删去：脱离句子后，数值的平均每 token 惊异度低于孤立的罕见词，
  总量又低于连成串的普通实词，单针 s2 降到 0.16。
- 只用于评测：训练（`enable_log_kv_training`）与 CPT 不支持。`eval.py` 与 `eval.sh` 在 YAML 设置时传
  `--log_kv_gamma_surprisal`；关 Alpha 时一并关闭。结果 JSON 记录该开关。
- 配置：[`eval_gamma_surprisal_p1024.yaml`](../exp/qwen1.7b-32k/eval_gamma_surprisal_p1024.yaml)（`top4`）与
  [`eval_gamma_surprisal_mean_p1024.yaml`](../exp/qwen1.7b-32k/eval_gamma_surprisal_mean_p1024.yaml)（`mean`）与
  [`eval_gamma_surprisal_mix_p1024.yaml`](../exp/qwen1.7b-32k/eval_gamma_surprisal_mix_p1024.yaml)（`mix`）、
  [`eval_gamma_surprisal_multi_p1024.yaml`](../exp/qwen1.7b-32k/eval_gamma_surprisal_multi_p1024.yaml)（`multi`）、
  [`eval_gamma_surprisal_word_max_p1024.yaml`](../exp/qwen1.7b-32k/eval_gamma_surprisal_word_max_p1024.yaml)（`word_max`）、
  [`eval_gamma_surprisal_word_top2_p1024.yaml`](../exp/qwen1.7b-32k/eval_gamma_surprisal_word_top2_p1024.yaml)（`word_top2`）、
  [`eval_gamma_surprisal_word_first_p1024.yaml`](../exp/qwen1.7b-32k/eval_gamma_surprisal_word_first_p1024.yaml)（`word_first`）与
  [`eval_gamma_surprisal_multi_word_p1024.yaml`](../exp/qwen1.7b-32k/eval_gamma_surprisal_multi_word_p1024.yaml)（`multi_word`），
  以及技巧单列的 `eval_gamma_surprisal_{mean,mix,word_first}_query_p1024.yaml`（再加查询感知 cue 与 cue 保护，两者的价值相加），
  都在不带 cue 的 P=1024 checkpoint（`arc_gamma_k12_b128_p1024_2k`）上，与它自身的评测只差这一项。

测试：`tests/test_surprisal_admission.py`（片段价值、分块前向的惊异度与返回的 logits、逐 token 解码一致、惊异片段进入精确池、
短而惊异的片段换掉长而平淡的片段、词分数按词总计编码长度、词首表来自 token 文本），
`tests/test_retention.py`（启动脚本的配置读取与 checkpoint 检查、选项解析与续训、片段价值公式与设备相加、新头不改变选择、prefill 与逐 token 一致、
能识别针的头把针留在精确池、校验与保存/加载、回放签名、合成事实的版式与种类、训练器能分开正负样本、
`demo.py` 的混入），`tests/test_exp_configs.py`（niah 配置只差所述开关），
`tests/test_semantic_eval_flags.py`（启动脚本只在设置时传头文件），`tests/test_niah_cue_tasks.py`（words 针），
`tests/test_surprisal_diag.py`（惊异度与逐位 log-softmax 一致、recent 窗口内缓存与不缓存一致、片段切分、针的名次、
复用对与阈值、`diag.sh`、惊异度存档的读写、离线重放的准入、遗忘与复用刷新、远距离复用命中）。

## 8. 长文本评测（64K / 128K，只评测）

模型与 CPT 都只到 32K。更长的评测不重训，用 YaRN 把 RoPE 拉长（factor = 目标长度 / 32768，原长 32768，beta_fast 32，
beta_slow 1，与 Qwen 的 `rope_scaling` 相同），dense 与 LogKV 用同一套，比较的是压缩而不是外推。

- YaRN：litgpt 原实现插值方向相反（对高频维插值、保留低频维），且默认不乘注意力温度；已按 transformers 的
  `_compute_yarn_parameters` 修正：低于 beta_fast 转数的高频维保持原频率，高于 beta_slow 的低频维插值，温度为
  `attention_factor`，否则 mscale / mscale_all_dim，否则 0.1·ln(factor)+1。LogKV 缓存的位置编码全部来自模型的 cos/sin 表，
  所以 YaRN 同样作用于精确池与梯子的期望 RoPE。
- 稠密 KV 缓存不再分配 max_seq_length × max_seq_length 的掩码（128K 时约 17 GB）：从位置 0 开始的预填充直接是因果注意力
  （SDPA 的 is_causal），只读已写入的那部分缓存；其余调用（解码、续接的分块）按需建 T × (最大位置+1) 的掩码。
- `eval.sh` 从 YAML 读：`eval_max_seq_lengths`（RULER / NIAH 的长度，默认 1K–32K）、`niah_benchmark`（NIAH 任务列表，命令行第
  三个参数优先）、`benchmark`、`limit`（每个任务的样本数）、`config_overrides`（`block_size`、`rope_adjustments`，以 JSON
  传给 `eval.py`）。`majob.sh` 只读 `eval_max_seq_lengths`。
- 任务：完整 RULER 13 个任务（8 个捞针，加变量追踪 vt、常见词与高频词提取 cwe / fwe、SQuAD 与 HotpotQA 问答）在
  65536 / 131072：捞针、vt、cwe、fwe 是合成任务，草堆仍是 Paul Graham 文章；两个问答任务运行时从 `rajpurkar.github.io`
  与 `curtis.ml.cmu.edu` 取 SQuAD / HotpotQA 的 dev 集。长文档困惑度（`eval_tasks/long_ppl`，`long_ppl_{32k,64k,128k}`）：
  取不短于 131072 token 的文档（按文件与行的顺序，所以各长度是同一批文档），截到 L，给定前 L−4096 个 token，计末尾
  4096 个 token 的困惑度（`ppl`）与每字节比特（`bits_per_byte`）；它衡量的是大部分上下文已在身后（LogKV 中已压缩）
  时的预测，不受任务难度的地板影响。数据由 YAML 的 `long_ppl_data` 或环境变量 `LONG_PPL_DATA` 给出（parquet / jsonl
  文件或目录，正文列 text / content / code），`LONG_PPL_DOCS` 限制文档数（默认 100），`LONG_PPL_CONCAT=1` 把同一文件的
  相邻行拼成一篇（行是长文的片段时），选出的文档缓存在 `LONG_PPL_CACHE`（默认 `$HF_HOME/logkv_long_ppl`，各节点共用）。训练读取
  `data_dir` 下的全部 parquet，所以只有不在其中的文件才是严格留出的。BABILong qa1–qa5
  （`eval_tasks/babilong_long`，任务名带长度：`babilong_qa1_64k` 等，组 `babilong_{32k,64k,128k}`；提示、示例与打分同 lm-eval
  的 `babilong_qaN`；数据 `RMT-team/babilong-1k-samples`，环境变量 `BABILONG_DATASET` 可指向本地副本）；LongBench v2
  （lm-eval 的 `longbench2`，数据 `recursal/longbench-v2`，四选一按似然打分，每题四次预填充；长于 block_size 的上下文
  从开头截断，保留问题。数据集卡片没有按子集的配置时，`litgpt/longbench2_patch.py` 改从仓库快照里按文件名读出该子集）。
- 数据：`eval.sh` 离线运行（`HF_HUB_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`），新数据要先在联网时放进它读的缓存：
  `bash diag.sh unused/fetch_long_eval_data.py [--longbench2] [--ppl-data <留出数据> --tokenizer <tokenizer 目录>]`
  用与 `eval.sh` 相同的 `HF_HOME` / `HF_DATASETS_CACHE` / `RULER_CACHE_DIR`，只为自己打开联网，下载 BABILong（32k / 64k / 128k）、
  RULER 问答的 SQuAD 与 HotpotQA（`litgpt/ruler_patch.py` 先读的本地文件）与可选的 LongBench v2，再以离线方式重新加载
  一遍；给了 `--ppl-data` 时还选出困惑度的文档并写入它的缓存（缓存键按 tokenizer 文件内容，与用哪个路径指向它无关）。
  平台代理用自签证书重签 HTTPS，脚本默认不校验证书（同 `litgpt/ruler_patch.py`），下载只走 requests、不用 hf_transfer / hf_xet，
  `--verify-ssl` 恢复校验；Hub 下载的读超时默认放宽到 300 秒（`HF_HUB_DOWNLOAD_TIMEOUT` 可覆盖）。hf-mirror 的共享 Hub 限额
  用完时返回 429，每个请求按 `Retry-After`（没有时 60 秒）等待后重试，最多 10 次。
- 配置：`eval_long{64k,128k}_{dense,baseline,word_first}.yaml`，分别是 dense CPT、同 checkpoint 的 K/V 准入基线与 `word_first`，
  与各自的基础配置只差上面几项：NIAH 阶段跑完整 RULER，主阶段跑 BABILong 与困惑度（64K 配置为 32K 与 64K，128K 配置为
  128K），每个任务 100 条。`babilong_dataset` 与 `long_ppl_data` 两个键（或同名环境变量）给出本地数据。
- 汇总表（`unused/parse_lmeval_table.py`）：RULER 13 个任务的 `<任务>@64k`、`AVG_NIAH@64k`、`AVG_NIAH_multi@64k`、
  `AVG_RULER@64k`（128k 同）、`babilong_qaN_<长度>`、`AVG_babilong_<长度>`、`long_ppl_<长度>`（越低越好）、`LongBench2`，
  都不计入 `AVG_all`。

测试：`tests/test_yarn.py`（YaRN 的 cos/sin 与 transformers 一致），`tests/test_long_context.py`（YaRN 进入 LogKV 缓存且 recent
窗口内与不缓存一致、启动脚本读长度/任务/覆盖/样本数/数据路径、BABILong 按长度加载、困惑度任务取同一批长文档的末尾、下载脚本预建困惑度缓存、稠密缓存无方阵掩码且
与不缓存一致），
`tests/test_exp_configs.py`（长文本配置只差长度相关的几项）。
