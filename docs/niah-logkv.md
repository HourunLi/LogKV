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

```bash
torchrun --nproc_per_node=8 train_retention.py \
  --config exp/qwen1.7b-32k/arc_gamma_k12_b128_p1024_2k.yaml \
  --out /home/ma-user/work/bucket-pangu-green/lihourun/ckpt-logKV/retention_heads/gamma_k12_b128_p1024.pth
```

- 按 YAML 构建与 `eval.py` 相同的模型与缓存（LM 冻结），按 `demo.py` 的方式读 YAML 的训练数据；每步取一块
  （默认 32768 token），插入合成事实，带着正在训练的头整段 prefill。
- 每次 flush 的选择把候选片段的 K/V 交给训练器（`litgpt.retention_train.HeadTrainer`），只取本次新归档的
  片段（每个 token 每层只计一次），含事实 token 的片段为正样本。损失是带正样本权重（负/正样本数之比，
  上限 200）的二元交叉熵，只更新头（K/V 截断梯度）；各 rank 取不同的块，平均头的梯度。
- 选择一直使用正在训练的头，深层看到的是推理时会看到的 K/V。
- 每 `--val-every` 步 rank 0 在固定的留出块上报告每层把事实 token 留在精确（精确池或 recent）的比例
  `kept_per_layer`，并保存头。第 0 步的数字就是只用 K/V 分数时的比例。
- `--kinds` 限定值的种类（例如去掉 `uuid`，再看 niah_single_3 是否泛化）；`--init` 从已有的头继续训练。

## 4. 评测与 CPT

- 只评测：[`eval_niah_heads_p1024.yaml`](../exp/qwen1.7b-32k/eval_niah_heads_p1024.yaml) 在不带 cue 的
  P=1024 梯子 checkpoint（`arc_gamma_k12_b128_p1024_2k`）上打开上面训出的头，与该 checkpoint 自身的评测
  只差这一项。`eval.py`、`eval.sh`、`majob.sh` 只在 YAML 设置路径时传 `--log_kv_retention_heads`；结果 JSON
  记录路径与 tag。
- CPT：[`arc_niah_heads_k12_b128_p1024_slack0_2k.yaml`](../exp/qwen1.7b-32k/arc_niah_heads_k12_b128_p1024_slack0_2k.yaml)
  为 P=1024 + `level_slack: 0`（B=128 预算下 B′=203、4 层），不开 cue、query、字面量；精确池选择用冻结的头，
  训练块以 0.5 的概率插入合成事实与问答。对照
  [`arc_niah_synth_k12_b128_p1024_slack0_2k.yaml`](../exp/qwen1.7b-32k/arc_niah_synth_k12_b128_p1024_slack0_2k.yaml)
  数据与结构相同，只是不用头。`demo.py` 把头文件复制到 `save_path/retention_heads.pth`。
- CPT 之后 K/V 分布变了，在新 checkpoint 上用 `--init` 重训头，再用
  [`eval_niah_heads_refit_p1024_slack0.yaml`](../exp/qwen1.7b-32k/eval_niah_heads_refit_p1024_slack0.yaml) 评测：

```bash
torchrun --nproc_per_node=8 train_retention.py \
  --config exp/qwen1.7b-32k/arc_niah_heads_k12_b128_p1024_slack0_2k.yaml --steps 500 \
  --init /home/ma-user/work/bucket-pangu-green/lihourun/ckpt-logKV/retention_heads/gamma_k12_b128_p1024.pth \
  --out /home/ma-user/work/bucket-pangu-green/lihourun/ckpt-logKV/retention_heads/niah_heads_k12_b128_p1024_slack0.pth
NIAH_ONLY=1 bash eval.sh exp/qwen1.7b-32k/eval_niah_heads_refit_p1024_slack0.yaml
```

报告口径：基准是同一梯子、同一 checkpoint 有无打分头；cue、query（保护 + 问题）在评测 YAML 里另行打开，
作为附加技巧单列。

**泛化任务。** `eval_tasks/niah_cue` 另有三个不含数字的针：`niah_single_2_words`（RULER 模板）、
`_words_qend`、`_words_paraphrase`，值是一个随机英文词，手写字面量规则对它们不起作用；不在默认的 NIAH
任务列表里，需显式给出：

```bash
NIAH_ONLY=1 bash eval.sh exp/qwen1.7b-32k/eval_niah_heads_p1024.yaml none \
  niah_single_2_words,niah_single_2_words_qend,niah_single_2_words_paraphrase
```

## 5. 限制

- 监督来自合成事实：头学的是"像可被查询的具体细节"的片段，句式与值的种类有限；要用未见过的句式（RULER）、
  未训练的值种类（`--kinds` 留出）和 words 针检验泛化。
- 不看问题时，精确池容量决定上限：全是同格式针的干草堆（`niah_multikey_2/3`）里几百根针同样需要保留，
  P=1024 装不下，也分不出被问的是哪一根。
- 头依赖它训练时的 checkpoint 的 K/V；换 checkpoint（包括 CPT 之后）需要重训，文件 tag 用来核对。
- 打分头的效果尚无 GPU 评测结果。

测试：`tests/test_retention.py`（片段价值公式与设备相加、新头不改变选择、prefill 与逐 token 一致、
能识别针的头把针留在精确池、校验与保存/加载、回放签名、合成事实的版式与种类、训练器能分开正负样本、
`demo.py` 的混入），`tests/test_exp_configs.py`（niah 配置只差所述开关），
`tests/test_semantic_eval_flags.py`（启动脚本只在设置时传头文件），`tests/test_niah_cue_tasks.py`（words 针）。
