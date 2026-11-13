# 512K / 1M 评测

使用 `eval_512k_1m_<method>.yaml` 显式开启，两档长度为 524288、1048576 tokens。
窗口额外留 1024 tokens，YaRN 以训练长度 32768 为原长，factor 为 16、32。
原有全量 YAML 仍只默认测 32K–256K；也可在原 YAML 上用 `EVAL_LENGTHS=512k,1m`。

## 数据准备

下载与评测节点必须看到相同的普通文件目录。以下在仓库根目录执行，下载脚本默认绕过证书检查；
不要加 `--verify-ssl`。Hub 的下载 URL 固定到本次查询的 revision，逐文件核对大小和 SHA256；
新数据流程不使用 snapshot 软链接，也不使用目录重命名发布。

```bash
export HF_HOME=/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache
export LONGBENCH2_DIR="${HF_HOME}/longbench2"
export BABILONG_DIR="${HF_HOME}/babilong"
export LONG_PPL_DOCS=10
export LONG_PPL_BOOKS="${HF_HOME}/long_ppl_books_1m/emozilla--pg19"

bash diag.sh unused/fetch_long_eval_data.py --lengths 512k,1m --longbench2 \
  --ppl-books-hub emozilla/pg19 \
  --tokenizer /home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-8B-Base \
  --out fetch_512k_1m.json

bash diag.sh unused/check_long_eval_data.py --lengths 512k,1m \
  --suites ruler,babilong,books,longbench2 \
  --tokenizer /home/ma-user/work/bucket-pangu-green/lihourun/checkpoints/Qwen/Qwen3-8B-Base
```

1.7B 将 tokenizer 改为 `Qwen3-1.7B-Base`。每个 tokenizer 有独立的 PPL 选择缓存。
必须确认所选书籍未用于当前 CPT；上面的 PG19 train 只是下载接口示例，不保证与训练集无重叠。
有已分离的长文时，改用 `--ppl-books-data <普通 parquet/jsonl 文件或目录>`。
下载器逐 shard 扫描，直到够 `LONG_PPL_DOCS` 本或遍历完来源；不足时明确报告实际数量，
没有任何合格长文时报错。不同方法必须使用同一目录、tokenizer 和 `LONG_PPL_DOCS`。

下载日志实时打印并写入 `fetch_512k_1m.summary.txt`，可在另一终端用
`tail -f fetch_512k_1m.summary.txt` 查看。日志显示阶段开始/结束、文件大小、已下载字节和速度、
SHA256 校验、BABILong JSON → Arrow，以及 PG19 的 shard/文档分词和合格数量。
耗时调用期间每 20 秒打印 `WAIT`、已耗时间和最近状态；它表示进程仍在该阶段，
不能单凭 `WAIT` 判定远端网络在传输。读超时及重试仍按原设置执行，429 会单独报告等待时间。

该命令先准备 RULER，再准备 BABILong、LongBench v2，最后扫描 PG19。
512K/1M 的 BABILong 共 10 个大 JSON，还需写 Arrow；PG19 按文件名下载完整 shard，
对候选长文分词后才能判断是否达到 1048576 tokens，耗时可能明显超过单次下载。
务必保留上面的 `export LONG_PPL_DOCS=10`：不设置时目标是 100 本，可能扫描更多 shard。
这个值也必须带到评测环境。成功下载且校验通过的普通文件会复用；中断留下的不完整文件会重新下载。
旧版本只在结束时统一打印摘要，`python -u` 本身不能补出这些尚未打印的状态；
拉取更新后须重新启动下载进程，已有进程不会自动获得新日志功能。

可单独准备：

```bash
bash diag.sh unused/fetch_long_eval_data.py --lengths 512k,1m --no-ruler-qa
bash diag.sh unused/fetch_long_eval_data.py --ruler-essays --no-babilong --no-ruler-qa
```

前者下载 BABILong qa1–qa5 官方 `512k`、`1M` JSON，各 100 条，并生成普通 Arrow 流文件。
评测优先以内存映射读取 Arrow，不创建 datasets builder 缓存；只有手工 JSON 时可只读加载 JSON。
后者准备 `RULER_CACHE_DIR/paul_graham_essays.txt` 和普通 NLTK punkt_tab 文件。
LongBench v2 两档共用原有 20 子集、503 道题，不另下一个长度版本。

## 任务口径

- RULER：512K/1M 使用 `extended-v1`。essay 背景不够长时循环重复背景词序列，再由原生成器插入针；
  不重复已插入的针，不改变答案和计分。新增档位默认每任务生成 100 条，可设 `RULER_LONG_SAMPLES`。
  样本长度与协议字段见 [指标口径](evaluation-metrics.md)；不足目标长度 4096 tokens 以上时报错。
  保留已有 SQuAD 排除规则，因此超长档为 12 项，不能把均值直接当成其他论文完整 13 项的均值。
  32K–256K 的生成参数保持不变。此协议必须在论文中披露，不能称为原版 128K RULER 协议。
- BABILong：官方长度用其原始 tokenizer 定义，不保证 Qwen 编码后恰好达到相同 token 数。
  计分、两条示例和 16-token 生成上限保持原样。读取普通数据失败时报错，不回退到 Hub builder。
- 长文 PPL：512K/1M 都选择至少 1048576 tokens 的同一组文档。曲线点为 8K、16K、32K、64K、
  128K、256K、512K，以及 1M 档额外的 1M 点；每点预测最后 4096 tokens。
  相同点在两档使用相同文本，YaRN 设置不同。旧的 <=256K 曲线继续使用原来的 256K 文档集合。
  独立的 `long_ppl_512k`、`long_ppl_1m` 也可用 `EVAL_SUITES=ppl` 选择，来源为 `LONG_PPL_DATA`。
- LongBench v2：沿用全部原始问题，在各档窗口下评测。不能把所有短样本都称作实际 1M 输入。
- InfiniteBench：目前只注册到 256K，新长度默认不加入；显式要求时会报错，避免制造名为 1M 的短输入评测。

生成明细同时记录原始、截断后 prompt 长度和截去的 token 数。比较不同方法时核对实际输入、
样本数、权重路径和 `log_kv_dense_mode`。本次入口复用已有权重；8B 的完整 Dense KV 在 1M 时需要
远高于 32K 的显存，应按实际硬件安排，不能把 OOM 当作任务分数。
