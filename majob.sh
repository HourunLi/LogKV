#!/bin/bash
# source /home/miniconda3/bin/activate megatron-lm-014
source /home/ma-user/anaconda3/bin/activate torch218
# DeepSeek 3Bv2 Sandwich Training Script with YAML Configuration
# Optimized for H200 GPUs (141GB VRAM)

export CUDA_DEVICE_MAX_CONNECTIONS=32
export CUDNN_LOGERR_DBG=1
export CUDNN_LOGDEST_DBG=stderr
# logKV 流式 attention 每 chunk 产生大量不等长的小分配（评测数千条样本、训练
# T/2 个 chunk），expandable_segments 让分配器按段扩展而非整块缓存，
# 显著缓解长时运行的碎片化 OOM。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PATH=/usr/local/cuda-12.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

export NVTE_WITH_USER_CUDA=1 # must turn on when multiple-nodes
export CUDNN_PATH=/usr # must turn on when multiple-nodes
export NVTE_CUDA_INCLUDE_DIR=/usr/local/cuda-12.8/include # must turn on when multiple-nodes

export NCCL_NVLS_ENABLE=0
export NCCL_NET_PLUGIN=none
export NCCL_IB_TIMEOUT=12000
export NCCL_NET_GDR_LEVEL=2  # Enable GPUDirect RDMA if RDMA is available # optim0129
export NCCL_MIN_NCHANNELS=4  # Increase NCCL channels # optim0129

# ==============================================================================
# Distributed Training Setup
# ==============================================================================
GPUS_PER_NODE=${MA_NUM_GPUS:-8}
NUM_NODES=${MA_NUM_HOSTS:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
if [ -n "${MA_VJ_NAME:-}" ]; then
    MASTER_ADDR="${MA_VJ_NAME}-${MA_TASK_NAME}-${MA_MASTER_INDEX:-0}.${MA_VJ_NAME}"
fi
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${VC_TASK_INDEX:-0}

pip install tensorboard
echo "🌍 正在启动多机多卡训练: Node ${NODE_RANK} / ${NUM_NODES}"
echo "🔗 Master 地址: ${MASTER_ADDR}:${MASTER_PORT}"

# ==============================================================================
# 🌟 自动化实验流水线 (Train -> Eval)
# ==============================================================================
# 设定你这次实验的 YAML 配置文件路径
# 1. 🌟 核心拦截：检查用户是否传了参数 ($1 代表传入的第一个参数)
if [ -z "$1" ]; then
    echo "❌ 致命错误：未提供 YAML 配置文件！"
    echo "💡 正确用法：bash $0 <你的yaml文件路径>"
    exit 1
fi

# 2. 把传入的第一个参数赋值给 CONFIG_FILE 变量
CONFIG_FILE=$1

# 3. 🌟 安全拦截：检查这个文件在硬盘上到底存不存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 致命错误：找不到配置文件 $CONFIG_FILE"
    exit 1
fi

echo "🔍 正在从 $CONFIG_FILE 中提取配置..."

# 🌟 核心魔法：使用 Python 一行流提取 yaml 里的 save_path
# 假设你的 yaml 里写的键名叫 save_path。如果是其他的，把下面单引号里的名字改掉
RAW_SAVE_DIR=$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG_FILE}'))['save_path'])")
eval SAVE_DIR="\"${RAW_SAVE_DIR}\""

# 安全检查：如果没提取到，立刻报错退出
if [ -z "$SAVE_DIR" ] || [ "$SAVE_DIR" == "None" ]; then
    echo "❌ 提取失败：在 ${CONFIG_FILE} 里没有找到 'save_path' 配置！"
    exit 1
fi

echo "✅ 成功提取模型保存路径: ${SAVE_DIR}"

# ==============================================================================
# 🧩 logKV 专属：从训练 YAML（跟随 'config:' 继承）提取 logKV 设置，
# 让评测使用与模型适配时相同的压缩注意力。这是 logKV 分支独有的开发代码。
# 本管线只跑 logKV 压缩路线，评测恒定启用（无 dense 分支）。
# ==============================================================================
read -r LOG_KV_B LOG_KV_RECENT LOG_KV_PREFILL LOG_KV_PIN LOG_KV_PIN_OBS SAVE_CKPT TOKENIZER_CANDIDATES <<< "$(python - "${CONFIG_FILE}" <<'EOF'
import os
import sys

import yaml


def load(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "config" in cfg:
        base = load(os.path.join(os.path.dirname(path), cfg.pop("config")))
        cfg = {**base, **cfg}
    return cfg


cfg = load(sys.argv[1])
arch_name = cfg.get("arch_name", "Qwen/Qwen3-0.6B-Base")
ckpt_dir = cfg.get("ckpt_dir") or os.path.join("checkpoints", arch_name)
resume_dir = cfg.get("resume_dir")
tokenizer_dir = cfg.get("tokenizer_dir")

candidates = []
for d in (tokenizer_dir, resume_dir, ckpt_dir, os.path.join("checkpoints", arch_name)):
    if d and d not in candidates:
        candidates.append(d)

print(
    cfg.get("log_kv_B", 512),
    cfg.get("log_kv_recent_size", 1024),
    cfg.get("log_kv_prefill_block", 256),
    cfg.get("log_kv_pin_size", 0),
    cfg.get("log_kv_pin_obs_window", 64),
    str(bool(cfg.get("save_ckpt", False))).lower(),
    ":".join(candidates),
)
EOF
)"

# 流水线前置检查：eval 阶段固定评测 ${SAVE_DIR}，如果既没有现成权重、
# 训练又不会保存（save_ckpt: false），训练完就会在 eval 的 tokenizer/权重
# 加载处崩溃。在开训前拦截，避免白跑几小时。
if [ "${SAVE_CKPT}" != "true" ] && [ ! -f "${SAVE_DIR}/lit_model.pth" ]; then
    echo "❌ 致命错误：${CONFIG_FILE} 中 save_ckpt 未开启，且 ${SAVE_DIR} 下没有现成权重。"
    echo "   训练结束后将没有 checkpoint 可评测。请在 YAML 中设置 save_ckpt: true。"
    exit 1
fi

has_tokenizer() {
    [ -n "$1" ] && { [ -f "$1/tokenizer.json" ] || [ -f "$1/tokenizer.model" ]; }
}

TOKENIZER_SOURCE=""
IFS=':' read -r -a TOKENIZER_CANDIDATE_ARRAY <<< "${TOKENIZER_CANDIDATES}"
for RAW_TOK_DIR in "${TOKENIZER_CANDIDATE_ARRAY[@]}"; do
    eval TOK_DIR="\"${RAW_TOK_DIR}\""
    if has_tokenizer "${TOK_DIR}"; then
        TOKENIZER_SOURCE="${TOK_DIR}"
        break
    fi
done

LOG_KV_ARGS="--log_kv_B ${LOG_KV_B} --log_kv_recent_size ${LOG_KV_RECENT} --log_kv_prefill_block ${LOG_KV_PREFILL} --log_kv_pin_size ${LOG_KV_PIN} --log_kv_pin_obs_window ${LOG_KV_PIN_OBS}"
TOKENIZER_ARGS=""
if [ -n "${TOKENIZER_SOURCE}" ]; then
    TOKENIZER_ARGS="--tokenizer_dir ${TOKENIZER_SOURCE}"
    echo "🔤 tokenizer source: ${TOKENIZER_SOURCE}"
else
    echo "⚠️ 未在候选目录中找到 tokenizer.json/tokenizer.model: ${TOKENIZER_CANDIDATES}"
    echo "   如 eval 仍报 tokenizer 缺失，请在 YAML 中设置 tokenizer_dir。"
fi
echo "🧩 logKV eval: B=${LOG_KV_B}, recent_size=${LOG_KV_RECENT}, prefill_block=${LOG_KV_PREFILL}, pin=${LOG_KV_PIN} (obs ${LOG_KV_PIN_OBS})"

ensure_checkpoint_tokenizer() {
    if has_tokenizer "${SAVE_DIR}"; then
        return 0
    fi
    if [ -z "${TOKENIZER_SOURCE}" ]; then
        return 1
    fi
    if [ "${NODE_RANK}" -eq 0 ]; then
        echo "🔤 ${SAVE_DIR} 缺少 tokenizer，正在从 ${TOKENIZER_SOURCE} 复制 tokenizer/config 文件..."
        mkdir -p "${SAVE_DIR}"
        cp -f "${TOKENIZER_SOURCE}"/*.json "${SAVE_DIR}/" 2>/dev/null || true
        cp -f "${TOKENIZER_SOURCE}"/*.model "${SAVE_DIR}/" 2>/dev/null || true
    fi
    return 0
}

# ==============================================================================
# 🌟 核心新增：检查 Checkpoint 是否已存在
# ==============================================================================
if [ -f "${SAVE_DIR}/lit_model.pth" ]; then
    echo "================================================="
    echo "⏩ [Node ${NODE_RANK}] 阶段一跳过：检测到模型权重已存在于 ${SAVE_DIR}/lit_model.pth"
    echo "⏩ 直接进入评测阶段！"
    echo "================================================="
else
    echo "================================================="
    echo "🚀 [Node ${NODE_RANK}] 阶段一：未找到现有权重，开始执行 Continual Pre-Training"
    echo "================================================="

    torchrun \
        --nnodes=${NUM_NODES} \
        --nproc_per_node=${GPUS_PER_NODE} \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        demo.py --config ${CONFIG_FILE}

    TRAIN_STATUS=$?
    if [ $TRAIN_STATUS -ne 0 ]; then
        echo "⚠️ [Node ${NODE_RANK}] 训练退出码非零 (Exit Code: $TRAIN_STATUS)，通常为 NCCL 正常销毁竞争，非实质错误。"
    fi

    echo "🎉 [Node ${NODE_RANK}] 阶段一（训练）结束"

    # 强制操作系统彻底回收显存
    sleep 15
fi

# ==============================================================================
# 🌟 阶段一产物核验：save_ckpt: true 时训练必须产出权重。缺失说明训练中途
# 崩溃/被杀（上面的非零退出码并不总是 NCCL 良性竞争）——在这里立刻失败，
# 否则 eval 阶段只会报一个误导性的「加载 checkpoint 出错」。
# ==============================================================================
if [ "${SAVE_CKPT}" == "true" ] && [ ! -f "${SAVE_DIR}/lit_model.pth" ]; then
    echo "❌ 致命错误：训练阶段结束，但 ${SAVE_DIR}/lit_model.pth 不存在（训练退出码见上方 ⚠️ 行）。"
    echo "   排查（在训练日志中从后往前找）："
    echo "   ➤ 无 'Reached max_steps' / 'Data exhausted' → 训练循环中途崩溃，向上翻最后一个 Traceback；"
    echo "   ➤ 有 'Training complete' 但无 'Final save dir:' → 生效配置 save_ckpt 为 false；"
    echo "   ➤ 有 'Saving to ... lit_model.pth' 但无 'Done.' → 保存阶段被杀（墙钟/内存/磁盘配额）；"
    echo "   ➤ 'Final save dir:' 显示 _v2 之类目录 → save_path 下已有旧权重被顺延，请清理或改 YAML。"
    exit 1
fi

if [ -f "${SAVE_DIR}/lit_model.pth" ]; then
    if ! ensure_checkpoint_tokenizer; then
        echo "❌ 致命错误：${SAVE_DIR} 下有 lit_model.pth，但没有 tokenizer.json/tokenizer.model。"
        echo "   请在 ${CONFIG_FILE} 中设置 tokenizer_dir 指向基座模型 tokenizer 目录，或手动复制 tokenizer 文件。"
        exit 1
    fi
fi

# ==============================================================================
# 🌟 文件锁 Barrier：每个节点写独立文件，避免共享文件系统缓存问题
# ==============================================================================
BARRIER_DIR="${SAVE_DIR}/.eval_barrier_d"
mkdir -p "${BARRIER_DIR}"

echo "================================================="
echo "🎯 [Node ${NODE_RANK}] 写入 Barrier 文件，等待所有 ${NUM_NODES} 个节点就绪..."
echo "================================================="

# 每个节点写自己的独立 barrier 文件（文件存在性比文件内容跨节点可见更快）
touch "${BARRIER_DIR}/node_${NODE_RANK}"

CUR_COUNT=0
LOOP_ITER=0
while [ "${CUR_COUNT}" -lt "${NUM_NODES}" ]; do
    # 统计有多少个节点的 barrier 文件已经存在
    CUR_COUNT=$(ls -1 "${BARRIER_DIR}"/node_* 2>/dev/null | wc -l)
    LOOP_ITER=$((LOOP_ITER + 1))
    echo "🔄 [Node ${NODE_RANK}] 第 ${LOOP_ITER} 轮检查: ${CUR_COUNT}/${NUM_NODES} 节点已就绪"
    sleep 5
done

echo "✅ [Node ${NODE_RANK}] 所有 ${NUM_NODES} 个节点已就绪，启动评测"

# 🌟 核心缓冲：休眠 30 秒确保 NCCL 彻底回收 + 避免 barrier 竞争
sleep 30

# 拼接 benchmark 列表（避免换行空格被解析进 task 名）
BENCHMARKS="boolq,piqa,social_iqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"
BENCHMARKS="${BENCHMARKS},mmlu,ceval-valid,ifeval,truthfulqa_gen,truthfulqa_mc1,truthfulqa_mc2"
BENCHMARKS="${BENCHMARKS},longbench_2wikimqa,longbench_dureader,longbench_gov_report,longbench_hotpotqa"
BENCHMARKS="${BENCHMARKS},longbench_lcc,longbench_lsht,longbench_multi_news,longbench_multifieldqa_en,longbench_multifieldqa_zh"
BENCHMARKS="${BENCHMARKS},longbench_musique,longbench_narrativeqa,longbench_passage_count,longbench_passage_retrieval_en"
BENCHMARKS="${BENCHMARKS},longbench_qasper,longbench_qmsum,longbench_repobench-p,longbench_samsum,longbench_trec,longbench_triviaqa,longbench_vcsum"
BENCHMARKS="${BENCHMARKS},longbench_2wikimqa_e,longbench_gov_report_e,longbench_hotpotqa_e,longbench_lcc_e,longbench_multi_news_e"
BENCHMARKS="${BENCHMARKS},longbench_multifieldqa_en_e,longbench_passage_count_e,longbench_passage_retrieval_en_e,longbench_qasper_e"
BENCHMARKS="${BENCHMARKS},longbench_repobench-p_e,longbench_samsum_e,longbench_trec_e,longbench_triviaqa_e"
NIAH_BENCHMARKS="niah_single_1,niah_single_2,niah_single_3"

# NIAH 任务需要的 metadata：tokenizer 路径 + 测试的上下文长度
# max_seq_lengths 可根据模型实际最大上下文调整
META='{"pretrained": "'"${SAVE_DIR}"'", "max_seq_lengths": [1024, 2048, 4096, 8192, 16384, 32768]}'

# 评测结果：rank0 写入 JSON（带时间戳），与 litgpt evaluate 惯例一致放在 checkpoint 下 evaluate/
EVAL_OUTPUT_DIR="${SAVE_DIR}/evaluate"

torchrun \
    --nnodes=${NUM_NODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    eval.py \
    --checkpoint_dir ${SAVE_DIR} \
    --benchmark ${BENCHMARKS} \
    --output_path "${EVAL_OUTPUT_DIR}" \
    ${LOG_KV_ARGS} \
    ${TOKENIZER_ARGS}

# for NIAH
torchrun \
    --nnodes=${NUM_NODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    eval.py \
    --checkpoint_dir ${SAVE_DIR} \
    --benchmark ${NIAH_BENCHMARKS} \
    --metadata "${META}" \
    --output_path "${EVAL_OUTPUT_DIR}" \
    ${LOG_KV_ARGS} \
    ${TOKENIZER_ARGS}

EVAL_STATUS=$?
if [ $EVAL_STATUS -ne 0 ]; then
    echo "❌ [Node ${NODE_RANK}] 评测阶段崩溃 (Exit Code: $EVAL_STATUS)！"
    exit $EVAL_STATUS
fi

# 清理 barrier 目录（rank 0 负责）
if [ ${NODE_RANK} -eq 0 ] && [ -d "${BARRIER_DIR}" ]; then
    rm -rf "${BARRIER_DIR}"
fi

echo "================================================="
echo "🎊 [Node ${NODE_RANK}] 全部流水线 (Training + Evaluation) 执行完美结束！"
echo "================================================="

set +x
