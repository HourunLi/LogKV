#!/bin/bash
# niahLogKV：在训练环境里训练学习保留打分头（train_retention.py），环境与分布式设置同 majob.sh。
# 用法：bash retention.sh <YAML>
#   YAML 给出 checkpoint（save_path 或 retention_checkpoint）、其 LogKV 缓存与训练数据，
#   以及打分头的输出路径（retention_out，缺省为 log_kv_retention_heads），例如
#   bash retention.sh exp/qwen1.7b-32k/eval_niah_heads_p1024.yaml
# 其余 retention_* 键见 train_retention.py；RETENTION_ARGS 追加命令行参数（覆盖 YAML）。
# 已训到 retention_steps 的头直接跳过；中断后重跑从最近一次保存继续。
# source /home/miniconda3/bin/activate megatron-lm-014
if ! source /home/ma-user/anaconda3/bin/activate torch218; then
    echo "❌ 无法激活 torch218，停止执行，避免使用错误的 Python 环境。"
    exit 1
fi

# Set caches before Python imports Hugging Face and freezes its cache paths.
export HF_HOME=/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache
export HF_DATASETS_CACHE="${HF_HOME}/hf_cache"
export HF_EVALUATE_CACHE="${HF_HOME}/evaluate"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export RULER_CACHE_DIR="${HF_HOME}/ruler_cache"
export NLTK_DATA="${HF_HOME}/nltk_data"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1

PYTHON_BIN=$(python -c 'import sys; print(sys.executable)') || exit 1
echo "🐍 流水线 Python: ${PYTHON_BIN}"

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
# IB/RoCE ACK 超时是指数而非秒数：4.096us x 2^N，网卡只取低 5 bit（有效 1..31）。
# 12000 截成 uint8 是 224，低 5 bit 为 0 = 无限等待：尾包丢失后永不重传，NCCL
# 内核以 100% 利用率空等，表现为随机卡死。22 约 17s 重传一次，重试耗尽后报错。
export NCCL_IB_TIMEOUT=22
# LogKV 路由决策与注意力重叠（侧流）。先在目标 GPU 上运行
# python unused/check_route_overlap.py，输出 bitwise_equal=true 后再设为 1。
export LOGKV_ROUTE_OVERLAP=${LOGKV_ROUTE_OVERLAP:-0}
export NCCL_NET_GDR_LEVEL=2  # Enable GPUDirect RDMA if RDMA is available # optim0129
export NCCL_MIN_NCHANNELS=4  # Increase NCCL channels # optim0129

# ==============================================================================
# Distributed Training Setup
# ==============================================================================
# Detect GPUs visible to this job (including CUDA_VISIBLE_DEVICES); ModelArts
# allocation is an upper bound. GPUS_PER_NODE may explicitly select fewer GPUs.
GPUS_PER_NODE=$("${PYTHON_BIN}" - <<'PY_GPU_COUNT'
import os
import sys

import torch

visible = torch.cuda.device_count()
if visible < 1:
    raise SystemExit("ERROR: no CUDA GPUs are visible; check CUDA_VISIBLE_DEVICES and the CUDA runtime.")
try:
    allocated = int(os.environ.get("MA_NUM_GPUS") or visible)
    available = min(visible, allocated)
    selected = int(os.environ.get("GPUS_PER_NODE") or available)
except ValueError:
    raise SystemExit("ERROR: GPUS_PER_NODE and MA_NUM_GPUS must be positive integers.")
if allocated < 1 or not 1 <= selected <= available:
    raise SystemExit(
        f"ERROR: GPUS_PER_NODE={selected} must be within 1..{available} "
        f"(visible={visible}, MA_NUM_GPUS={allocated})."
    )
print(f"GPU detection: visible={visible}, MA_NUM_GPUS={allocated}, using={selected}", file=sys.stderr)
print(selected)
PY_GPU_COUNT
) || exit 1
NUM_NODES=${MA_NUM_HOSTS:-${GROUP_WORLD_SIZE:-1}}
MASTER_ADDR=${MASTER_ADDR:-localhost}
if [ -n "${MA_VJ_NAME:-}" ]; then
    MASTER_ADDR="${MA_VJ_NAME}-${MA_TASK_NAME}-${MA_MASTER_INDEX:-0}.${MA_VJ_NAME}"
fi
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${VC_TASK_INDEX:-${NODE_RANK:-${GROUP_RANK:-0}}}

for name in GPUS_PER_NODE NUM_NODES NODE_RANK MASTER_PORT; do
    value=${!name}
    if [[ -z "${value}" || "${value}" == *[!0-9]* ]]; then
        echo "❌ 致命错误：${name} 必须是非负整数，当前值为 '${value}'"
        exit 1
    fi
done

if [ "${NUM_NODES}" -gt 1 ] && [ "${MASTER_ADDR}" = "localhost" ]; then
    echo "❌ 致命错误：NUM_NODES=${NUM_NODES} 但 MASTER_ADDR=localhost。"
    echo "   多节点 torchrun 必须使用所有节点可访问的 master 地址；否则每个节点会各自组成 world_size=${GPUS_PER_NODE}。"
    exit 1
fi


RETENTION_MASTER_PORT=${RETENTION_MASTER_PORT:-${MASTER_PORT}}
export LITGPT_EXPECTED_WORLD_SIZE=${LITGPT_EXPECTED_WORLD_SIZE:-$((NUM_NODES * GPUS_PER_NODE))}

echo "🌍 打分头训练: Node ${NODE_RANK} / ${NUM_NODES}"
echo "🧮 期望 world size: ${LITGPT_EXPECTED_WORLD_SIZE} (= ${NUM_NODES} nodes × ${GPUS_PER_NODE} gpus)"
echo "🔗 rendezvous: ${MASTER_ADDR}:${RETENTION_MASTER_PORT}"

if [ -z "$1" ]; then
    echo "❌ 致命错误：未提供 YAML 配置文件！"
    echo "💡 正确用法：bash $0 <yaml>（例如 exp/qwen1.7b-32k/eval_niah_heads_p1024.yaml）"
    exit 1
fi
CONFIG_FILE=$1
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "❌ 致命错误：找不到配置文件 ${CONFIG_FILE}"
    exit 1
fi

# 跟随 `config:` 继承并展开 ${VAR}，与 eval.sh 读 YAML 的方式相同。
CONFIG_EXPORTS=$("${PYTHON_BIN}" - "${CONFIG_FILE}" <<'EOF'
import os
import re
import shlex
import sys

import yaml


def expand_string(text):
    def replace(match):
        inner = match.group(1)
        if ":-" in inner:
            name, default = inner.split(":-", 1)
        elif "-" in inner:
            name, default = inner.split("-", 1)
        else:
            name, default = inner, ""
        return os.environ.get(name, default)

    return re.sub(r"\$\{([^}^{]+)\}", replace, text)


def expand(value):
    if isinstance(value, str):
        return expand_string(value)
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value


def load(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "config" in cfg:
        base = load(os.path.join(os.path.dirname(path), cfg.pop("config")))
        cfg = {**base, **cfg}
    return cfg


def emit(name, value):
    print(f"{name}={shlex.quote('' if value is None else str(value))}")


cfg = expand(load(sys.argv[1]))
emit("SAVE_DIR", cfg.get("retention_checkpoint") or cfg.get("save_path") or "")
emit("RETENTION_OUT", cfg.get("retention_out") or cfg.get("log_kv_retention_heads") or "")
emit("RETENTION_STEPS", cfg.get("retention_steps") or 2000)
semantic = bool(cfg.get("log_kv_semantic_clusters"))
alpha = cfg.get("log_kv_alpha_exact_tokens")
emit("ALPHA_EXACT_TOKENS", (256 if alpha is None else int(alpha)) if semantic else 0)
EOF
) || { echo "❌ 致命错误：解析 ${CONFIG_FILE} 失败。"; exit 1; }
eval "${CONFIG_EXPORTS}"

if [ -z "${SAVE_DIR}" ]; then
    echo "❌ 致命错误：${CONFIG_FILE} 里没有 save_path（或 retention_checkpoint），不知道在哪个 checkpoint 上训练打分头。"
    exit 1
fi
if [ -z "${RETENTION_OUT}" ]; then
    echo "❌ 致命错误：${CONFIG_FILE} 里没有 retention_out（或 log_kv_retention_heads），不知道打分头写到哪里。"
    exit 1
fi
if [ "${ALPHA_EXACT_TOKENS}" = "0" ]; then
    echo "❌ 致命错误：${CONFIG_FILE} 没有 Alpha 精确池（log_kv_alpha_exact_tokens=0 或未开语义簇），打分头无处可用。"
    exit 1
fi

# checkpoint：save_path 下的 lit_model.pth 或 step_*/lit_model.pth（eval.py 取最新的 step_*）。
checkpoint_exists() {
    [ -f "$1" ] || { [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }
}
FOUND_CHECKPOINT=""
for CANDIDATE in "${SAVE_DIR}"/lit_model.pth "${SAVE_DIR}"/step_*/lit_model.pth; do
    if checkpoint_exists "${CANDIDATE}"; then
        FOUND_CHECKPOINT="${CANDIDATE}"
        break
    fi
done
if [ -z "${FOUND_CHECKPOINT}" ]; then
    echo "❌ 致命错误：${SAVE_DIR} 下没有 lit_model.pth（也没有 step_*/lit_model.pth）。"
    echo "   先用 majob.sh 训练该 checkpoint（BENCHMARKS=none NIAH_BENCHMARKS=none bash majob.sh <它的训练 YAML>），"
    echo "   或在 YAML 里用 retention_checkpoint 指向已有的 checkpoint 目录。"
    exit 1
fi

echo "✅ checkpoint: ${SAVE_DIR}（找到 ${FOUND_CHECKPOINT}）"
echo "🎯 打分头输出: ${RETENTION_OUT}（目标 ${RETENTION_STEPS} 步）"
mkdir -p "$(dirname "${RETENTION_OUT}")"
RETENTION_ARGS=${RETENTION_ARGS:-}
if [ -n "${RETENTION_ARGS}" ]; then
    echo "🧪 额外参数: ${RETENTION_ARGS}"
fi

"${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes=${NUM_NODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${RETENTION_MASTER_PORT} \
    train_retention.py --config "${CONFIG_FILE}" ${RETENTION_ARGS}

STATUS=$?
if [ ${STATUS} -ne 0 ]; then
    echo "❌ [Node ${NODE_RANK}] 打分头训练失败 (Exit Code: ${STATUS})；请查看上方第一个 Traceback。"
    exit "${STATUS}"
fi
if [ "${NODE_RANK}" -eq 0 ] && [ ! -s "${RETENTION_OUT}" ]; then
    echo "❌ 致命错误：训练进程返回成功，但 ${RETENTION_OUT} 不存在或为空。"
    exit 1
fi
echo "🎉 [Node ${NODE_RANK}] 打分头训练结束：${RETENTION_OUT}"
echo "   评测：NIAH_ONLY=1 bash eval.sh ${CONFIG_FILE}"
