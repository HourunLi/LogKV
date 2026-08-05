#!/bin/bash

# Eval-only companion to majob.sh. It uses the same runtime environment and
# logKV argument extraction, but never launches CPT training.
#
# Usage:
#   bash eval.sh <training_yaml_or_eval_yaml> [benchmark_csv|none] [niah_csv|none]
#
# Examples:
#   bash eval.sh exp/qwen1.7b-32k/arc.yaml
#   bash eval.sh exp/qwen1.7b-32k/arc.yaml piqa none
#   DIAG_ARGS="--log_kv_diag_mode baseline --log_kv_diag_exact_from_layer 21" \
#       bash eval.sh exp/qwen1.7b-32k/diag.yaml niah_single_1 none

if [ -f /home/ma-user/anaconda3/bin/activate ]; then
    source /home/ma-user/anaconda3/bin/activate torch218
fi

export CUDA_DEVICE_MAX_CONNECTIONS=32
export CUDNN_LOGERR_DBG=1
export CUDNN_LOGDEST_DBG=stderr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PATH=/usr/local/cuda-12.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

export NVTE_WITH_USER_CUDA=1
export CUDNN_PATH=/usr
export NVTE_CUDA_INCLUDE_DIR=/usr/local/cuda-12.8/include

export NCCL_NVLS_ENABLE=0
export NCCL_NET_PLUGIN=none
export NCCL_IB_TIMEOUT=12000
export NCCL_NET_GDR_LEVEL=2
export NCCL_MIN_NCHANNELS=4

GPUS_PER_NODE=${MA_NUM_GPUS:-8}
NUM_NODES=${MA_NUM_HOSTS:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
if [ -n "${MA_VJ_NAME:-}" ]; then
    MASTER_ADDR="${MA_VJ_NAME}-${MA_TASK_NAME}-${MA_MASTER_INDEX:-0}.${MA_VJ_NAME}"
fi
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${VC_TASK_INDEX:-0}

echo "Starting eval-only job: Node ${NODE_RANK} / ${NUM_NODES}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"

if [ -z "$1" ]; then
    echo "ERROR: missing YAML config."
    echo "Usage: bash $0 <training_yaml_or_eval_yaml> [benchmark_csv|none] [niah_csv|none]"
    exit 1
fi

CONFIG_FILE=$1
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: config file not found: ${CONFIG_FILE}"
    exit 1
fi

echo "Reading eval settings from ${CONFIG_FILE}"

CONFIG_EXPORTS=$(python - "${CONFIG_FILE}" <<'EOF'
import os
import re
import shlex
import sys

import yaml


def expand_string(text: str) -> str:
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
        base_path = os.path.join(os.path.dirname(path), cfg.pop("config"))
        base = load(base_path)
        cfg = {**base, **cfg}
    return expand(cfg)


def emit(name, value):
    if value is None:
        value = ""
    print(f"{name}={shlex.quote(str(value))}")


cfg = load(sys.argv[1])
arch_name = cfg.get("arch_name", "Qwen/Qwen3-0.6B-Base")
save_dir = cfg.get("save_path") or cfg.get("checkpoint_dir")
if not save_dir:
    raise SystemExit("Config must define save_path (training YAML) or checkpoint_dir (eval YAML).")

ckpt_dir = cfg.get("ckpt_dir") or os.path.join("checkpoints", arch_name)
resume_dir = cfg.get("resume_dir")
tokenizer_dir = cfg.get("tokenizer_dir")
output_path = cfg.get("output_path") or os.path.join(save_dir, "evaluate")
benchmark = cfg.get("benchmark") or ""

try:
    max_steps = int(float(cfg.get("max_steps", 0) or 0))
except (TypeError, ValueError):
    max_steps = 0
try:
    num_epochs = int(float(cfg.get("num_epochs", 10) or 0))
except (TypeError, ValueError):
    num_epochs = 0

candidates = []
for path in (tokenizer_dir, resume_dir, ckpt_dir, os.path.join("checkpoints", arch_name)):
    if path and path not in candidates:
        candidates.append(path)

emit("SAVE_DIR", save_dir)
emit("EVAL_OUTPUT_DIR", output_path)
emit("CONFIG_BENCHMARKS", benchmark)
emit("LOG_KV_B", cfg.get("log_kv_B", 512))
emit("LOG_KV_RECENT", cfg.get("log_kv_recent_size", 1024))
emit("LOG_KV_PREFILL", cfg.get("log_kv_prefill_block", 256))
emit("LOG_KV_PIN", cfg.get("log_kv_pin_size", 0))
emit("LOG_KV_PIN_OBS", cfg.get("log_kv_pin_obs_window", 64))
emit("LOG_KV_SECOND_ORDER_SCALE", cfg.get("log_kv_second_order_scale", 1.0))
emit("MAX_STEPS", max_steps)
emit("NUM_EPOCHS", num_epochs)
emit("TOKENIZER_CANDIDATES", ":".join(candidates))
EOF
)
CONFIG_STATUS=$?
if [ ${CONFIG_STATUS} -ne 0 ]; then
    echo "ERROR: failed to parse config ${CONFIG_FILE}"
    exit ${CONFIG_STATUS}
fi
eval "${CONFIG_EXPORTS}"

checkpoint_exists() {
    [ -f "$1" ] || { [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }
}

checkpoint_finished() {
    python - "${SAVE_DIR}" "${MAX_STEPS}" "${NUM_EPOCHS}" <<'EOF'
import os
import sys

import yaml


save_dir = sys.argv[1]
try:
    max_steps = int(float(sys.argv[2] or 0))
except (TypeError, ValueError):
    max_steps = 0
try:
    num_epochs = int(float(sys.argv[3] or 0))
except (TypeError, ValueError):
    num_epochs = 0

meta_path = os.path.join(save_dir, "checkpoint_meta.yaml")
if not os.path.exists(meta_path):
    sys.exit(0)

with open(meta_path, encoding="utf-8") as f:
    meta = yaml.safe_load(f) or {}

if meta.get("in_progress", False):
    sys.exit(1)
if meta.get("finished", False):
    sys.exit(0)

try:
    global_step = int(meta.get("global_step", -1))
except (TypeError, ValueError):
    global_step = -1
try:
    completed_epochs = int(meta.get("completed_epochs", -1))
except (TypeError, ValueError):
    completed_epochs = -1
try:
    data_epoch = int(meta.get("data_epoch", -1))
except (TypeError, ValueError):
    data_epoch = -1

epoch_reached = completed_epochs >= num_epochs if num_epochs > 0 and completed_epochs >= 0 else False
if not epoch_reached and num_epochs > 0 and data_epoch > num_epochs:
    epoch_reached = True
step_reached = max_steps > 0 and global_step >= max_steps

sys.exit(0 if epoch_reached or step_reached or (max_steps <= 0 and num_epochs <= 0) else 1)
EOF
}

checkpoint_step_label() {
    python - "${SAVE_DIR}" "${MAX_STEPS}" "${NUM_EPOCHS}" <<'EOF'
import os
import sys

import yaml


save_dir, max_steps, num_epochs = sys.argv[1], sys.argv[2], sys.argv[3]
meta_path = os.path.join(save_dir, "checkpoint_meta.yaml")
if not os.path.exists(meta_path):
    print("legacy/no-meta")
    raise SystemExit
with open(meta_path, encoding="utf-8") as f:
    meta = yaml.safe_load(f) or {}
print(
    f"step={meta.get('global_step', 'unknown')}/{max_steps}, "
    f"epoch={meta.get('completed_epochs', 'unknown')}/{num_epochs} "
    f"(current={meta.get('data_epoch', 'unknown')}), "
    f"finished={meta.get('finished', False)}, "
    f"in_progress={meta.get('in_progress', False)}"
)
EOF
}

has_tokenizer() {
    [ -n "$1" ] && { [ -f "$1/tokenizer.json" ] || [ -f "$1/tokenizer.model" ]; }
}

TOKENIZER_SOURCE=""
IFS=':' read -r -a TOKENIZER_CANDIDATE_ARRAY <<< "${TOKENIZER_CANDIDATES}"
for TOK_DIR in "${TOKENIZER_CANDIDATE_ARRAY[@]}"; do
    if has_tokenizer "${TOK_DIR}"; then
        TOKENIZER_SOURCE="${TOK_DIR}"
        break
    fi
done

LOG_KV_ARG_LIST=(--log_kv_B "${LOG_KV_B}" --log_kv_recent_size "${LOG_KV_RECENT}" --log_kv_prefill_block "${LOG_KV_PREFILL}" --log_kv_pin_size "${LOG_KV_PIN}" --log_kv_pin_obs_window "${LOG_KV_PIN_OBS}")
if [ -n "${LOG_KV_SECOND_ORDER_SCALE}" ]; then
    LOG_KV_ARG_LIST+=(--log_kv_second_order_scale "${LOG_KV_SECOND_ORDER_SCALE}")
fi
DIAG_ARGS=${DIAG_ARGS:-}
TOKENIZER_ARGS=""
if [ -n "${TOKENIZER_SOURCE}" ]; then
    TOKENIZER_ARGS="--tokenizer_dir ${TOKENIZER_SOURCE}"
    echo "Tokenizer source: ${TOKENIZER_SOURCE}"
else
    echo "WARNING: tokenizer not found in candidates: ${TOKENIZER_CANDIDATES}"
    echo "If eval.py fails to load a tokenizer, set tokenizer_dir in the YAML."
fi

echo "Checkpoint: ${SAVE_DIR}"
echo "Output dir: ${EVAL_OUTPUT_DIR}"
echo "logKV eval: B=${LOG_KV_B}, recent_size=${LOG_KV_RECENT}, prefill_block=${LOG_KV_PREFILL}, pin=${LOG_KV_PIN} (obs ${LOG_KV_PIN_OBS}), second_order_scale=${LOG_KV_SECOND_ORDER_SCALE}"
if [ -n "${DIAG_ARGS}" ]; then
    echo "Extra eval args: ${DIAG_ARGS}"
fi

if ! checkpoint_exists "${SAVE_DIR}/lit_model.pth"; then
    echo "ERROR: checkpoint not found: ${SAVE_DIR}/lit_model.pth"
    exit 1
fi
if ! checkpoint_finished; then
    echo "ERROR: checkpoint exists but is not marked finished: $(checkpoint_step_label)"
    echo "Refusing eval-only run on an unfinished training checkpoint."
    exit 1
fi

ensure_checkpoint_tokenizer() {
    if has_tokenizer "${SAVE_DIR}"; then
        return 0
    fi
    if [ -z "${TOKENIZER_SOURCE}" ]; then
        return 1
    fi
    if [ "${NODE_RANK}" -eq 0 ]; then
        echo "Checkpoint is missing tokenizer files; copying from ${TOKENIZER_SOURCE}"
        mkdir -p "${SAVE_DIR}"
        cp -f "${TOKENIZER_SOURCE}"/*.json "${SAVE_DIR}/" 2>/dev/null || true
        cp -f "${TOKENIZER_SOURCE}"/*.model "${SAVE_DIR}/" 2>/dev/null || true
    fi
    return 0
}

if ! ensure_checkpoint_tokenizer; then
    echo "ERROR: ${SAVE_DIR} has lit_model.pth but no tokenizer.json/tokenizer.model."
    echo "Set tokenizer_dir in ${CONFIG_FILE} or copy tokenizer files into the checkpoint dir."
    exit 1
fi

BARRIER_DIR="${SAVE_DIR}/.eval_only_barrier_d"
mkdir -p "${BARRIER_DIR}"
touch "${BARRIER_DIR}/node_${NODE_RANK}"

CUR_COUNT=0
LOOP_ITER=0
while [ "${CUR_COUNT}" -lt "${NUM_NODES}" ]; do
    CUR_COUNT=$(ls -1 "${BARRIER_DIR}"/node_* 2>/dev/null | wc -l)
    LOOP_ITER=$((LOOP_ITER + 1))
    echo "Barrier check ${LOOP_ITER}: ${CUR_COUNT}/${NUM_NODES} nodes ready"
    sleep 5
done

echo "All nodes are ready. Starting eval.py"
sleep 30

DEFAULT_BENCHMARKS="boolq,piqa,social_iqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"
DEFAULT_BENCHMARKS="${DEFAULT_BENCHMARKS},mmlu,ceval-valid,ifeval,truthfulqa_gen,truthfulqa_mc1,truthfulqa_mc2"
DEFAULT_BENCHMARKS="${DEFAULT_BENCHMARKS},longbench_2wikimqa,longbench_dureader,longbench_gov_report,longbench_hotpotqa"
DEFAULT_BENCHMARKS="${DEFAULT_BENCHMARKS},longbench_lcc,longbench_lsht,longbench_multi_news,longbench_multifieldqa_en,longbench_multifieldqa_zh"
DEFAULT_BENCHMARKS="${DEFAULT_BENCHMARKS},longbench_musique,longbench_narrativeqa,longbench_passage_count,longbench_passage_retrieval_en"
DEFAULT_BENCHMARKS="${DEFAULT_BENCHMARKS},longbench_qasper,longbench_qmsum,longbench_repobench-p,longbench_samsum,longbench_trec,longbench_triviaqa,longbench_vcsum"
DEFAULT_BENCHMARKS="${DEFAULT_BENCHMARKS},longbench_2wikimqa_e,longbench_gov_report_e,longbench_hotpotqa_e,longbench_lcc_e,longbench_multi_news_e"
DEFAULT_BENCHMARKS="${DEFAULT_BENCHMARKS},longbench_multifieldqa_en_e,longbench_passage_count_e,longbench_passage_retrieval_en_e,longbench_qasper_e"
DEFAULT_BENCHMARKS="${DEFAULT_BENCHMARKS},longbench_repobench-p_e,longbench_samsum_e,longbench_trec_e,longbench_triviaqa_e"

BENCHMARKS=${2:-}
if [ -z "${BENCHMARKS}" ]; then
    BENCHMARKS=${CONFIG_BENCHMARKS:-}
fi
if [ -z "${BENCHMARKS}" ]; then
    BENCHMARKS=${DEFAULT_BENCHMARKS}
fi

NIAH_BENCHMARKS=${3:-niah_single_1,niah_single_2,niah_single_3}
META='{"pretrained": "'"${SAVE_DIR}"'", "max_seq_lengths": [1024, 2048, 4096, 8192, 16384, 32768]}'
MAIN_METADATA_ARGS=()
if [ "${BENCHMARKS}" != "none" ] && [[ "${BENCHMARKS}" == *niah_* ]]; then
    MAIN_METADATA_ARGS=(--metadata "${META}")
fi

if [ "${BENCHMARKS}" != "none" ] && [ -n "${BENCHMARKS}" ]; then
    torchrun \
        --nnodes=${NUM_NODES} \
        --nproc_per_node=${GPUS_PER_NODE} \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        eval.py \
        --checkpoint_dir "${SAVE_DIR}" \
        --benchmark "${BENCHMARKS}" \
        --output_path "${EVAL_OUTPUT_DIR}" \
        "${MAIN_METADATA_ARGS[@]}" \
        "${LOG_KV_ARG_LIST[@]}" \
        ${DIAG_ARGS} \
        ${TOKENIZER_ARGS}

    EVAL_STATUS=$?
    if [ ${EVAL_STATUS} -ne 0 ]; then
        echo "ERROR: main eval failed with exit code ${EVAL_STATUS}"
        exit ${EVAL_STATUS}
    fi
else
    echo "Skipping main benchmark eval."
fi

if [ "${NIAH_BENCHMARKS}" != "none" ] && [ -n "${NIAH_BENCHMARKS}" ]; then
    torchrun \
        --nnodes=${NUM_NODES} \
        --nproc_per_node=${GPUS_PER_NODE} \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        eval.py \
        --checkpoint_dir "${SAVE_DIR}" \
        --benchmark "${NIAH_BENCHMARKS}" \
        --metadata "${META}" \
        --output_path "${EVAL_OUTPUT_DIR}" \
        "${LOG_KV_ARG_LIST[@]}" \
        ${DIAG_ARGS} \
        ${TOKENIZER_ARGS}

    EVAL_STATUS=$?
    if [ ${EVAL_STATUS} -ne 0 ]; then
        echo "ERROR: NIAH eval failed with exit code ${EVAL_STATUS}"
        exit ${EVAL_STATUS}
    fi
else
    echo "Skipping NIAH eval."
fi

if [ ${NODE_RANK} -eq 0 ] && [ -d "${BARRIER_DIR}" ]; then
    rm -rf "${BARRIER_DIR}"
fi

echo "Eval-only job finished."
