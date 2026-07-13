#!/bin/bash
# ==============================================================================
# Training + Evaluation pipeline for logKV models
#
# Usage:
#   bash majob.sh exp/qwen0.6b-32k/cpt-base.yaml
#
# The YAML config must contain a 'save_path' key.
# If the checkpoint already exists (save_path/lit_model.pth), training is skipped
# and evaluation runs directly.
# ==============================================================================

set -e

# ── Environment setup ──
source /home/ma-user/anaconda3/bin/activate torch218 2>/dev/null || true

export CUDA_DEVICE_MAX_CONNECTIONS=32
export PATH=/usr/local/cuda-12.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}

# ── Distributed setup ──
GPUS_PER_NODE=${MA_NUM_GPUS:-8}
NUM_NODES=${MA_NUM_HOSTS:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
if [ -n "${MA_VJ_NAME:-}" ]; then
    MASTER_ADDR="${MA_VJ_NAME}-${MA_TASK_NAME}-${MA_MASTER_INDEX:-0}.${MA_VJ_NAME}"
fi
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${VC_TASK_INDEX:-0}

echo "Node ${NODE_RANK}/${NUM_NODES} | Master: ${MASTER_ADDR}:${MASTER_PORT}"

# ── Config validation ──
if [ -z "$1" ]; then
    echo "ERROR: No YAML config provided."
    echo "Usage: bash $0 <yaml_config_path>"
    exit 1
fi

CONFIG_FILE=$1
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Loading config from: $CONFIG_FILE"

# Extract save_path from YAML
SAVE_DIR=$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG_FILE}'))['save_path'])")
eval SAVE_DIR="\"${SAVE_DIR}\""

if [ -z "$SAVE_DIR" ] || [ "$SAVE_DIR" == "None" ]; then
    echo "ERROR: 'save_path' not found in ${CONFIG_FILE}"
    exit 1
fi

echo "Save path: ${SAVE_DIR}"

# Extract logKV settings from the training YAML (following 'config:' inheritance)
# so evaluation runs with the same attention the model was adapted to.
read -r LOG_KV_TRAINING LOG_KV_B LOG_KV_RECENT <<< "$(python - "${CONFIG_FILE}" <<'EOF'
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
print(
    str(bool(cfg.get("log_kv_training", False))).lower(),
    cfg.get("log_kv_B", 512),
    cfg.get("log_kv_recent_size", 1024),
)
EOF
)"

LOG_KV_ARGS=""
if [ "${LOG_KV_TRAINING}" == "true" ]; then
    LOG_KV_ARGS="--use_log_kv true --log_kv_B ${LOG_KV_B} --log_kv_recent_size ${LOG_KV_RECENT}"
    echo "logKV eval ENABLED: B=${LOG_KV_B}, recent_size=${LOG_KV_RECENT}"
else
    echo "logKV eval disabled (log_kv_training not set in ${CONFIG_FILE})"
fi

# ==============================================================================
# Phase 1: Training (skip if checkpoint exists)
# ==============================================================================
if [ -f "${SAVE_DIR}/lit_model.pth" ]; then
    echo "================================================="
    echo "Checkpoint exists at ${SAVE_DIR}/lit_model.pth"
    echo "Skipping training, going directly to evaluation."
    echo "================================================="
else
    echo "================================================="
    echo "Phase 1: Continue Pre-Training"
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
        echo "ERROR: Training failed (exit code: $TRAIN_STATUS)"
        exit $TRAIN_STATUS
    fi

    echo "Phase 1 complete. Model saved to ${SAVE_DIR}"
    sleep 15
fi

# ==============================================================================
# Phase 2: Evaluation
# ==============================================================================
echo "================================================="
echo "Phase 2: Evaluation"
echo "================================================="

sleep 30  # Allow FSDP cleanup across nodes

EVAL_OUTPUT_DIR="${SAVE_DIR}/evaluate"

# Standard benchmarks
BENCHMARKS="boolq,piqa,social_iqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"
BENCHMARKS="${BENCHMARKS},mmlu,ceval-valid,ifeval,truthfulqa_gen,truthfulqa_mc1,truthfulqa_mc2"

# LongBench tasks
BENCHMARKS="${BENCHMARKS},longbench_2wikimqa,longbench_dureader,longbench_gov_report,longbench_hotpotqa"
BENCHMARKS="${BENCHMARKS},longbench_lcc,longbench_lsht,longbench_multi_news,longbench_multifieldqa_en"
BENCHMARKS="${BENCHMARKS},longbench_multifieldqa_zh,longbench_musique,longbench_narrativeqa"
BENCHMARKS="${BENCHMARKS},longbench_passage_count,longbench_passage_retrieval_en"
BENCHMARKS="${BENCHMARKS},longbench_qasper,longbench_qmsum,longbench_repobench-p,longbench_samsum"
BENCHMARKS="${BENCHMARKS},longbench_trec,longbench_triviaqa,longbench_vcsum"
BENCHMARKS="${BENCHMARKS},longbench_2wikimqa_e,longbench_gov_report_e,longbench_hotpotqa_e"
BENCHMARKS="${BENCHMARKS},longbench_lcc_e,longbench_multi_news_e,longbench_multifieldqa_en_e"
BENCHMARKS="${BENCHMARKS},longbench_passage_count_e,longbench_passage_retrieval_en_e"
BENCHMARKS="${BENCHMARKS},longbench_qasper_e,longbench_repobench-p_e,longbench_samsum_e"
BENCHMARKS="${BENCHMARKS},longbench_trec_e,longbench_triviaqa_e"

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
    ${LOG_KV_ARGS}

EVAL_STATUS=$?
if [ $EVAL_STATUS -ne 0 ]; then
    echo "ERROR: Evaluation failed (exit code: $EVAL_STATUS)"
    exit $EVAL_STATUS
fi

echo "================================================="
echo "Pipeline complete!"
echo "================================================="
