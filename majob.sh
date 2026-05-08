#!/bin/bash
# source /home/miniconda3/bin/activate megatron-lm-014
source /home/ma-user/anaconda3/bin/activate torch218
# DeepSeek 3Bv2 Sandwich Training Script with YAML Configuration
# Optimized for H200 GPUs (141GB VRAM)

export CUDA_DEVICE_MAX_CONNECTIONS=32
export CUDNN_LOGERR_DBG=1
export CUDNN_LOGDEST_DBG=stderr

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
        echo "❌ [Node ${NODE_RANK}] 训练阶段崩溃或被手动中断 (Exit Code: $TRAIN_STATUS)，已拦截后续评测！"
        exit $TRAIN_STATUS
    fi

    echo "🎉 [Node ${NODE_RANK}] 阶段一完成！模型已成功保存至 ${SAVE_DIR}"
    
    # 强制操作系统彻底回收显存
    sleep 15
fi


echo "================================================="
echo "🎉 [Node ${NODE_RANK}] 阶段一完成！模型已成功保存至 ${SAVE_DIR}"
echo "🚀 [Node ${NODE_RANK}] 阶段二：释放显存，准备执行分布式自动评测"
echo "================================================="

# 🌟 核心缓冲：休眠 60 秒
# 1. 强制操作系统彻底回收 demo.py 占用的所有显存
# 2. 补偿不同 Node 之间保存 checkpoint 到共享存储 (SFS/NAS) 时的极其微小的时间差
sleep 60

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
    --output_path "${EVAL_OUTPUT_DIR}"

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
    --output_path "${EVAL_OUTPUT_DIR}"

EVAL_STATUS=$?
if [ $EVAL_STATUS -ne 0 ]; then
    echo "❌ [Node ${NODE_RANK}] 评测阶段崩溃 (Exit Code: $EVAL_STATUS)！"
    exit $EVAL_STATUS
fi

echo "================================================="
echo "🎊 [Node ${NODE_RANK}] 全部流水线 (Training + Evaluation) 执行完美结束！"
echo "================================================="

set +x