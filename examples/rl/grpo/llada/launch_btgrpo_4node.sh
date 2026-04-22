#!/usr/bin/env bash
# =============================================================================
# BT-GRPO 4-node × 8×A100-40GB launch for AzureML (torchrun over SSH)
# =============================================================================
# Assumes:
#   * /home/aiscuser on each node has the dllm repo cloned at the same path
#   * conda env /opt/conda/envs/ptca has all deps installed on each node
#   * Passwordless SSH node-0 -> node-{1,2,3} works
#   * 4 nodes named node-0 .. node-3 in hostfile
#
# Usage:
#   bash examples/rl/grpo/llada/launch_btgrpo_4node.sh
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/../../../.."        # repo root
REPO_ROOT="$(pwd)"
PY="/opt/conda/envs/ptca/bin/python"
NNODES=4
NPROC_PER_NODE=8
MASTER_ADDR="${MASTER_ADDR:-node-0}"
MASTER_PORT="${MASTER_PORT:-29500}"

export PYTHONPATH="${REPO_ROOT}"
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=13
# Keep PyTorch allocator happy during the generation-gather spike
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
# Local HuggingFace cache on each node (no shared blob mount across nodes)
export HF_HOME="${HF_HOME:-/scratch/hf_cache}"
mkdir -p "${HF_HOME}"

# ---- Hyperparameters (see docs/BT_GRPO.md §5) ----
MODEL="GSAI-ML/LLaDA-8B-Instruct"
DATASET="gsm8k"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/dllm_runs/llada-btgrpo-4n}"
mkdir -p "${OUTPUT_DIR}"

PER_DEVICE_BS=4
NUM_GENERATIONS=4           # == fork group G (must equal PER_DEVICE_BS so each GPU holds one full fork group)
NUM_ITERATIONS=1            # pure on-policy — matches BT-GRPO's per-step-credit design best + 4x faster
GRAD_ACC=1

FORK_FRAC=0.5               # back to run1 — keeps CoT coherence before fork
BLOCK_SIZE=32
STEPS=64
P_MASK_PROMPT=0.15
MAX_PROMPT_LEN=256
MAX_COMPLETION_LEN=200

LR=1.5e-6
BETA=0.0                    # KILL KL term entirely — k3 estimator explodes on dLLM (r in [-50,50])
                            # and even bounded KL dominated loss 1000:1 over reward gradient.
                            # PPO clip below is our only trust region (original DeepSeek GRPO setting).
EPSILON=0.2                 # tight PPO clip (was 0.3) — sole trust-region mechanism now
MAX_STEPS=8000

LORA_R=64                   # halved LoRA capacity → smaller effective step size, less overshoot
LORA_ALPHA=32
LORA_DROPOUT=0.05

# ---- Accelerate config (ZeRO-3) ----
ACC_CFG="${REPO_ROOT}/scripts/accelerate_configs/zero2.yaml"

# ---- Per-node launch function ----
LOG_DIR="${REPO_ROOT}/.logs"
mkdir -p "${LOG_DIR}"
RUN_TAG="btgrpo-$(date +%Y%m%d-%H%M%S)"

launch_cmd() {
    local node_rank=$1
    cat <<EOF
cd ${REPO_ROOT} && \
export PYTHONPATH=${REPO_ROOT} && \
export HF_HOME=/scratch/hf_cache && \
export TOKENIZERS_PARALLELISM=false && \
export NCCL_ASYNC_ERROR_HANDLING=1 && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 && \
${PY} -m accelerate.commands.launch \
  --config_file ${ACC_CFG} \
  --num_machines ${NNODES} \
  --num_processes $((NNODES * NPROC_PER_NODE)) \
  --machine_rank ${node_rank} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --rdzv_backend c10d \
  --mixed_precision bf16 \
  examples/rl/grpo/llada/train_btgrpo.py \
    --model_name_or_path ${MODEL} \
    --dataset ${DATASET} \
    --output_dir ${OUTPUT_DIR} \
    --lora_r ${LORA_R} --lora_alpha ${LORA_ALPHA} --lora_dropout ${LORA_DROPOUT} \
    --bf16 True \
    --learning_rate ${LR} \
    --max_steps ${MAX_STEPS} \
    --per_device_train_batch_size ${PER_DEVICE_BS} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --num_generations ${NUM_GENERATIONS} \
    --num_iterations ${NUM_ITERATIONS} \
    --max_prompt_length ${MAX_PROMPT_LEN} \
    --max_completion_length ${MAX_COMPLETION_LEN} \
    --fork_frac ${FORK_FRAC} \
    --block_size ${BLOCK_SIZE} \
    --steps ${STEPS} \
    --p_mask_prompt ${P_MASK_PROMPT} \
    --beta ${BETA} \
    --epsilon ${EPSILON} \
    --reward_weights 0.25 0.25 0.25 0.25 5.0 \
    --scale_rewards False \
    --sync_ref_model False \
    --logging_steps 1 --save_steps 256 --save_total_limit 3 \
    --seed 42 \
    --run_name "${RUN_TAG}" \
    --report_to none
EOF
}

# ---- Fan out to node-1..3 via ssh (backgrounded), then run node-0 in fg ----
for rk in 1 2 3; do
    CMD=$(launch_cmd $rk)
    LOG="${LOG_DIR}/${RUN_TAG}-node${rk}.log"
    echo "[launcher] ssh node-${rk} -> ${LOG}"
    ssh -o StrictHostKeyChecking=no -f "node-${rk}" \
        "bash -lc '${CMD}' > ${LOG} 2>&1" &
done
wait

# Rank 0 runs in foreground so we see logs directly and keep the script alive
CMD0=$(launch_cmd 0)
LOG0="${LOG_DIR}/${RUN_TAG}-node0.log"
echo "[launcher] node-0 foreground, tee -> ${LOG0}"
bash -lc "${CMD0}" 2>&1 | tee "${LOG0}"
