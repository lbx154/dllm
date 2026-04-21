#!/usr/bin/env bash
# =============================================================================
# 4-node × 8 × A100-40GB GRPO training for LLaDA-8B-Instruct on GSM8K
# =============================================================================
# Usage:
#   sbatch examples/rl/grpo/llada/train_4node_a100_40g.slurm.sh
#
# What this configures
#   - Base model: GSAI-ML/LLaDA-8B-Instruct  (~16GB bf16 weights)
#   - Dataset:    gsm8k
#   - Parallelism: DeepSpeed ZeRO-3 across 32 GPUs, LoRA (r=128)
#                  No separate reference model — use LoRA disable_adapter trick
#                  to keep only one 16GB weight copy per GPU (after gathering).
#   - Generation: block_size=32, 128 denoising steps, bf16
#   - GRPO:       num_generations=8, num_iterations=8, beta=0.04, epsilon=0.5
#
# Memory budget per 40GB GPU (worst case = generation gather under ZeRO-3):
#   ~16GB full weights (temporarily gathered)
#   + ~4GB LoRA optimizer states
#   + ~8–12GB activations @ seq_len=512, per_device_batch=4
#   => ~32GB peak, safe margin for 40GB.
#
# If you OOM: lower --per_device_train_batch_size to 2,
#             or --max_completion_length to 192,
#             or enable gradient_checkpointing (add `--gradient_checkpointing True`)
# =============================================================================

#SBATCH --job-name=llada-grpo-4n
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --output=./.logs/%x-%j.out
#SBATCH --err=./.logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=3-00:00:00

set -euo pipefail

# ============ Cluster variables ============
NUM_NODES=${SLURM_NNODES}
GPUS_PER_NODE=8
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
NODELIST=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
MASTER_ADDR=${NODELIST[0]}
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

echo "===== System ====="
echo "NUM_NODES=${NUM_NODES}  GPUS_PER_NODE=${GPUS_PER_NODE}  WORLD_SIZE=${WORLD_SIZE}"
echo "MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
printf 'Nodes:\n'; printf '  - %s\n' "${NODELIST[@]}"
echo "=================="

# ============ Runtime env ============
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=13
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
# Reduce PyTorch allocator fragmentation (important for the generation-gather spike)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# ============ Hyperparameters ============
MODEL="GSAI-ML/LLaDA-8B-Instruct"
DATASET="gsm8k"
OUTPUT_DIR=".models/LLaDA-8B-Instruct/grpo-4n-32gpu"

# Global effective batch = WORLD_SIZE * per_device_train_batch_size
#                        = 32 * 4 = 128 completions per optimizer step
# With num_generations=8  => 16 unique prompts per optimizer step.
PER_DEVICE_BS=4
NUM_GENERATIONS=8
GRAD_ACC=1
NUM_ITERATIONS=8      # inner PPO iterations per generation batch

# Diffusion sampler
BLOCK_SIZE=32
STEPS=128
P_MASK_PROMPT=0.15
CFG_SCALE=0.0
MAX_PROMPT_LEN=256
MAX_COMPLETION_LEN=256

# GRPO
LR=3e-6
BETA=0.04
EPSILON=0.5
MAX_STEPS=15000

# LoRA (8B base fits 40GB only with sharding + LoRA; no full FT here)
LORA_R=128
LORA_ALPHA=64
LORA_DROPOUT=0.05

echo "===== Hyperparams ====="
cat <<EOF
model=${MODEL}
dataset=${DATASET}
output_dir=${OUTPUT_DIR}
world_size=${WORLD_SIZE}
per_device_bs=${PER_DEVICE_BS}  num_generations=${NUM_GENERATIONS}  grad_acc=${GRAD_ACC}  iters=${NUM_ITERATIONS}
block_size=${BLOCK_SIZE}  steps=${STEPS}  prompt_len=${MAX_PROMPT_LEN}  comp_len=${MAX_COMPLETION_LEN}
lr=${LR}  beta=${BETA}  epsilon=${EPSILON}  max_steps=${MAX_STEPS}
lora: r=${LORA_R} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT}
EOF
echo "======================="

# ============ Launch ============
# One srun task per node; accelerate fans out to 8 processes inside each node.
srun --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --nodelist="${SLURM_JOB_NODELIST}" \
  accelerate launch \
    --config_file scripts/accelerate_configs/zero3.yaml \
    --num_machines "${NUM_NODES}" \
    --num_processes "${WORLD_SIZE}" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    --machine_rank "${SLURM_PROCID}" \
    --rdzv_backend c10d \
    --mixed_precision bf16 \
    examples/rl/grpo/llada/train.py \
      --model_name_or_path "${MODEL}" \
      --dataset "${DATASET}" \
      --output_dir "${OUTPUT_DIR}" \
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
      --block_size ${BLOCK_SIZE} \
      --steps ${STEPS} \
      --p_mask_prompt ${P_MASK_PROMPT} \
      --cfg_scale ${CFG_SCALE} \
      --beta ${BETA} \
      --epsilon ${EPSILON} \
      --scale_rewards False \
      --sync_ref_model True \
      --ref_model_sync_steps 64 \
      --ref_model_mixup_alpha 1.0 \
      --logging_steps 1 \
      --save_steps 256 \
      --save_total_limit 3 \
      --seed 42 \
      --report_to wandb \
      --run_name "llada8b-grpo-4n-gsm8k"
