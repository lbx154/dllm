#!/usr/bin/env bash
# =============================================================================
# BT-GRPO run5 — single-node 8x GPU restart with the run4 diagnostic fixes.
#
# Differences from the run4 cmdline (see /root/dllm/.logs/btgrpo-run4.log):
#   * num_iterations 1 -> 2          (PPO clip actually engages now)
#   * beta 0.0 -> 0.02 + k2_clipped  (light KL trust-region backstop)
#   * learning_rate 1.5e-6 -> 1e-5   (LoRA-appropriate magnitude)
#   * apply_divergent_mask False     (rule out the 78%-token-zero confound;
#                                     re-enable after we see loss move)
#   * output_dir -> run5             (don't overwrite run4 checkpoints)
#
# Usage:
#   bash /root/dllm/scripts/launch_btgrpo_run5.sh
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

export PYTHONPATH="${REPO_ROOT}"
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

LOG_DIR="${REPO_ROOT}/.logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/btgrpo-run5.log"

ACC_CFG="${REPO_ROOT}/scripts/accelerate_configs/zero2.yaml"

accelerate launch \
  --config_file "${ACC_CFG}" \
  --num_machines 1 \
  --num_processes 8 \
  --mixed_precision bf16 \
  examples/rl/grpo/llada/train_btgrpo.py \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset gsm8k \
    --output_dir .models/llada-btgrpo-run5 \
    --lora_r 64 --lora_alpha 32 --lora_dropout 0.05 \
    --bf16 True \
    --learning_rate 3e-6 \
    --max_steps 8000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_generations 4 \
    --num_iterations 1 \
    --max_completion_length 200 \
    --fork_frac 0.5 \
    --block_size 32 \
    --steps 64 \
    --p_mask_prompt 0.15 \
    --beta 0.0 \
    --epsilon 0.2 \
    --apply_divergent_mask False \
    --reward_weights 0.25 0.25 0.25 0.25 5.0 \
    --scale_rewards True \
    --sync_ref_model False \
    --logging_steps 1 --save_steps 256 --save_total_limit 3 \
    --seed 42 \
    --run_name btgrpo-run5-1node \
    --report_to none \
  >> "${LOG}" 2>&1
