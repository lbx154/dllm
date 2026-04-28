#!/usr/bin/env bash
# =============================================================================
# BT-GRPO run21 — first canary-oracle-guided config.
#
# Designed by HealthOracle grid search over run20 baseline (P_any=0.99) to
# minimise predicted failure probability across 6 signatures.  Resulting
# prediction (LOO LR over runs 5–19):
#
#     P(any failure within 50 steps) = 0.37  (was 0.99 on run20)
#       starved_signal      0.31   (was 0.98) - re-enable strict reward
#       fork_saturated      0.00   (was 0.03) - freeze fork-head
#       len_collapsing      0.01   (was 0.17) - flatten easy rewards
#       corr_negative_slope 0.08   (was 0.18) - narrower fork_frac
#       grad_blowup / corr_dead_early ≈ 0
#
# Diffs vs run20:
#   - reward_weights      1.0 1.0 0.0 1.0 2.0  →  0.5 0.5 1.0 0.5 2.0
#                          (re-enable strict_format, flatten easy shadows;
#                           reward ratio 8× → 4×)
#   - learn_fork_frac     True   →  False     (freeze fork head; biggest
#                                              +coef on fork_saturated)
#   - fork_frac           0.3    →  0.2       (lower corr_negative_slope risk)
#   - max_completion_length 512  →  384       (lower starved_signal risk)
#   - seed                43     →  44
#
# Launch via the canary stack:
#     bash scripts/launch_with_canary.sh scripts/launch_btgrpo_run21.sh
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
LOG="${LOG_DIR}/btgrpo-run21.log"

ACC_CFG="${REPO_ROOT}/scripts/accelerate_configs/zero2.yaml"

accelerate launch \
  --config_file "${ACC_CFG}" \
  --num_machines 1 \
  --num_processes 8 \
  --mixed_precision bf16 \
  examples/rl/grpo/llada/train_btgrpo.py \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset gsm8k \
    --output_dir .models/llada-btgrpo-run21 \
    --lora_r 64 --lora_alpha 32 --lora_dropout 0.05 \
    --bf16 True \
    --learning_rate 3e-6 \
    --max_steps 8000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_generations 8 \
    --num_iterations 1 \
    --max_completion_length 384 \
    --fork_frac 0.2 \
    --learn_fork_frac False \
    --fork_frac_min 0.02 --fork_frac_max 0.7 \
    --fork_head_lr 1e-3 \
    --block_size 64 \
    --steps 64 \
    --p_mask_prompt 0.15 \
    --beta 0.0 \
    --epsilon 0.2 \
    --apply_divergent_mask False \
    --reward_weights 0.5 0.5 1.0 0.5 2.0 \
    --scale_rewards False \
    --filter_zero_std_groups True \
    --filter_zero_correct_groups True \
    --log_rollouts_every 25 \
    --sync_ref_model False \
    --logging_steps 1 --save_steps 256 --save_total_limit 3 \
    --seed 44 \
    --run_name btgrpo-run21-1node \
    --report_to none \
  >> "${LOG}" 2>&1
