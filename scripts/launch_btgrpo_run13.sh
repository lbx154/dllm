#!/usr/bin/env bash
# =============================================================================
# BT-GRPO run13 — re-enable learned fork_frac with range [0,1] + init 0.5,
# after fixing the mean-clamp gradient bug (FORK_HEAD.md §5.5).
#
# Changes vs run12:
#   (1) --learn_fork_frac True  (head re-enabled)
#   (2) --fork_frac_min 0.0 --fork_frac_max 1.0  (full range; init bias at
#       0.5*(lo+hi) = 0.5 keeps behaviour identical to hand-picked 0.5 at
#       step 0, so if the head misbehaves we degrade gracefully to run12)
#   (3) fork_head.py: removed `mean = raw_mean.clamp(lo,hi)` in _mean_sigma
#       — the clamp severed gradient once raw_mean exited [lo,hi] and was
#       the mechanism by which run11 got glued at μ=0.800.  Now only the
#       sampled action is clamped; log_prob stays differentiable.
#   (4) --num_iterations 4 -> 2
#       run12 showed 3/14 batches with grad_norm spikes to 1000-35000
#       (PPO clip is asymmetric: bounds positive advantage with huge
#       ratio but NOT negative advantage with huge ratio).  Halving the
#       reuse count halves the expected ratio drift per batch while
#       still keeping 2x the gradient density of num_iter=1.
#
# Kept from run12 (validated: xmlcount mean +0.34 vs -0.05 on run11,
# completion length mean ~230 with <5% hit 512 cap):
#   max_completion_length=512, block_size=64, G=8, per_device_bs=8,
#   reward_weights with strict_format=0, scale_rewards=True, beta=0,
#   epsilon=0.2, lora_r=64, lr=3e-6.
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
LOG="${LOG_DIR}/btgrpo-run13.log"

ACC_CFG="${REPO_ROOT}/scripts/accelerate_configs/zero2.yaml"

accelerate launch \
  --config_file "${ACC_CFG}" \
  --num_machines 1 \
  --num_processes 8 \
  --mixed_precision bf16 \
  examples/rl/grpo/llada/train_btgrpo.py \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset gsm8k \
    --output_dir .models/llada-btgrpo-run13 \
    --lora_r 64 --lora_alpha 32 --lora_dropout 0.05 \
    --bf16 True \
    --learning_rate 3e-6 \
    --max_steps 8000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_generations 8 \
    --num_iterations 2 \
    --max_completion_length 512 \
    --fork_frac 0.5 \
    --learn_fork_frac True \
    --fork_frac_min 0.0 --fork_frac_max 1.0 \
    --fork_head_lr 1e-3 \
    --block_size 64 \
    --steps 64 \
    --p_mask_prompt 0.15 \
    --beta 0.0 \
    --epsilon 0.2 \
    --apply_divergent_mask False \
    --reward_weights 0.25 0.25 0.0 0.25 5.0 \
    --scale_rewards True \
    --sync_ref_model False \
    --logging_steps 1 --save_steps 256 --save_total_limit 3 \
    --seed 42 \
    --run_name btgrpo-run13-1node \
    --report_to none \
  >> "${LOG}" 2>&1
