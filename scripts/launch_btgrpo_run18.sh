#!/usr/bin/env bash
# =============================================================================
# BT-GRPO run18 — fix ForkHead value_coupling init (1.0 → 0.0)
#
# Diagnosis of run17 (~360 steps, reward flat at MA20 ≈ 1.5):
#   * rewards/correctness_reward_func/mean MA20 stuck at ~0.55 the whole run
#     (no upward trend from step 20 to 360).
#   * btgrpo/fork_frac_mean spent the first ~150 steps in [0.75, 0.92], then
#     REINFORCE overshot and it collapsed into [0.03, 0.15] — bang-bang, not
#     learning.
#   * frac_groups_filtered MA20 ≈ 0.97 for the first 150 steps, then slowly
#     dropping to ~0.83. In other words: for most of the run almost every
#     group was filtered out (whole-group-wrong or whole-group-right on
#     correctness), so the main-loop gradient signal was a trickle.
#
#   Root cause: ForkHead.value_coupling was initialised at +1.0 and the value
#   head is trained by MSE toward the mean reward (~1.5). Within a few dozen
#   steps V(z) ≈ 1.5, so the policy mean becomes
#       mean = lo + (hi-lo) * sigmoid(proj(z) + 1 * V(z))
#            = 0.02 + 0.96 * sigmoid(~1.5)
#            ≈ 0.02 + 0.96 * 0.82 ≈ 0.81
#   Phase-1 ≈ 0.8 means the 8 fork siblings only diverge in the last 20% of
#   denoising and therefore emit nearly identical completions → identical
#   correctness → zero std → filtered out.
#
# Change:
#   fork_head.py: value_coupling init 1.0 → 0.0.
#   At t=0, mean = sigmoid(proj.bias)=0.5 regardless of V, matching the
#   legacy fixed fork_frac=0.5 default. REINFORCE can still grow
#   value_coupling if the difficulty hypothesis is borne out by data.
#
# Kept from run17:
#   reward_weights [1.0 1.0 0.0 1.0 2.0], lenient answer parser,
#   scale_rewards=False, filter_zero_std_groups, filter_zero_correct_groups,
#   log_rollouts_every=25, fork_frac ∈ [0.02, 0.98], G=8, bs=8, lr=3e-6,
#   β=0, ε=0.2, lora_r=64.
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
LOG="${LOG_DIR}/btgrpo-run18.log"

ACC_CFG="${REPO_ROOT}/scripts/accelerate_configs/zero2.yaml"

accelerate launch \
  --config_file "${ACC_CFG}" \
  --num_machines 1 \
  --num_processes 8 \
  --mixed_precision bf16 \
  examples/rl/grpo/llada/train_btgrpo.py \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset gsm8k \
    --output_dir .models/llada-btgrpo-run18 \
    --lora_r 64 --lora_alpha 32 --lora_dropout 0.05 \
    --bf16 True \
    --learning_rate 3e-6 \
    --max_steps 8000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_generations 8 \
    --num_iterations 1 \
    --max_completion_length 512 \
    --fork_frac 0.5 \
    --learn_fork_frac True \
    --fork_frac_min 0.02 --fork_frac_max 0.98 \
    --fork_head_lr 1e-3 \
    --block_size 64 \
    --steps 64 \
    --p_mask_prompt 0.15 \
    --beta 0.0 \
    --epsilon 0.2 \
    --apply_divergent_mask False \
    --reward_weights 1.0 1.0 0.0 1.0 2.0 \
    --scale_rewards False \
    --filter_zero_std_groups True \
    --filter_zero_correct_groups True \
    --log_rollouts_every 25 \
    --sync_ref_model False \
    --logging_steps 1 --save_steps 256 --save_total_limit 3 \
    --seed 42 \
    --run_name btgrpo-run18-1node \
    --report_to none \
  >> "${LOG}" 2>&1
