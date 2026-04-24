#!/usr/bin/env bash
# =============================================================================
# BT-GRPO run15 — kill the advantage-noise loop
#
# Diagnosis of run14 (see dashboard @ step 608):
#
#   Live log evidence:
#     rewards/correctness_reward_func/mean  = 0.0   (std=0.0)   [most batches]
#     rewards/xmlcount_reward_func/mean     = 0.25  (std=0.14)
#     reward_std                            = 0.015–0.03
#     frac_reward_zero_std                  = 0.625–0.875
#     Policy entropy                        : 0.30 → 0.45 (rising = degrading)
#     Completion length                     : 200  → 150 (learning short+fake)
#
#   Root cause chain:
#     (a) Base-model pass rate on GSM8K ≈ 1–2% at G=8, so 70–90% of groups
#         have the correctness signal collapse (std=0 on the one reward that
#         matters).
#     (b) `scale_rewards=True` + tiny reward_std (~0.02) divides the remaining
#         tiny format-reward variance into ±1 advantages: noise amplifier.
#     (c) Even with scale_rewards off, the zero-correct-std groups still
#         feed their format-only residual into the loss denominator, diluting
#         the real signal from the mixed groups.
#
# Fix (trainer.py):
#   • scale_rewards=False         — remove the /std noise amplifier.
#   • filter_zero_std_groups      — drop groups with zero total-reward std.
#   • filter_zero_correct_groups  — drop groups where *correctness* alone is
#                                   zero-std. These are the "format-only
#                                   noise" groups; we want 0 gradient from
#                                   them, both numerator and denominator.
#   • log_rollouts_every=100      — dump one prompt/completion/extracted/gt
#                                   tuple periodically to verify the parser
#                                   is matching the model output (sanity).
#
# Kept from run14:
#   G=8, per_device_bs=8, block_size=64, max_completion_length=512,
#   num_iterations=1, learn_fork_frac=True (sigmoid-bounded, value-coupled),
#   beta=0, epsilon=0.2, lora_r=64, lr=3e-6, reward_weights unchanged so
#   run14 vs run15 is a clean A/B on the noise-suppression machinery.
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
LOG="${LOG_DIR}/btgrpo-run15.log"

ACC_CFG="${REPO_ROOT}/scripts/accelerate_configs/zero2.yaml"

accelerate launch \
  --config_file "${ACC_CFG}" \
  --num_machines 1 \
  --num_processes 8 \
  --mixed_precision bf16 \
  examples/rl/grpo/llada/train_btgrpo.py \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset gsm8k \
    --output_dir .models/llada-btgrpo-run15 \
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
    --reward_weights 0.25 0.25 0.0 0.25 5.0 \
    --scale_rewards False \
    --filter_zero_std_groups True \
    --filter_zero_correct_groups True \
    --log_rollouts_every 100 \
    --sync_ref_model False \
    --logging_steps 1 --save_steps 256 --save_total_limit 3 \
    --seed 42 \
    --run_name btgrpo-run15-1node \
    --report_to none \
  >> "${LOG}" 2>&1
