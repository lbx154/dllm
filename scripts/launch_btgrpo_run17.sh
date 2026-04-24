#!/usr/bin/env bash
# =============================================================================
# BT-GRPO run17 — rebalance reward weights (post-run16)
#
# Diagnosis of run16 (~20 steps):
#   The answer-extraction fix worked: correctness mean jumped from run15's
#   ~0.01 to 0.25–1.25 per batch. But with the legacy weights
#       [xmlcount 0.25, soft 0.25, strict 0.0, int 0.25, correctness 5.0]
#   correctness now dominates the weighted reward ~27:1 over all format
#   rewards combined (10.0 vs 0.375 for a fully-correct, fully-formatted
#   rollout). This was harmless when correctness was dead, but now it
#   squashes the format learning signal — and on LLaDA specifically we
#   *need* the model to adopt `<answer>…</answer>` for the strict extractor
#   and downstream evaluation to work.
#
# Change:
#   --reward_weights 0.25 0.25 0.0 0.25 5.0
#   → --reward_weights 1.0  1.0  0.0 1.0  2.0
#   Per fully-correct rollout: correctness = 2.0×2.0 = 4.0 ; format total
#   (xmlcount+soft+int, all at ≈0.5 max) = 1.5. Ratio ≈ 2.7:1 — correctness
#   still dominates but format is no longer a rounding error.
#
# Kept from run16:
#   lenient answer parser, scale_rewards=False, filter_zero_std_groups,
#   filter_zero_correct_groups, log_rollouts_every=25, fork_frac ∈ [0.02,
#   0.98], G=8, bs=8, lr=3e-6, β=0, ε=0.2, lora_r=64.
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
LOG="${LOG_DIR}/btgrpo-run17.log"

ACC_CFG="${REPO_ROOT}/scripts/accelerate_configs/zero2.yaml"

accelerate launch \
  --config_file "${ACC_CFG}" \
  --num_machines 1 \
  --num_processes 8 \
  --mixed_precision bf16 \
  examples/rl/grpo/llada/train_btgrpo.py \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset gsm8k \
    --output_dir .models/llada-btgrpo-run17 \
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
    --run_name btgrpo-run17-1node \
    --report_to none \
  >> "${LOG}" 2>&1
