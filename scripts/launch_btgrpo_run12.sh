#!/usr/bin/env bash
# =============================================================================
# BT-GRPO run12 — deep-diagnosis restart after run11's fork-head dead-end.
#
# Findings that drove this config (see docs/RUN_HISTORY.md §run11 + §run12):
#   (A) run11 correctness only rose +0.027 over 243 steps (S/N = -0.5 vs raw
#       per-step σ); underlying base-GRPO was learning *too slowly to see*.
#   (B) 98% of completions hit the 266-token cap -> </answer> truncated ->
#       format + correctness parsers fail silently on completions that did
#       solve the math.
#   (C) xmlcount_reward_func had a "penalty for tail text after </answer>"
#       that made its per-step mean NEGATIVE; counter-acted correctness.
#       (code fix in format.py, now pure +0.5 on well-formed structure)
#   (D) strict_format_reward_func permanently 0 -> 12.5% of reward weight
#       was dead.  Reweighted to 0.
#   (E) num_iterations=1 + beta=0 gave only 1 gradient update per batch;
#       run1 (num_iter=4 + beta=0.02) reached corr=0.47 early but was
#       never re-tried without KL.  Going back to num_iter=4 with beta=0.
#   (F) block_size 32 + max_comp 512 would cost 9x sequential chunks; bump
#       block_size to 64 to halve sampling wall-clock on the B200s.
#   (G) G=4 + per_device_bs=4 gave only 16 rollouts/step/gpu -> 22% of
#       groups had zero intra-group std -> signal starved.  Double to
#       G=8 + per_device_bs=8 (keeps per_device_bs == G so each gpu
#       holds exactly 1 fork group).
#   (H) learned fork_frac head disabled: mean-clamp zeroed its gradient
#       once saturated, and the per-batch reward signal was too coarse
#       for the head to learn anyway.  Park the idea until base learns.
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
LOG="${LOG_DIR}/btgrpo-run12.log"

ACC_CFG="${REPO_ROOT}/scripts/accelerate_configs/zero2.yaml"

accelerate launch \
  --config_file "${ACC_CFG}" \
  --num_machines 1 \
  --num_processes 8 \
  --mixed_precision bf16 \
  examples/rl/grpo/llada/train_btgrpo.py \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset gsm8k \
    --output_dir .models/llada-btgrpo-run12 \
    --lora_r 64 --lora_alpha 32 --lora_dropout 0.05 \
    --bf16 True \
    --learning_rate 3e-6 \
    --max_steps 8000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_generations 8 \
    --num_iterations 4 \
    --max_completion_length 512 \
    --fork_frac 0.5 \
    --learn_fork_frac False \
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
    --run_name btgrpo-run12-1node \
    --report_to none \
  >> "${LOG}" 2>&1
