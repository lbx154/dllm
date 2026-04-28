#!/usr/bin/env bash
# =============================================================================
# Vanilla GRPO run1 — original recipe, NO BT-GRPO extensions
#
# Purpose:
#   Sanity-check whether the BASE GRPO trainer (no fork-head, no divergent_mask,
#   no Bradley-Terry pairing) can train LLaDA-8B-Instruct on GSM8K. If vanilla
#   trains stably while BT-GRPO repeatedly fails, the failure is in the BT-GRPO
#   layer; if vanilla also fails, the issue is deeper (KL estimator, dLLM
#   logprob structure, or LoRA target modules).
#
# Recipe is faithful to examples/rl/grpo/llada/train.py docstring:
#   - load_in_4bit True            (bitsandbytes 4-bit NF4 quant of base weights)
#   - lora_r 128, alpha 64
#   - β=0.04, ε=0.5
#   - num_generations=6, num_iterations=12  (12 PPO iters per rollout batch)
#   - block_size=32, steps=128                (smaller block, more steps)
#   - sync_ref_model=False (forced — current TRL disallows True with PEFT;
#                           docstring's True is a stale recipe). With LoRA the
#                           ref is recovered by disabling the adapter.
#
# Single-node 8 × B200 (180GB ea). With 4-bit base + LoRA r=128, peak mem
# ~12GB/GPU, well under budget.
#
# Differences from BT-GRPO runs (run1..run22):
#   - examples/rl/grpo/llada/train.py     (vs train_btgrpo.py)
#   - no --fork_frac, --apply_divergent_mask, --filter_zero_correct_groups
#   - num_iterations=12 (PPO style) vs 1 (on-policy GRPO style)
#   - epsilon 0.5 vs 0.2 (looser PPO clip; matches the orig recipe)
#   - block_size 32, steps 128 (vs 64/64 in run22)
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    --num_machines 1 \
    --num_processes 8 \
    --mixed_precision bf16 \
    examples/rl/grpo/llada/train.py \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --load_in_4bit True \
    --dataset gsm8k \
    --output_dir .models/llada-grpo-vanilla-run1 \
    --lora_r 128 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --bf16 True \
    --learning_rate 3e-6 \
    --max_steps 8000 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 2 \
    --num_generations 6 \
    --num_iterations 12 \
    --max_completion_length 384 \
    --block_size 32 \
    --steps 128 \
    --p_mask_prompt 0.15 \
    --beta 0.04 \
    --epsilon 0.5 \
    --sync_ref_model False \
    --logging_steps 1 \
    --save_steps 256 \
    --save_total_limit 3 \
    --seed 47 \
    --run_name grpo-vanilla-run1-1node \
    --report_to none
