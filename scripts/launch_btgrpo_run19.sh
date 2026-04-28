#!/usr/bin/env bash
# =============================================================================
# BT-GRPO run19 — strict-XML correctness reward + earlier fork
#
# Diagnosis of run17 (~4400 steps) and run18 (10 steps, killed):
#   * Reward decayed monotonically: 0.92 (step 200) → 0.08 (step 1200+),
#     and stayed there. By step ~4000 every group had `frac_correct_zero_std=1.0`
#     and the model had drifted to producing `<Reasoning>` (capital R) headers
#     with NO `<answer>` tags — completions look like
#         "<Reasoning>\nToargamel bought 4 tires ... was8"
#     i.e. format collapsed to a fragment ending in a random digit.
#
#   Two coupled root causes:
#
#   (a) `correctness_reward_func` used `extract_answer_lenient`, which falls
#       back to "last number anywhere in the completion". A completion with
#       NO XML tags but a lucky trailing number got correctness=2.0, while a
#       well-formatted-but-wrong completion got 0. With weights
#       [xml 1, soft 1, strict 0, int 1, corr 2] the cost/benefit was:
#         well-formatted but answer wrong : 0.5+0.5+0.5+0    = 1.5
#         format-collapsed but lucky digit: 0  +0  +0  +2.0  = 2.0
#       → gradient pushed the model to abandon the XML wrapper. Once the
#       wrapper is gone, `int_reward` (strict) and `soft_format` are both
#       always 0, only `xmlcount` and `correctness` survive, and the
#       lenient correctness signal becomes pure noise (any completion with
#       digits in it is in the running).
#
#   (b) `frac_correct_zero_std` was ≥0.97 most of the time even at fork_frac=0.5
#       — fork siblings rarely diverged on correctness because LLaDA's block
#       decoding + low_confidence remasking commits the answer-bearing tokens
#       in Phase 1, before the fork. With `filter_zero_correct_groups=True`
#       this means almost no groups produced gradient, and the few that did
#       were exactly the borderline cases where (a) was loudest — amplifying
#       the format-collapse drift.
#
# Changes vs run18:
#   1. dllm/pipelines/rl/grpo/rewards/math.py: `correctness_reward_func`
#      switched to a strict `<answer>...</answer>` extractor. No `<answer>`
#      tag → correctness=0. This restores a real format anchor: lose the
#      wrapper, lose the reward.
#   2. fork_frac 0.5 → 0.3 (start forking earlier so more denoising happens
#      independently per sibling; should reduce frac_correct_zero_std).
#   3. fork_frac_max 0.98 → 0.7 (cap how late the head can push the fork —
#      fork_frac>0.7 reliably produces near-identical siblings, no point
#      letting the head learn it).
#
# Kept from run18:
#   reward_weights [1.0 1.0 0.0 1.0 2.0], scale_rewards=False,
#   filter_zero_std_groups=True, filter_zero_correct_groups=True,
#   ForkHead value_coupling init = 0 (run18 fix), G=8, bs=8, lr=3e-6,
#   β=0, ε=0.2, lora_r=64, max_completion_length=512.
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
LOG="${LOG_DIR}/btgrpo-run19.log"

ACC_CFG="${REPO_ROOT}/scripts/accelerate_configs/zero2.yaml"

accelerate launch \
  --config_file "${ACC_CFG}" \
  --num_machines 1 \
  --num_processes 8 \
  --mixed_precision bf16 \
  examples/rl/grpo/llada/train_btgrpo.py \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset gsm8k \
    --output_dir .models/llada-btgrpo-run19 \
    --lora_r 64 --lora_alpha 32 --lora_dropout 0.05 \
    --bf16 True \
    --learning_rate 3e-6 \
    --max_steps 8000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_generations 8 \
    --num_iterations 1 \
    --max_completion_length 512 \
    --fork_frac 0.3 \
    --learn_fork_frac True \
    --fork_frac_min 0.02 --fork_frac_max 0.7 \
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
    --run_name btgrpo-run19-1node \
    --report_to none \
  >> "${LOG}" 2>&1
