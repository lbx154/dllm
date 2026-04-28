#!/usr/bin/env bash
# =============================================================================
# BT-GRPO run23 — recover from run22's KL gradient blow-up.
#
# run22 postmortem (aborted at step 1 by Tier-1 watcher):
#   loss = 3.25e8,  grad_norm = 2.65e10
#
#   Root cause: β=0.04 on LLaDA dLLM with apply_divergent_mask=False.
#   - dLLM per-token log-ratios reach |r| > 5 even at LoRA init (mask positions)
#   - kl_ratio_clip=5 bounds exp(r) ≤ 148 PER TOKEN
#   - max_completion_length=384 × batch=64 × β=0.04 still scales gradient enough
#     for first-step blow-up before optimizer warmup kicks in
#   - GRPO paper used autoregressive LM (no mask positions) → β=0.04 is fine there
#
#   This is a NEW failure mode the L1 oracle never saw and L4 forward proof
#   warned about under "L-smooth assumption requires bounded β·∇²KL" — but the
#   exact β threshold for dLLM is empirical, not analytic.
#
# run23 changes from run22:
#   - β: 0.04 → 0.001                 (40× smaller; still > 0 to satisfy L4 A4)
#   - apply_divergent_mask: F → T      (KL only on tokens where forks disagree,
#                                       reduces effective β by ~3× more)
#
# Effective KL strength = run22's / ~120  → safely small while preserving the
# anchor that L4 wanted in the first place.
#
# Other layers unchanged from run22:
#   L4: filter_zero_correct=False (no selection bias), w_strict=0 (T7 density)
#   L2: reward weights still corr-dominant (≈92% effective signal)
#   L1: oracle prediction same as run22 (P_starve=0.91, watcher will validate)
#
# Postmortem of run21:
#   - strict_format_reward_func/std = 0.003  (37 steps × 64 completions)
#       → w_strict=1.0 was a vacuous weight; reward function never fired
#   - frac_groups_filtered = 50% avg (peak 75%)
#       → filter_zero_correct=True is throwing away half the data
#       → violates GRPO unbiasedness assumption (L4 forward-proof gap A3.3)
#   - clip_ratio = 0 across all steps
#       → ε=0.2 inert; ρ never leaves [0.8, 1.2]; PPO clip is decorative
#   - entropy stuck at 0.06–0.16  →  β=0 → no L-smooth (L4 gap A5.1)
#
# Layer-wise diagnosis → run22 changes:
#   L4 (forward proof):
#     - β: 0.0 → 0.04   (TRPO trust region; L-smooth via KL Hessian)
#     - filter_zero_correct: True → False  (drop selection bias on prompt distrib)
#   L3 (backward theorems, NEW T7 reward signal density):
#     - w_strict: 1.0 → 0  (its observed std≈0 → wasted weight)
#   L2 (reward replay on run19 rollouts):
#     - confirmed corr/std=0.81 stays > 0.5 with new weights
#     - new effective signal mix: corr=76% (vs 88% in run21), more balanced
#   L1 (oracle):
#     - search constrained to {β>0, filter=False, w_strict=0} → 64 candidates
#     - oracle warns P_starve=0.90 (out-of-history territory) → watcher must verify
#
# This is the first config where L1 (empirical) and L4 (theoretical) DISAGREE.
# We trust L4 because the assumption violation is provable, then rely on the
# live watcher to validate that fzs stays < 0.5 in practice.
#
# Launch via:
#     bash scripts/launch_with_canary.sh scripts/launch_btgrpo_run22.sh
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
LOG="${LOG_DIR}/btgrpo-run23.log"

ACC_CFG="${REPO_ROOT}/scripts/accelerate_configs/zero2.yaml"

accelerate launch \
  --config_file "${ACC_CFG}" \
  --num_machines 1 \
  --num_processes 8 \
  --mixed_precision bf16 \
  examples/rl/grpo/llada/train_btgrpo.py \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset gsm8k \
    --output_dir .models/llada-btgrpo-run23 \
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
    --beta 0.001 \
    --epsilon 0.2 \
    --apply_divergent_mask True \
    --reward_weights 0.5 0.5 0.0 0.5 2.5 \
    --scale_rewards False \
    --filter_zero_std_groups True \
    --filter_zero_correct_groups False \
    --log_rollouts_every 25 \
    --sync_ref_model False \
    --logging_steps 1 --save_steps 256 --save_total_limit 3 \
    --seed 46 \
    --run_name btgrpo-run23-1node \
    --report_to none \
  >> "${LOG}" 2>&1
