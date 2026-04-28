#!/usr/bin/env bash
# =============================================================================
# BT-GRPO run20 — first run with the Canary-RL stack on top.
#
# Config = literal clone of run19 (last hand-tuned config before canary), with
# only the seed bumped (42 -> 43) so we don't redrive the exact same trajectory.
# Any divergence from run19 is therefore attributable to the canary safety net,
# not to a config change.
#
# Use the unified launcher:
#     bash scripts/launch_with_canary.sh scripts/launch_btgrpo_run20.sh
# That wrapper runs Tier-0 preflight, starts training in the background,
# attaches canary_watcher.py (which will SIGINT this trainer on Tier-1 abort),
# and runs dashboard.py for live PNGs.
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
LOG="${LOG_DIR}/btgrpo-run20.log"

ACC_CFG="${REPO_ROOT}/scripts/accelerate_configs/zero2.yaml"

accelerate launch \
  --config_file "${ACC_CFG}" \
  --num_machines 1 \
  --num_processes 8 \
  --mixed_precision bf16 \
  examples/rl/grpo/llada/train_btgrpo.py \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset gsm8k \
    --output_dir .models/llada-btgrpo-run20 \
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
    --seed 43 \
    --run_name btgrpo-run20-1node \
    --report_to none \
  >> "${LOG}" 2>&1
