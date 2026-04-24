#!/usr/bin/env bash
# =============================================================================
# BT-GRPO run16 — fix the answer-extraction format bug (post-run15 hotfix)
#
# Diagnosis of run15:
#   run15 shipped the noise-suppression machinery (scale_rewards=False + zero-
#   std/zero-correct group filtering). The dashboard showed correctness still
#   pinned at 0 and frac_groups_filtered saturating at 0.875–1.0. Root cause
#   found in the rollout dumps:
#
#     LLaDA-8B-Instruct emits `<reasoning>...</reasoning>` followed by a free-
#     prose answer ("Bob created 91 questions.") and almost NEVER wraps it in
#     `<answer>...</answer>` tags. The GSM8K correctness parser
#     `extract_xml_answer` used `.split("<answer>")[-1]`, which returns the
#     WHOLE completion when the tag is absent. That string never equals the
#     numeric ground truth ("91"), so correctness stayed ~0 regardless of
#     whether the model's math was right.
#
# Fix (rewards/format.py, rewards/math.py):
#   • Added `extract_answer_lenient(text)` that tries, in order:
#       1. <answer>…</answer>   (requested format)
#       2. #### <n>              (GSM8K-native, LLaDA's natural output)
#       3. \boxed{<n>}           (math-style fallback)
#       4. last number in the completion (permissive)
#   • `correctness_reward_func` switched to the lenient extractor and
#     normalises both sides (strip commas, trailing ".0").
#   • `extract_xml_answer` kept strict so int_reward_func and the strict/soft
#     format rewards still reward the model for actually wrapping its answer.
#   • Rollout dump now prints both strict and lenient extractions for audit.
#
# Kept from run15:
#   scale_rewards=False, filter_zero_std_groups=True,
#   filter_zero_correct_groups=True, log_rollouts_every=100,
#   fork_frac_min=0.02, fork_frac_max=0.98, G=8, bs=8, lr=3e-6, beta=0,
#   epsilon=0.2, lora_r=64. Everything else unchanged so run15 vs run16 is
#   a clean A/B on the extraction fix.
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
LOG="${LOG_DIR}/btgrpo-run16.log"

ACC_CFG="${REPO_ROOT}/scripts/accelerate_configs/zero2.yaml"

accelerate launch \
  --config_file "${ACC_CFG}" \
  --num_machines 1 \
  --num_processes 8 \
  --mixed_precision bf16 \
  examples/rl/grpo/llada/train_btgrpo.py \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset gsm8k \
    --output_dir .models/llada-btgrpo-run16 \
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
    --run_name btgrpo-run16-1node \
    --report_to none \
  >> "${LOG}" 2>&1
