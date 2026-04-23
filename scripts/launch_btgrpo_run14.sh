#!/usr/bin/env bash
# =============================================================================
# BT-GRPO run14 — hypothesis-driven fork head + num_iter 2->1 fallback
#
# User's hypothesis (the important one):
#   fork_frac SHOULD correlate INVERSELY with difficulty.
#     - HARD prompt  -> fork EARLY  -> low  fork_frac: branches need to
#                       explore different reasoning paths from the start.
#     - EASY prompt  -> fork LATE   -> high fork_frac: most tokens are
#                       formatting / brackets / math-formulae boilerplate
#                       and should be shared across branches.
#   Confirmed by `mdlm_branching.py:42`:
#     "fork_frac = fraction of total steps spent in the shared phase"
#
# Diagnosis of run13 that drove these changes:
#
#   (I)  Fork head drifted μ to 1.507 (out of [0,1]) with sampled action
#        saturating at 0.999 from batch 7 onwards — converged on
#        "fork LATE for everything", which is exactly wrong for hard
#        prompts (which dominate GSM8K and give negative advantage).
#        Cause: we removed the clamp (fixing §5.5) but did not replace
#        it with a bounded, differentiable parameterisation.
#
#   (II) num_iter=2 was not enough to avoid the asymmetric PPO-clip
#        blow-up in the main policy: batches 7/11/12/17 of run13 had
#        pre-clip loss 441–9010 and grad_norm 3900–81000. PPO's
#        clip(r·A, (1-ε)A ... (1+ε)A) only bounds A>0 branches;
#        A<0 with runaway ratio still produces large gradients.
#
# run14 changes:
#
#   (1) fork_head.py: sigmoid parameterisation of the mean
#         raw   = proj(z) + value_coupling * V(z).detach()
#         mean  = lo + (hi-lo) * sigmoid(raw)
#       - bounded by construction (no clamp = no gradient severance)
#       - gradient shrinks at boundaries but never zero (graceful)
#       - V(z) as a feature structurally encodes the user's hypothesis:
#         high V (easy) -> high raw -> high mean (fork LATE);
#         low V (hard)  -> low raw  -> low  mean (fork EARLY).
#         value_coupling is a scalar learnable parameter, init=+1.
#         At init (V=0, proj=0), mean=sigmoid(0)=0.5 — exactly matches
#         the legacy fixed fork_frac=0.5 default.
#
#   (2) --num_iterations 2 -> 1
#       Bypass PPO ratio drift entirely. Run5-11 already showed num_iter=1
#       is stable on dLLMs; the only reason we tried num_iter>1 was to
#       increase gradient density, but on B200 we now have 2x the batch
#       (G=8, per_device_bs=8) which already doubles gradient density
#       at the data level.
#
# Kept from run13 (validated):
#   max_completion_length=512 (truncation gone),
#   block_size=64 (halved sequential depth),
#   G=8, per_device_bs=8 (group baseline meaningful),
#   reward_weights strict_format=0 (dead weight zeroed),
#   xmlcount no tail penalty (fixed in format.py),
#   learn_fork_frac=True, fork_frac_min=0.0, fork_frac_max=1.0,
#   fork_head_lr=1e-3, beta=0, epsilon=0.2, lora_r=64, lr=3e-6.
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
LOG="${LOG_DIR}/btgrpo-run14.log"

ACC_CFG="${REPO_ROOT}/scripts/accelerate_configs/zero2.yaml"

accelerate launch \
  --config_file "${ACC_CFG}" \
  --num_machines 1 \
  --num_processes 8 \
  --mixed_precision bf16 \
  examples/rl/grpo/llada/train_btgrpo.py \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --dataset gsm8k \
    --output_dir .models/llada-btgrpo-run14 \
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
    --run_name btgrpo-run14-1node \
    --report_to none \
  >> "${LOG}" 2>&1
