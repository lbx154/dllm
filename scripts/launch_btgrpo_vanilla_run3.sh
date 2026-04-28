#!/usr/bin/env bash
# =============================================================================
# Vanilla GRPO run3 — EXACT docstring recipe from examples/rl/grpo/llada/train.py
#
# Verbatim 8-GPU ZeRO-2 + LoRA r=128 command from the train.py "Local users"
# block, with two unavoidable upstream-TRL deviations marked below.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_DIR="$(pwd)/.logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/btgrpo-vanilla_run3.log"

accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    --num_machines 1 \
    --num_processes 8 \
    --mixed_precision bf16 \
    examples/rl/grpo/llada/train.py \
    --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
    --load_in_4bit True --lora_r 128 --lora_alpha 64 --lora_dropout 0.05 \
    --dataset gsm8k \
    --max_steps 15000 --learning_rate 3e-6 \
    --num_generations 6 --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 2 --num_iterations 12 \
    --block_size 32 --steps 128 \
    --p_mask_prompt 0.15 --beta 0.04 --epsilon 0.5 \
    --sync_ref_model False \
    --output_dir .models/LLaDA-8B-Instruct/grpo-run3 \
    --logging_steps 1 --save_steps 256 --save_total_limit 3 \
    --seed 49 \
    --run_name grpo-vanilla-run3-1node \
    --report_to none \
    --bf16 True \
  >> "${LOG}" 2>&1
# ----------------------------------------------------------------------------
# DEVIATIONS FROM DOCSTRING (forced by upstream changes since recipe was written):
#   - sync_ref_model: True → False
#       New TRL raises NotImplementedError when sync_ref_model=True with PEFT
#       (PEFT path doesn't keep a separate ref model; it disables the adapter
#       to recover the base. Sync is therefore impossible.)
#   - ref_model_sync_steps: dropped (only meaningful when sync_ref_model=True)
# Everything else (β=0.04, ε=0.5, num_iterations=12, block=32, steps=128,
# 4-bit base + LoRA r=128, lr=3e-6, num_gen=6, bs=6, accum=2, p_mask=0.15)
# is the docstring's verbatim 8-GPU ZeRO-2 recipe.
# ============================================================================
