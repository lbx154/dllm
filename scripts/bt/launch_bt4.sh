#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=disabled
exec accelerate launch --config_file scripts/accelerate_configs/zero2.yaml \
  examples/rl/grpo/llada/train_bt.py \
  --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
  --load_in_4bit True --lora_r 128 --lora_alpha 64 --lora_dropout 0.05 \
  --dataset gsm8k \
  --max_steps 1000 --learning_rate 3e-6 \
  --num_generations 6 --per_device_train_batch_size 6 \
  --gradient_accumulation_steps 2 --num_iterations 12 \
  --block_size 32 --steps 128 \
  --p_mask_prompt 0.15 --beta 0.04 --epsilon 0.5 \
  --sync_ref_model False \
  --fork_frac 0.25 \
  --per_block_fork True \
  --apply_divergent_mask True \
  --apply_adv_scale False \
  --rollout_dump_path .logs/rollouts-bt4-run1.jsonl \
  --rollout_dump_every 24 \
  --logging_steps 10 \
  --save_steps 1000 \
  --output_dir .models/LLaDA-8B-Instruct/grpo-bt4
