"""
GRPO training for LLaDA with **branching** diffusion denoising.

Identical to `train.py` except the sampler is swapped from
`MDLMSampler` to `BranchingMDLMSampler`, which lets each fork-group of
`num_generations` siblings share a common denoising prefix for
`fork_frac` of the total steps before diverging.

Run (mirrors the vanilla 8-GPU LoRA + DeepSpeed ZeRO-2 recipe):
    accelerate launch --config_file scripts/accelerate_configs/zero2.yaml \
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
        --fork_frac 0.5 \
        --output_dir .models/LLaDA-8B-Instruct/grpo-bt
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Optional

from peft import LoraConfig
from trl import ModelConfig, TrlParser

import dllm
from dllm.core.samplers.mdlm_branching import (
    BranchingMDLMSampler,
    BranchingMDLMSamplerConfig,
)
from dllm.pipelines.rl import get_dataset_and_rewards
from dllm.pipelines.rl.btgrpo import BTGRPOConfig, BTGRPOTrainer

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class TrainingArguments(BTGRPOConfig):
    output_dir: str = ".models/LLaDA-8B-Instruct/grpo-bt"
    dataset: Optional[str] = field(
        default="gsm8k",
        metadata={"help": "Dataset to train on: gsm8k, countdown, sudoku, math, code."},
    )
    verbose_reward: bool = field(
        default=False,
        metadata={"help": "Enable verbose printing in reward functions."},
    )


def train():
    parser = TrlParser((TrainingArguments, ModelConfig))
    training_args, model_config = parser.parse_args_and_config()

    if not model_config.model_name_or_path:
        model_config.model_name_or_path = "GSAI-ML/LLaDA-8B-Instruct"

    dataset, reward_functions = get_dataset_and_rewards(training_args.dataset)

    if training_args.verbose_reward:
        reward_functions = [partial(fn, verbose=True) for fn in reward_functions]

    train_set = dataset.shuffle(seed=training_args.seed)

    model_args = dllm.utils.ModelArguments(
        model_name_or_path=model_config.model_name_or_path,
        load_in_4bit=(
            model_config.load_in_4bit
            if hasattr(model_config, "load_in_4bit")
            else False
        ),
    )
    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    model.config.use_cache = False

    peft_config = None
    if model_config.lora_r and model_config.lora_r > 0:
        peft_config = LoraConfig(
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ],
            task_type="CAUSAL_LM",
            lora_dropout=model_config.lora_dropout,
        )

    # ---- The ONE change vs train.py: branching sampler ------------------------
    sampler = BranchingMDLMSampler(model=model, tokenizer=tokenizer)
    sampler_config = BranchingMDLMSamplerConfig(
        steps=training_args.steps,
        max_new_tokens=training_args.max_completion_length,
        block_size=training_args.block_size,
        temperature=training_args.temperature or 0.0,
        cfg_scale=training_args.cfg_scale,
        remasking=training_args.remasking,
        num_branches=training_args.num_generations,
        fork_frac=training_args.fork_frac,
        per_block_fork=training_args.per_block_fork,
    )

    logger.info(
        "Start BT-GRPO training (fork_frac=%s, num_branches=%s, apply_divergent_mask=%s)...",
        training_args.fork_frac,
        training_args.num_generations,
        training_args.apply_divergent_mask,
    )
    trainer = BTGRPOTrainer(
        model=model,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_set,
        processing_class=tokenizer,
        peft_config=peft_config,
        sampler=sampler,
        sampler_config=sampler_config,
    )

    if training_args.save_steps % training_args.num_iterations != 0:
        import warnings

        warnings.warn(
            f"save_steps ({training_args.save_steps}) is not divisible by "
            f"num_iterations ({training_args.num_iterations}). If resuming from a checkpoint, "
            f"you may need to manually pick a checkpoint where the step is divisible by "
            f"{training_args.num_iterations}."
        )

    trainer.train()


if __name__ == "__main__":
    train()
