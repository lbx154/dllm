"""BT-GRPO training for LLaDA (diffusion denoising with branching rollouts).

Entry point analogous to examples/rl/grpo/llada/train.py but wiring in the
BranchingMDLMSampler + BTGRPOTrainer.  See docs/BT_GRPO.md for algorithm.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Optional

from peft import LoraConfig
from trl import ModelConfig, TrlParser

import dllm
from dllm.pipelines.rl import BTGRPOConfig, BTGRPOTrainer, get_dataset_and_rewards

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class TrainingArguments(BTGRPOConfig):
    output_dir: str = ".models/LLaDA-8B-Instruct/btgrpo"
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

    logger.info("Start BT-GRPO training...")
    trainer = BTGRPOTrainer(
        model=model,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_set,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    if training_args.save_steps % training_args.num_iterations != 0:
        import warnings

        warnings.warn(
            f"save_steps ({training_args.save_steps}) is not divisible by "
            f"num_iterations ({training_args.num_iterations})."
        )

    trainer.train()


if __name__ == "__main__":
    train()
