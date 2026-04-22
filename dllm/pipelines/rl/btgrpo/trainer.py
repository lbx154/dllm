"""
BT-GRPO — Branching Trajectory GRPO for masked diffusion language models.

Extends DiffuGRPOTrainer:
  1. Swaps MDLMSampler for BranchingMDLMSampler (shared-prefix fork-group).
  2. Multiplies completion_mask by a `divergent_mask` so the PPO/KL update
     only applies to positions where the G fork-siblings disagree.

All other GRPO machinery (clip, β-KL, ref-sync, logging) is inherited
unchanged from trl.GRPOTrainer -> DiffuGRPOTrainer.

See docs/BT_GRPO.md for the theoretical justification.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
from accelerate.utils import gather_object
from datasets import Dataset, IterableDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_decorator
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_trainer import nanstd
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from transformers.utils import is_peft_available

from dllm.core.samplers.mdlm_branching import (
    BranchingMDLMSampler,
    BranchingMDLMSamplerConfig,
)
from dllm.pipelines.rl.grpo.trainer import DiffuGRPOConfig, DiffuGRPOTrainer

if is_peft_available():
    from peft import PeftConfig

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


@dataclass
class BTGRPOConfig(DiffuGRPOConfig):
    """BT-GRPO config.  Extends DiffuGRPOConfig with the fork knob."""

    fork_frac: float = 0.5
    # If True (default), zero out gradient contribution from positions where
    # all G fork-siblings share the same token.  Theoretically a no-op because
    # such positions contribute 0 to the group-relative advantage, but makes
    # the mechanism explicit (and a tiny bit faster).
    apply_divergent_mask: bool = True


class BTGRPOTrainer(DiffuGRPOTrainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs,
        args: Optional[BTGRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes=None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Build a BranchingMDLMSampler configured from the training args.
        # Passed up through DiffuGRPOTrainer.__init__, which accepts a
        # sampler/sampler_config kwarg and will use them verbatim.
        # We have to wait for super().__init__ to run before we can look at
        # self.model / self.processing_class, so we build a placeholder config
        # first and inject the sampler afterwards.
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        # Replace the sampler initialised by DiffuGRPOTrainer.__init__ with a
        # branching one.  num_branches is taken from args.num_generations so
        # that the TRL-duplicated prompt layout matches the fork group size.
        self.sampler = BranchingMDLMSampler(
            model=self.model, tokenizer=self.processing_class
        )
        self.sampler_config = BranchingMDLMSamplerConfig(
            steps=self.args.steps,
            max_new_tokens=self.args.max_completion_length,
            block_size=self.args.block_size,
            temperature=self.args.temperature or 0.0,
            cfg_scale=self.args.cfg_scale,
            remasking=self.args.remasking,
            num_branches=self.args.num_generations,
            fork_frac=self.args.fork_frac,
        )

    # ------------------------------------------------------------------ #
    # Override _generate_and_score_completions to inject the divergent mask
    # ------------------------------------------------------------------ #

    @profiling_decorator
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Mostly duplicates DiffuGRPOTrainer._generate_and_score_completions but
        multiplies the completion_mask by the per-group divergent_mask before
        returning the batch dict (so TRL's loss computation naturally zeros out
        gradient on positions shared across the fork group).
        """
        # --- Reuse parent's generation/scoring by calling super ----------
        batch = super()._generate_and_score_completions(inputs)

        if not self.args.apply_divergent_mask:
            return batch

        # --- Compute divergent_mask: positions where fork siblings disagree ---
        # batch["completion_ids"] is [N, L] with N == B_unique * G in this-process
        # order: [p0b0, p0b1, …, p0b(G-1), p1b0, …].
        G = self.args.num_generations
        completion_ids = batch["completion_ids"]
        N, L = completion_ids.shape
        if N % G != 0:
            # Unexpected — bail out without masking
            return batch
        B_unique = N // G
        grouped = completion_ids.view(B_unique, G, L)
        # A position is divergent iff not all G branches share the same token
        first = grouped[:, 0:1, :]  # [B_unique, 1, L]
        divergent = (grouped != first).any(dim=1)  # [B_unique, L]
        divergent_flat = (
            divergent.unsqueeze(1)
            .expand(B_unique, G, L)
            .contiguous()
            .view(N, L)
            .to(batch["completion_mask"].dtype)
        )
        batch["completion_mask"] = batch["completion_mask"] * divergent_flat

        # Log the divergent fraction for diagnostics
        mode = "train" if self.model.training else "eval"
        self._metrics[mode].setdefault("btgrpo/divergent_frac", []).append(
            float(divergent.float().mean().item())
        )
        return batch
