"""BT-GRPO trainer — minimal version (no ForkHead, no KL-estimator changes).

Subclasses DiffuGRPOTrainer and overrides _generate_and_score_completions
to:
  1. multiply completion_mask by a per-position divergent_mask (=0 where all
     G fork-siblings emitted the same token);
  2. rescale advantages by 1/f_D, where f_D = mean(divergent_mask) globally
     across ranks, so that the expected policy-gradient magnitude matches
     vanilla GRPO.

No other behaviour is changed — sampler is expected to be a
BranchingMDLMSampler with num_branches == args.num_generations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from trl.extras.profiling import profiling_decorator

from dllm.pipelines.rl.grpo.trainer import DiffuGRPOConfig, DiffuGRPOTrainer


@dataclass
class BTGRPOConfig(DiffuGRPOConfig):
    """BT-GRPO config = DiffuGRPOConfig + fork knobs."""

    fork_frac: float = 0.5
    apply_divergent_mask: bool = True
    div_frac_floor: float = 0.1  # cap adv_scale at 1 / floor (=10x by default)


class BTGRPOTrainer(DiffuGRPOTrainer):
    """DiffuGRPOTrainer + divergent_mask + 1/f_D advantage rescaling."""

    @profiling_decorator
    def _generate_and_score_completions(self, inputs):
        batch = super()._generate_and_score_completions(inputs)

        if not getattr(self.args, "apply_divergent_mask", True):
            return batch

        G = self.args.num_generations
        completion_ids = batch["completion_ids"]
        N, L = completion_ids.shape
        if N % G != 0:
            return batch
        B_unique = N // G

        grouped = completion_ids.view(B_unique, G, L)
        first = grouped[:, 0:1, :]
        divergent = (grouped != first).any(dim=1)  # [B_unique, L]
        divergent_flat = (
            divergent.unsqueeze(1)
            .expand(B_unique, G, L)
            .contiguous()
            .view(N, L)
            .to(batch["completion_mask"].dtype)
        )
        batch["completion_mask"] = batch["completion_mask"] * divergent_flat

        # Mean divergent fraction across the full distributed batch (so every
        # rank uses the same scale and ZeRO doesn't smear LRs across ranks).
        device = batch["completion_mask"].device
        div_frac_local = divergent.float().mean().detach().to(device)
        div_frac_global = self.accelerator.reduce(div_frac_local, reduction="mean")
        div_frac = float(div_frac_global.item())
        floor = float(self.args.div_frac_floor)
        adv_scale = 1.0 / max(div_frac, floor)

        if "advantages" in batch and isinstance(batch["advantages"], torch.Tensor):
            batch["advantages"] = batch["advantages"] * adv_scale

        mode = "train" if self.model.training else "eval"
        self._metrics[mode].setdefault("btgrpo/divergent_frac", []).append(div_frac)
        self._metrics[mode].setdefault("btgrpo/adv_scale", []).append(adv_scale)
        return batch
