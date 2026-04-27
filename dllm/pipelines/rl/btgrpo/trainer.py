"""BT-GRPO trainer — minimal version (no ForkHead, no KL-estimator changes).

Only differences from DiffuGRPOTrainer:

1. ``apply_divergent_mask`` (default True) — zero out completion_mask at
   token positions that were *finalized during the shared (pre-fork)
   phase* of the BranchingMDLMSampler.  These positions are by
   construction identical across all G fork siblings (the sampler tiles
   x_fork to G branches and Phase-2 inherits that frozen state), so the
   per-token gradient on every sibling is identical and group-normalized
   advantage cancels them to 0 in expectation — they are pure noise in
   the policy gradient and only dilute the mean-normalized loss.

   The shared-phase mask comes from the sampler itself
   (BranchingMDLMSampler._shared_phase_masks_chunks).  This is cleaner
   than the earlier "all G siblings agree" heuristic, which also masked
   *post-fork* positions where branches happened to coincide
   organically — those positions carry real learning signal and should
   not be removed.

2. ``apply_adv_scale`` (default False) — optional 1 / max(active_frac,
   floor) advantage rescaling.  Originally added as a "compensation for
   fewer training tokens", but TRL's GRPO loss is already
   mean-normalized over the mask, so the scaling does NOT compensate;
   it only inflates effective LR on active positions and pushes policy
   out of the trust region.  Kept behind a flag for ablations only.

3. Optional rollout dump (rank-0, jsonl, every N optimizer steps) for
   inspection.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from trl.extras.profiling import profiling_decorator

from dllm.pipelines.rl.grpo.trainer import DiffuGRPOConfig, DiffuGRPOTrainer


@dataclass
class BTGRPOConfig(DiffuGRPOConfig):
    """BT-GRPO config = DiffuGRPOConfig + fork knobs."""

    fork_frac: float = 0.5
    per_block_fork: bool = True
    apply_divergent_mask: bool = True
    apply_adv_scale: bool = False
    div_frac_floor: float = 0.1
    rollout_dump_path: str = ""
    rollout_dump_every: int = 16  # every N optimizer steps


class BTGRPOTrainer(DiffuGRPOTrainer):
    """DiffuGRPOTrainer + divergent_mask (+ optional 1/f_D adv scale + rollout dump)."""

    @profiling_decorator
    def _generate_and_score_completions(self, inputs):
        # Reset the per-step shared-phase mask buffer on the sampler so that
        # (super) sample() loop only writes this batch's masks.
        self.sampler._shared_phase_masks_chunks = []

        batch = super()._generate_and_score_completions(inputs)

        completion_ids = batch["completion_ids"]
        N, L = completion_ids.shape

        active_frac = None
        chunks = getattr(self.sampler, "_shared_phase_masks_chunks", []) or []
        if (
            getattr(self.args, "apply_divergent_mask", True)
            and chunks
        ):
            shared_full = torch.cat(chunks, dim=0)  # [N_local, T_full]
            # T_full = prompt_len + completion_len.  Slice the completion tail.
            shared_completion = shared_full[:, -L:].to(batch["completion_mask"].device)
            keep = (~shared_completion.bool()).to(batch["completion_mask"].dtype)
            batch["completion_mask"] = batch["completion_mask"] * keep

            # Fraction of completion-area tokens that were NOT filled in the
            # shared phase, i.e. the fraction that contributes to the loss.
            active_local = keep.float().mean().detach().to(batch["completion_mask"].device)
            active_global = self.accelerator.reduce(active_local, reduction="mean")
            active_frac = float(active_global.item())

            mode = "train" if self.model.training else "eval"
            self._metrics[mode].setdefault("btgrpo/active_frac", []).append(active_frac)
            self._metrics[mode].setdefault(
                "btgrpo/shared_filled_frac", []
            ).append(1.0 - active_frac)

            if getattr(self.args, "apply_adv_scale", False):
                floor = float(self.args.div_frac_floor)
                adv_scale = 1.0 / max(active_frac, floor)
                if "advantages" in batch and isinstance(batch["advantages"], torch.Tensor):
                    batch["advantages"] = batch["advantages"] * adv_scale
                self._metrics[mode].setdefault("btgrpo/adv_scale", []).append(adv_scale)

        # Drop the cached masks so they don't accumulate across iterations.
        self.sampler._shared_phase_masks_chunks = []

        self._maybe_dump_rollouts(batch, active_frac)
        return batch

    # ------------------------------------------------------------------
    # rollout dumping
    # ------------------------------------------------------------------
    def _maybe_dump_rollouts(self, batch, div_frac):
        path = getattr(self.args, "rollout_dump_path", "") or ""
        if not path:
            return
        if not self.accelerator.is_main_process:
            return
        every = max(1, int(getattr(self.args, "rollout_dump_every", 16)))
        step = int(self.state.global_step) if hasattr(self, "state") else 0
        if step % every != 0:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tok = self.processing_class
            prompt_ids = batch.get("prompt_ids")
            completion_ids = batch.get("completion_ids")
            advantages = batch.get("advantages")
            comp_mask = batch.get("completion_mask")
            if completion_ids is None:
                return
            prompts_text = (
                tok.batch_decode(prompt_ids, skip_special_tokens=True)
                if prompt_ids is not None else [""] * completion_ids.size(0)
            )
            completions_text = tok.batch_decode(completion_ids, skip_special_tokens=False)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": time.time(),
                    "step": step,
                    "div_frac": div_frac,
                    "n_rows": int(completion_ids.size(0)),
                    "G": int(self.args.num_generations),
                }) + "\n")
                for i in range(completion_ids.size(0)):
                    rec = {
                        "step": step,
                        "i": i,
                        "prompt": prompts_text[i],
                        "completion": completions_text[i],
                    }
                    if isinstance(advantages, torch.Tensor):
                        rec["advantage"] = float(advantages[i].item()) if advantages.dim() == 1 else None
                    if isinstance(comp_mask, torch.Tensor):
                        rec["mask_sum"] = int(comp_mask[i].sum().item())
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[btgrpo] rollout dump failed: {exc}")
