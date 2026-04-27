"""BT-GRPO trainer — minimal version (no ForkHead, no KL-estimator changes).

Only differences from DiffuGRPOTrainer:

1. ``divergent_mask`` — completion_mask is zero'd on positions where every
   sibling in a fork-group emitted the same token (those positions
   contribute identical log-probs to every branch and yield a 0 advantage
   contribution).  TRL's GRPO loss is mean-normalized over the mask, so
   no extra advantage rescaling is needed; the per-divergent-token
   gradient magnitude already matches vanilla per-token gradient magnitude.

2. (optional, OFF by default) ``adv_scale = 1 / max(f_D, floor)``.  This
   was originally added to "compensate for fewer training tokens", but
   under mean-normalized loss it is *not* a compensation — it just
   inflates effective LR on divergent positions by 1/f_D and pushes
   policy out of the trust region.  Kept behind a flag for ablations.

3. Optional rollout dump (rank-0 only, jsonl) so we can inspect what the
   model is actually generating when reward collapses.
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
    apply_divergent_mask: bool = True
    apply_adv_scale: bool = False
    div_frac_floor: float = 0.1
    rollout_dump_path: str = ""
    rollout_dump_every: int = 16  # every N optimizer steps


class BTGRPOTrainer(DiffuGRPOTrainer):
    """DiffuGRPOTrainer + divergent_mask (+ optional 1/f_D adv scale + rollout dump)."""

    @profiling_decorator
    def _generate_and_score_completions(self, inputs):
        batch = super()._generate_and_score_completions(inputs)

        G = self.args.num_generations
        completion_ids = batch["completion_ids"]
        N, L = completion_ids.shape

        div_frac = None
        if getattr(self.args, "apply_divergent_mask", True) and N % G == 0:
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

            device = batch["completion_mask"].device
            div_frac_local = divergent.float().mean().detach().to(device)
            div_frac_global = self.accelerator.reduce(div_frac_local, reduction="mean")
            div_frac = float(div_frac_global.item())

            mode = "train" if self.model.training else "eval"
            self._metrics[mode].setdefault("btgrpo/divergent_frac", []).append(div_frac)

            if getattr(self.args, "apply_adv_scale", False):
                floor = float(self.args.div_frac_floor)
                adv_scale = 1.0 / max(div_frac, floor)
                if "advantages" in batch and isinstance(batch["advantages"], torch.Tensor):
                    batch["advantages"] = batch["advantages"] * adv_scale
                self._metrics[mode].setdefault("btgrpo/adv_scale", []).append(adv_scale)

        self._maybe_dump_rollouts(batch, div_frac)
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
