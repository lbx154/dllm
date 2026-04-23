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
from dllm.pipelines.rl.btgrpo.fork_head import ForkHead
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
    # Group-std normalization for advantages. BT-GRPO defaults to True (classic
    # GRPO formula) because binary correctness rewards at G=4 produce many
    # zero-std groups whose advantage magnitude otherwise differs wildly across
    # batches. Normalizing makes the PPO clip behave scale-invariantly.
    scale_rewards: bool = True
    # KL estimator: "k3" (TRL default, exp(r)-1-r — unstable on dLLM where
    # log-ratio magnitudes can reach 50+) or "k2" (0.5 * r^2, bounded and
    # strictly non-negative). "k2_clipped" additionally clamps r to [-5, 5].
    kl_estimator: str = "k2_clipped"

    # ---- Learned per-prompt fork_frac (REINFORCE) -------------------------
    # When True, a tiny ForkHead module decides fork_frac from the (mean-pooled)
    # last-layer prompt hidden state. Trained end-to-end with REINFORCE using
    # the per-call mean reward and an EMA baseline. When False, uses fixed
    # `fork_frac` above (legacy behaviour).
    learn_fork_frac: bool = False
    fork_head_lr: float = 1e-3
    fork_frac_min: float = 0.2
    fork_frac_max: float = 0.8
    # EMA decay for the REINFORCE baseline (b_t = decay*b_{t-1} + (1-decay)*r_t).
    fork_baseline_decay: float = 0.9


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

        # ---- Optional learned fork_frac head ----------------------------
        self.fork_head = None
        self.fork_optim = None
        self._fork_baseline = None  # EMA baseline for REINFORCE
        if getattr(self.args, "learn_fork_frac", False):
            hidden_size = int(self.model.config.hidden_size)
            self.fork_head = ForkHead(
                hidden_size,
                lo=self.args.fork_frac_min,
                hi=self.args.fork_frac_max,
            ).to(self.accelerator.device)
            # Keep ForkHead in fp32 even when the main model is bf16: this
            # head is tiny (~hidden_size + 2 params) and Adam moments need
            # fp32 to accumulate the small REINFORCE updates without underflow.
            # We cast the input pooled hidden state to fp32 at use time.
            self.fork_head = self.fork_head.to(torch.float32)
            self.fork_optim = torch.optim.Adam(
                self.fork_head.parameters(), lr=self.args.fork_head_lr
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
        # --- Optional: let ForkHead pick fork_frac from prompt ----------
        # Done BEFORE super() so the updated sampler_config is used during
        # the parent's diffusion sampling pass.
        pending_fork_log_prob = None  # set if we sampled an action this call
        pending_fork_action = None
        pending_fork_value = None
        if self.fork_head is not None:
            try:
                (
                    fork_action,
                    fork_log_prob,
                    fork_mean,
                    fork_value,
                ) = self._sample_fork_frac(inputs)
                self.sampler_config.fork_frac = fork_action
                pending_fork_log_prob = fork_log_prob
                pending_fork_action = fork_action
                pending_fork_value = fork_value
                mode_pre = "train" if self.model.training else "eval"
                self._metrics[mode_pre].setdefault("btgrpo/fork_frac", []).append(
                    fork_action
                )
                self._metrics[mode_pre].setdefault(
                    "btgrpo/fork_frac_mean", []
                ).append(fork_mean)
                self._metrics[mode_pre].setdefault(
                    "btgrpo/fork_value", []
                ).append(float(fork_value.detach().item()))
            except Exception as exc:  # pragma: no cover - never break training
                import warnings
                warnings.warn(f"ForkHead sample failed, using fixed fork_frac: {exc}")

        # --- Reuse parent's generation/scoring by calling super ----------
        batch = super()._generate_and_score_completions(inputs)

        # --- REINFORCE update on ForkHead (does not touch main optimizer) ---
        if pending_fork_log_prob is not None:
            try:
                self._reinforce_fork_head(
                    pending_fork_log_prob,
                    pending_fork_action,
                    pending_fork_value,
                )
            except Exception as exc:  # pragma: no cover - keep training alive
                import warnings
                warnings.warn(f"ForkHead REINFORCE step failed: {exc}")

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

        # ---- BT-GRPO advantage rescaling: 1/f_D correction ----
        # Standard GRPO applies A_g over all T completion tokens; BT-GRPO only
        # applies it over divergent tokens (fraction f_D). To keep the expected
        # policy-gradient magnitude matched to vanilla GRPO we multiply the
        # advantage by 1/f_D (clipped to avoid blow-ups when f_D -> 0).
        # Compute div_frac as a global average across ranks so that every
        # rank applies the same adv_scale (otherwise per-rank scales differ
        # and the effective LR drifts across the batch in ZeRO).
        div_frac_local = divergent.float().mean().detach()
        div_frac_global = self.accelerator.reduce(
            div_frac_local.to(batch["completion_mask"].device), reduction="mean"
        )
        div_frac = float(div_frac_global.item())
        adv_scale = 1.0 / max(div_frac, 0.1)  # cap at 10x
        if "advantages" in batch and isinstance(batch["advantages"], torch.Tensor):
            batch["advantages"] = batch["advantages"] * adv_scale

        # Log the divergent fraction for diagnostics
        mode = "train" if self.model.training else "eval"
        self._metrics[mode].setdefault("btgrpo/divergent_frac", []).append(
            div_frac
        )
        self._metrics[mode].setdefault("btgrpo/adv_scale", []).append(adv_scale)
        return batch

    # ------------------------------------------------------------------ #
    # Learned fork_frac helpers (REINFORCE)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _pool_prompt_hidden(self, inputs) -> torch.Tensor:
        """Mean-pool the last-layer hidden state over UNIQUE prompt tokens.

        We tokenize the unique prompts (every G-th element of `inputs` since
        TRL pre-duplicates each prompt num_generations times), forward through
        the policy model with output_hidden_states=True, and average over
        non-pad positions. Returns a [hidden_size] vector (averaged over the
        local batch of unique prompts).
        """
        from trl.data_utils import maybe_apply_chat_template

        G = max(1, self.args.num_generations)
        unique_inputs = inputs[::G] if len(inputs) >= G else inputs
        prompts_text = [
            maybe_apply_chat_template(ex, self.processing_class)["prompt"]
            for ex in unique_inputs
        ]
        toks = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        device = self.accelerator.device
        input_ids = toks["input_ids"].to(device)
        attn = toks["attention_mask"].to(device)
        # Cap to max_prompt_length to match the sampler's prompt window.
        mpl = getattr(self, "max_prompt_length", None)
        if mpl is not None:
            input_ids = input_ids[:, -mpl:]
            attn = attn[:, -mpl:]

        was_training = self.model.training
        self.model.eval()
        try:
            out = self.model(
                input_ids=input_ids,
                attention_mask=attn,
                output_hidden_states=True,
                use_cache=False,
            )
        finally:
            if was_training:
                self.model.train()

        h = out.hidden_states[-1]  # [B, T, H]
        mask = attn.unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)  # [B, H]
        return pooled.mean(dim=0)  # [H]

    def _sample_fork_frac(self, inputs):
        """Sample one fork_frac for the whole local batch via ForkHead."""
        pooled = self._pool_prompt_hidden(inputs)
        # Cast to head dtype to keep matmul happy.
        pooled = pooled.to(next(self.fork_head.parameters()).dtype)
        return self.fork_head.sample(pooled)

    def _reinforce_fork_head(
        self,
        log_prob: torch.Tensor,
        action: float,
        value: torch.Tensor,
    ) -> None:
        """Apply one actor-critic update on the ForkHead.

        Uses V(prompt) as a per-prompt baseline (removes prompt-difficulty
        bias from the REINFORCE signal) plus an MSE regression loss on V.
        A global EMA baseline is also tracked for logging / diagnostics.
        """
        mode = "train" if self.model.training else "eval"
        rewards_log = self._metrics.get(mode, {}).get("reward", [])
        if not rewards_log:
            return
        local_reward = float(rewards_log[-1])
        # All-reduce the reward across ranks so every rank uses the same scalar
        # for its REINFORCE update (head params are replicated; gradients are
        # also all-reduced below).
        device = self.accelerator.device
        r_t = torch.tensor(local_reward, device=device, dtype=torch.float32)
        r_global = self.accelerator.reduce(r_t, reduction="mean")
        reward = float(r_global.item())

        # --- Per-prompt value baseline (cancels prompt-difficulty bias) ---
        value_f = value.to(torch.float32)
        baseline_pp = float(value_f.detach().item())
        advantage = reward - baseline_pp
        adv_t = torch.tensor(
            advantage, device=log_prob.device, dtype=log_prob.dtype
        )
        reward_t = torch.tensor(
            reward, device=value_f.device, dtype=torch.float32
        )
        policy_loss = -log_prob * adv_t
        value_loss = (value_f - reward_t) ** 2
        loss = policy_loss + value_loss

        self.fork_optim.zero_grad(set_to_none=True)
        loss.backward()
        # All-reduce gradients across ranks so replicated head stays in sync.
        if self.accelerator.num_processes > 1:
            for p in self.fork_head.parameters():
                if p.grad is not None:
                    self.accelerator.reduce(p.grad, reduction="mean")
        self.fork_optim.step()

        # Still track the EMA baseline for diagnostics (not used for advantage).
        decay = float(self.args.fork_baseline_decay)
        if self._fork_baseline is None:
            self._fork_baseline = reward
        self._fork_baseline = decay * self._fork_baseline + (1.0 - decay) * reward

        self._metrics[mode].setdefault("btgrpo/fork_baseline", []).append(
            self._fork_baseline
        )
        self._metrics[mode].setdefault("btgrpo/fork_advantage", []).append(
            advantage
        )
        self._metrics[mode].setdefault("btgrpo/fork_loss", []).append(
            float(loss.detach().item())
        )
        self._metrics[mode].setdefault("btgrpo/fork_value_loss", []).append(
            float(value_loss.detach().item())
        )
        with torch.no_grad():
            sigma_val = float(self.fork_head.log_sigma.exp().clamp(0.05, 0.3).item())
        self._metrics[mode].setdefault("btgrpo/fork_sigma", []).append(sigma_val)
