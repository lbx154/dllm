"""
GRPO (Group Relative Policy Optimization) trainer for diffusion language models.

References:
  diffu-grpo: https://github.com/dllm-reasoning/d1/tree/main/diffu-grpo
  GRPO: https://arxiv.org/abs/2402.03300
"""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
from accelerate.utils import gather_object, set_seed
from datasets import Dataset, IterableDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from trl import GRPOConfig
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_decorator
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_trainer import GRPOTrainer, nanstd, split_tensor_dict
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from transformers.utils import is_peft_available

from dllm.core.samplers import (
    BaseSampler,
    BaseSamplerConfig,
    MDLMSampler,
    MDLMSamplerConfig,
)

if is_peft_available():
    from peft import PeftConfig

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


@dataclass
class DiffuGRPOConfig(GRPOConfig):
    """
    Configuration for DiffuGRPOTrainer, extending GRPOConfig with diffusion-specific parameters.
    """

    block_size: int = 64
    steps: int = 64
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"
    p_mask_prompt: float = 0.3
    # Disable advantage std-normalization by default to match the diffu-GRPO reference
    # implementation (d1), which does not divide advantages by their std.
    scale_rewards: bool = False
    # Clip |log(pi_ref / pi_policy)| per token before plugging into TRL's k3
    # KL estimator  KL ~ exp(r) - 1 - r.  Without this, dLLM's unbounded
    # per-token log-probs cause exp() to overflow and yield KL values ~1e9.
    # Set to None to disable clipping (unsafe on dLLM).
    kl_ratio_clip: Optional[float] = 5.0

    # -------- Noise suppression for sparse-correctness regimes ---------------
    # When correctness is very rare (e.g. <5% pass rate), most groups collapse
    # to a single shared reward. Their advantage is mathematically 0 but their
    # tokens still enter the PPO loss denominator (num_items_in_batch), which
    # dilutes the signal from the few informative groups. With scale_rewards=True
    # it is much worse: tiny residual format-reward variance (std~0.01) gets
    # divided out and turns into ±1 advantages — pure noise amplification.
    #
    # filter_zero_std_groups: zero BOTH advantages and completion_mask for
    #   groups whose total-reward std == 0. Strictly a loss-denominator fix.
    # filter_zero_correct_groups: same treatment for groups where the
    #   correctness reward specifically has zero std (i.e. whole group right or
    #   whole group wrong). This kills the format-only "noise gradient" in the
    #   typical-case group where nobody solves the problem. Only active if a
    #   reward func whose name contains "correct" is present.
    filter_zero_std_groups: bool = False
    filter_zero_correct_groups: bool = False
    # If >0, every N optimizer steps print one (prompt, completion, extracted,
    # answer, correctness) quadruple for sanity-checking the reward parser.
    log_rollouts_every: int = 0


class DiffuGRPOTrainer(GRPOTrainer):
    """
    GRPO Trainer adapted for masked diffusion language models.

    Overrides:
    - `_generate_and_score_completions`: replaces autoregressive generation with iterative denoising
    - `_get_per_token_logps`: replaces causal log-prob with diffusion forward-process log-prob
    - `_compute_loss`: injects per-iteration mask seeds and precomputed old/ref log-probs
    - `_prepare_inputs`: disables shuffle to keep mask seeds reproducible across iterations

    PPO clipping, KL regularization, reference model management, and distributed training
    are inherited from TRL's GRPOTrainer.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[DiffuGRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (
            None,
            None,
        ),
        peft_config: Optional["PeftConfig"] = None,
        sampler: Optional[BaseSampler] = None,
        sampler_config: Optional[BaseSamplerConfig] = None,
    ):
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
        self.sampler = sampler or MDLMSampler(
            model=self.model, tokenizer=self.processing_class
        )
        self.sampler_config = sampler_config or MDLMSamplerConfig(
            steps=self.args.steps,
            max_new_tokens=self.args.max_completion_length,
            block_size=self.args.block_size,
            temperature=self.args.temperature or 0.0,
            cfg_scale=self.args.cfg_scale,
            remasking=self.args.remasking,
        )
        # Per-iteration mask seed state (aligned with d1's approach).
        # Seeds are batch-level, stored as instance variables.
        # Per-iteration logps are stored in the batch dict to stay aligned with sequences
        # (shuffle is disabled in _prepare_inputs).
        self._mask_seeds: list[int] = []
        self._diffu_iter_idx: int = 0

    def _forward_process(self, batch, prompt_index, mask_id, seed=None):
        """
        Apply the MDLM forward process (noising).

        - Prompt tokens are masked with probability p_mask_prompt.
        - Completion tokens are always masked.
        """
        if seed is not None:
            set_seed(seed)
        is_mask_prompt = prompt_index & (
            torch.rand(batch.shape, device=batch.device) < self.args.p_mask_prompt
        )
        noised_input_ids = torch.where(is_mask_prompt | ~prompt_index, mask_id, batch)
        return noised_input_ids

    # -----------------------------------------------------------------------
    # Override: per-token log probabilities (diffusion forward process)
    # -----------------------------------------------------------------------

    def _get_per_token_logps(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        seed=None,
    ) -> torch.Tensor:
        """
        Compute per-token log probabilities under the diffusion forward process.

        Replaces TRL's causal log-prob computation with:
        1. Apply random masking (prompt: p_mask_prompt, completion: always masked)
        2. Single bidirectional forward pass through the diffusion model (with optional CFG)
        3. Cross-entropy on completion tokens only → log-prob
        """
        if seed is None:
            seed = getattr(self, "_current_mask_seed", None)
        batch_size = batch_size or input_ids.size(0)
        mask_id = self.processing_class.mask_token_id
        cfg_scale = self.sampler_config.cfg_scale
        seq_len = input_ids.size(1)
        prompt_length = seq_len - logits_to_keep

        prompt_index = torch.arange(seq_len, device=input_ids.device) < prompt_length

        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            batch = input_ids[i : i + batch_size]
            batch_mask = attention_mask[i : i + batch_size]

            noised_input_ids = self._forward_process(
                batch, prompt_index, mask_id, seed=seed
            )

            if cfg_scale > 0.0:
                prompt_index_expanded = prompt_index.unsqueeze(0).repeat(
                    noised_input_ids.shape[0], 1
                )
                un_batch = noised_input_ids.clone()
                un_batch[prompt_index_expanded] = mask_id
                logits, un_logits = torch.chunk(
                    model(
                        torch.cat([noised_input_ids, un_batch]),
                        attention_mask=batch_mask.repeat(2, 1),
                    ).logits,
                    2,
                    dim=0,
                )
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(noised_input_ids, attention_mask=batch_mask).logits

            completion_logits = logits[:, -logits_to_keep:, :]
            completion_targets = batch[:, -logits_to_keep:]
            loss = F.cross_entropy(
                completion_logits.reshape(-1, completion_logits.size(-1)),
                completion_targets.reshape(-1),
                reduction="none",
            )
            all_logps.append(
                -loss.view(batch.size(0), logits_to_keep).to(torch.float32)
            )

        per_token_logps = torch.cat(all_logps, dim=0)
        # Zero out post-EOS logps to prevent inf*0=nan in TRL's loss computation
        completion_mask = attention_mask[:, -logits_to_keep:]
        per_token_logps = per_token_logps * completion_mask
        return per_token_logps

    # -----------------------------------------------------------------------
    # Override: generation (diffusion instead of autoregressive)
    # -----------------------------------------------------------------------

    @profiling_decorator
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Override TRL's _generate_and_score_completions to use diffusion generation.

        Replaces autoregressive generation with iterative denoising. Reward computation
        uses the parent's `_calculate_rewards`; advantage normalization and logging are
        handled inline.
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        _max_prompt_length = getattr(self, "max_prompt_length", None)
        if _max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -_max_prompt_length :]
            prompt_mask = prompt_mask[:, -_max_prompt_length :]
            prompts_text = self.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        # ---- Diffusion generation via sampler ----
        generation_batch_size = self.args.generation_batch_size or prompt_ids.size(0)

        with unwrap_model_for_generation(
            self.model_wrapped,
            self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
        ) as unwrapped_model:
            with (
                FSDP.summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled
                else nullcontext()
            ):
                self.sampler.model = unwrapped_model
                prompt_completion_ids_all = []
                for i in range(0, prompt_ids.size(0), generation_batch_size):
                    batch = list(prompt_ids[i : i + generation_batch_size])
                    out = self.sampler.sample(batch, self.sampler_config)
                    prompt_completion_ids_all.append(out)
                    torch.cuda.empty_cache()

        prompt_completion_ids = torch.cat(prompt_completion_ids_all, dim=0)

        # Extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Build completion_mask (attention mask): 1 up to and including the first EOS, 0 after
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m]
            for row, mask_row in zip(completion_ids, completion_mask)
        ]
        completion_lengths = completion_mask.sum(1)

        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = (
                completion_mask * (~truncated_completions).unsqueeze(1).int()
            )

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        # Generate one mask seed per inner iteration (aligned with d1's approach).
        # old/ref/current log-probs within each iteration use the same mask seed,
        # keeping importance-weight ratios stable.
        self._mask_seeds = [
            torch.randint(0, 2**31 - 1, (1,)).item() for _ in range(self.num_iterations)
        ]
        self._diffu_iter_idx = 0
        self._current_mask_seed = self._mask_seeds[0]

        need_old_logps = (
            self.num_iterations > 1
            or self.args.steps_per_generation > self.args.gradient_accumulation_steps
        )

        with torch.no_grad():
            if need_old_logps:
                # Precompute old_per_token_logps for every inner iteration.
                # Stacked into [N, num_iterations, L] and included in the batch dict
                # to stay aligned with sequences (shuffle is disabled in _prepare_inputs).
                old_logps_list = []
                for s in self._mask_seeds:
                    logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size,
                        seed=s,
                    )
                    old_logps_list.append(logps)
                old_logps_all = torch.stack(
                    old_logps_list, dim=1
                )  # [N, num_iterations, L]
                old_per_token_logps = old_logps_list[0]
            else:
                old_logps_all = None
                old_per_token_logps = None

            if self.beta != 0.0:
                # Precompute ref_per_token_logps for every inner iteration.
                ref_logps_list = []
                for s in self._mask_seeds:
                    if self.ref_model is not None:
                        ref_logps = self._get_per_token_logps(
                            self.ref_model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size,
                            seed=s,
                        )
                    else:
                        with self.accelerator.unwrap_model(
                            self.model
                        ).disable_adapter():
                            ref_logps = self._get_per_token_logps(
                                self.model,
                                prompt_completion_ids,
                                attention_mask,
                                logits_to_keep,
                                batch_size,
                                seed=s,
                            )
                    ref_logps_list.append(ref_logps)
                ref_logps_all = torch.stack(
                    ref_logps_list, dim=1
                )  # [N, num_iterations, L]
                ref_per_token_logps = ref_logps_list[0]

                # --- KL estimator stabilization (k3 -> bounded k3) ---
                # TRL uses k3:  KL ~ exp(r) - 1 - r  with  r = logp_ref - logp_policy.
                # On dLLM, per-token |r| can reach 50+ because masked-position
                # log-probs are unbounded, making exp(r) numerically explode
                # (observed KL ~ 1e9). We clamp r into [-c, c] (default c=5) so
                # exp(r) is bounded by e^c = 148 per token. On-policy (first
                # iter) old_logps == policy_logps, so clipping r against
                # old_logps is the tightest valid approximation.
                clip = getattr(self.args, "kl_ratio_clip", None)
                if clip is not None and old_logps_all is not None:
                    ref_logps_all = old_logps_all + torch.clamp(
                        ref_logps_all - old_logps_all, -clip, clip
                    )
                    ref_per_token_logps = ref_logps_all[:, 0]
            else:
                ref_logps_all = None
                ref_per_token_logps = None

        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = (
                    prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
        else:
            completions = completions_text

        # Use TRL's reward computation helper
        rewards_per_func = self._calculate_rewards(
            inputs, prompts, completions, completion_ids_list
        )

        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(
            std_grouped_rewards, torch.zeros_like(std_grouped_rewards)
        )

        # --- zero-correctness-std mask (per group) ----------------------------
        # Identifies groups where the correctness reward func produced identical
        # values across all G siblings (typically: all wrong, occasionally: all
        # right). Updates to these groups come purely from format-reward noise.
        correctness_idx = next(
            (i for i, n in enumerate(self.reward_func_names) if "correct" in n.lower()),
            None,
        )
        if correctness_idx is not None:
            corr_per_group = rewards_per_func[:, correctness_idx].view(
                -1, self.num_generations
            )
            is_correct_std_zero = torch.isclose(
                corr_per_group.std(dim=1),
                torch.zeros_like(corr_per_group.std(dim=1)),
            )
        else:
            is_correct_std_zero = torch.zeros_like(is_std_zero)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = rewards - mean_grouped_rewards
        # NOTE: TRL normalizes args.scale_rewards to one of {"group","batch","none"},
        # mapping bool False -> "none". The string "none" is truthy in Python, so
        # `if self.scale_rewards:` silently kept dividing. Compare against "none"
        # explicitly to honour the user's intent.
        if self.scale_rewards and self.scale_rewards != "none":
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # --- Apply zero-std / zero-correct-std filters ------------------------
        # Mask BOTH advantages and completion_mask so filtered tokens contribute
        # neither numerator nor denominator to the PPO loss.
        filter_mask = torch.zeros_like(std_grouped_rewards, dtype=torch.bool)
        if getattr(self.args, "filter_zero_std_groups", False):
            filter_mask = filter_mask | is_std_zero.repeat_interleave(
                self.num_generations, dim=0
            )
        if getattr(self.args, "filter_zero_correct_groups", False) and correctness_idx is not None:
            filter_mask = filter_mask | is_correct_std_zero.repeat_interleave(
                self.num_generations, dim=0
            )
        # Note: rewards/advantages are GLOBAL (gathered across world), completion_mask
        # is LOCAL (per-rank). Slice filter_mask to the local view before masking mask.
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        if filter_mask.any():
            advantages = advantages * (~filter_mask).to(advantages.dtype)
            local_filter_mask = filter_mask[process_slice]
            completion_mask = completion_mask * (~local_filter_mask).to(
                completion_mask.dtype
            ).unsqueeze(1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        # ---- Metrics & logging ----
        acc = self.accelerator
        m = self._metrics[mode]

        def log_stats(prefix, t):
            t = t.float()
            m[f"{prefix}/mean"].append(t.mean().item())
            m[f"{prefix}/min"].append(t.min().item())
            m[f"{prefix}/max"].append(t.max().item())

        if mode == "train":
            self.state.num_input_tokens_seen += (
                acc.gather(attention_mask.sum()).sum().item()
            )
        m["num_tokens"] = [self.state.num_input_tokens_seen]

        agg_completion_lengths = acc.gather(completion_lengths)
        log_stats("completions/length", agg_completion_lengths)

        agg_terminated_with_eos = acc.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        m["completions/clipped_ratio"].append(
            1 - len(term_completion_lengths) / len(agg_completion_lengths)
        )
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        log_stats("completions/terminated_length", term_completion_lengths)

        for i, name in enumerate(self.reward_func_names):
            self._metrics[mode][f"rewards/{name}/mean"].append(
                torch.nanmean(rewards_per_func[:, i]).item()
            )
            self._metrics[mode][f"rewards/{name}/std"].append(
                nanstd(rewards_per_func[:, i]).item()
            )
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(
            is_std_zero.float().mean().item()
        )
        self._metrics[mode]["frac_correct_zero_std"].append(
            is_correct_std_zero.float().mean().item()
        )
        if filter_mask.any() or getattr(self.args, "filter_zero_std_groups", False) \
                or getattr(self.args, "filter_zero_correct_groups", False):
            self._metrics[mode]["frac_groups_filtered"].append(
                filter_mask.view(-1, self.num_generations)[:, 0].float().mean().item()
            )

        # -- Periodic rollout dump for parser/reward sanity checking ----------
        log_every = getattr(self.args, "log_rollouts_every", 0) or 0
        if (
            log_every > 0
            and mode == "train"
            and self.accelerator.is_main_process
            and (self.state.global_step % log_every == 0)
        ):
            try:
                from dllm.pipelines.rl.grpo.rewards.format import (
                    extract_xml_answer,
                    extract_answer_lenient,
                )
                ans_list = inputs[0].get("answer", None) if isinstance(inputs[0], dict) else None
                for j in range(min(2, len(completions_text))):
                    ext_strict = extract_xml_answer(completions_text[j])
                    ext_lenient = extract_answer_lenient(completions_text[j])
                    corr_val = (
                        rewards_per_func[j, correctness_idx].item()
                        if correctness_idx is not None else float("nan")
                    )
                    print(
                        f"\n[rollout step={self.state.global_step} idx={j}] "
                        f"correctness={corr_val:.3f} "
                        f"answer_gt={ans_list!r}\n"
                        f"---completion---\n{completions_text[j][:600]}\n"
                        f"---extracted lenient--- {ext_lenient!r}\n"
                        f"---extracted strict XML--- {ext_strict[:120]!r}\n",
                        flush=True,
                    )
            except Exception as e:  # pragma: no cover
                print(f"[rollout dump failed: {e}]", flush=True)

        _logs = getattr(self, "_textual_logs", None) or getattr(self, "_logs", None)
        if _logs is not None:
            _logs["prompt"].extend(gather_object(prompts_text))
            _logs["completion"].extend(gather_object(completions_text))
            for i, name in enumerate(self.reward_func_names):
                _logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
            _logs["advantages"].extend(all_process_advantages.tolist())

        # trl>=0.19 expects num_items_in_batch in the batch dict (DAPO normalizer).
        # Compute from completion_mask and aggregate across processes.
        local_num = completion_mask.sum()
        num_items_in_batch = self.accelerator.reduce(local_num.to(device), reduction="sum")
        # Safety floor: if filter_zero_*_groups masks out EVERY token in the
        # batch, num_items_in_batch becomes 0 and the PPO loss divides by zero,
        # producing NaN gradients that silently corrupt LoRA weights (after
        # which every completion saturates max_completion_length). Clamp to 1
        # so the (numerator=0)/(denominator=1)=0 case is well-defined and
        # training simply skips this step.
        num_items_in_batch = torch.clamp(num_items_in_batch, min=1)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "num_items_in_batch": num_items_in_batch,
            # Per-iteration stacked logps [N, num_iterations, L]; included in the
            # batch dict to stay aligned with sequences (shuffle is disabled in _prepare_inputs).
            "diffu_old_logps_all": old_logps_all,
            "diffu_ref_logps_all": ref_logps_all,
        }

    # -----------------------------------------------------------------------
    # Override: per-iteration seed injection in compute_loss
    # -----------------------------------------------------------------------

    def _compute_loss(self, model, inputs):
        """
        Inject the iteration-specific mask seed and precomputed old/ref log-probs
        so that old/ref/current all use the same forward-process mask per iteration.
        """
        if self._mask_seeds and (
            inputs.get("diffu_old_logps_all") is not None
            or inputs.get("diffu_ref_logps_all") is not None
        ):
            idx = self._diffu_iter_idx % self.num_iterations
            inputs = dict(inputs)  # shallow copy — don't mutate TRL's buffered dict
            if inputs.get("diffu_old_logps_all") is not None:
                inputs["old_per_token_logps"] = inputs["diffu_old_logps_all"][:, idx, :]
            if inputs.get("diffu_ref_logps_all") is not None:
                inputs["ref_per_token_logps"] = inputs["diffu_ref_logps_all"][:, idx, :]
            self._current_mask_seed = self._mask_seeds[idx]
            self._diffu_iter_idx += 1
        return super()._compute_loss(model, inputs)

    # -----------------------------------------------------------------------
    # Override: skip shuffle to keep set_seed+rand masks reproducible across old/ref/current forwards
    # -----------------------------------------------------------------------

    @profiling_decorator
    def _prepare_inputs(self, generation_batch):
        """
        Override TRL's _prepare_inputs to skip shuffle_tensor_dict.

        Diffusion masks are position-dependent (set_seed + torch.rand(batch.shape)),
        so shuffling sequences would cause old/ref masks to mismatch current-step
        masks. Disabling shuffle keeps sequence order fixed across all inner
        iterations — identical to d1's non-TRL training loop.
        """
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                generation_batch = self._generate_and_score_completions(
                    generation_batch
                )
                # Do NOT call shuffle_tensor_dict (unlike TRL default).
                self._buffered_inputs = split_tensor_dict(
                    generation_batch, self.args.steps_per_generation
                )
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs
