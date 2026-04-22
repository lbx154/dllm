"""
BranchingMDLMSampler — masked-diffusion sampler with a shared-prefix fork point.

For each unique prompt the sampler runs MDLM denoising in two phases:

    Phase 1 (shared):   x_T  →  x_{k*}      once per prompt
    Phase 2 (divergent): x_{k*} → x_0        G independent copies per prompt

The input list is expected to be [p_0, p_0, …, p_0,  p_1, p_1, …, p_1,  …]  —
i.e. each prompt repeated G consecutive times, which is exactly the layout
produced by TRL's GRPOTrainer when num_generations == G.  We detect the grouping
from `num_branches` (= G).

Only Phase 2 introduces stochasticity across branches — Phase 1 is a single
deterministic (seed-controlled) denoising.  Positions committed during Phase 1
are therefore identical across all G branches of the same prompt; positions
committed during Phase 2 may diverge.

The implementation mirrors `MDLMSampler.sample` but factors the inner step
loop into a helper that can be invoked twice with a hand-off of state `(x,
attention_mask, unmasked_index, …)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSamplerOutput
from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens


@dataclass
class BranchingMDLMSamplerConfig(MDLMSamplerConfig):
    num_branches: int = 8
    fork_frac: float = 0.5  # fraction of total steps spent in the shared phase


@dataclass
class BranchingMDLMSampler(MDLMSampler):
    """MDLM sampler with a single shared-prefix fork point.

    Contract:
      `inputs` has length B * G where G == config.num_branches and every G
      consecutive prompts are identical.  Returns B * G completions in the same
      order.  Phase-1 state is computed B times and then tiled to B * G before
      Phase 2.
    """

    # --------------------------------------------------------------------- #
    # Step-loop helper (shared by both phases)
    # --------------------------------------------------------------------- #

    def _run_blocks(
        self,
        *,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        unmasked_index: torch.Tensor,
        prompt_lens: list[int],
        max_new_tokens: int,
        block_size: int,
        steps_per_block: int,
        block_range: tuple[int, int],
        transfer_fracs_per_block: tuple[float, float],
        temperature: float,
        cfg_scale: float,
        remasking: str,
        stochastic_transfer: bool,
        suppress_tokens: Optional[list[int]],
        begin_suppress_tokens: Optional[list[int]],
        right_shift_logits: bool,
        mask_id: int,
        histories: Optional[list] = None,
    ) -> torch.Tensor:
        """Run MDLM denoising for the requested slice of each block.

        `block_range = (b_start, b_end)` selects which blocks to process.
        `transfer_fracs_per_block = (frac_first, frac_last)` controls, for the
        first and last selected block, which contiguous segment of steps is run
        (start fraction, end fraction).  All blocks in between run fully.

        In practice we invoke this twice per full sample:
          Phase 1:  blocks = [0, b_fork_idx],   fracs on last block = (0, frac)
          Phase 2:  blocks = [b_fork_idx, end], fracs on first block = (frac, 1)
        """
        B = x.size(0)
        num_blocks = math.ceil(max_new_tokens / block_size)
        assert steps_per_block >= 1

        b_start, b_end = block_range
        b_end = min(b_end, num_blocks)

        for b in range(b_start, b_end):
            # Per-sample block_mask_index aligned to each prompt's tail
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=x.device
            )
            for j in range(B):
                start = prompt_lens[j] + b * block_size
                end = min(start + block_size, prompt_lens[j] + max_new_tokens, x.size(1))
                if start < end:
                    width = end - start
                    block_mask_index[j, :width] = x[j, start:end] == mask_id

            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )
            effective_steps = num_transfer_tokens.size(1)

            # Which step-range inside this block to actually run
            if b == b_start and b == b_end - 1:
                s_lo = int(transfer_fracs_per_block[0] * effective_steps)
                s_hi = int(transfer_fracs_per_block[1] * effective_steps)
            elif b == b_start:
                s_lo = int(transfer_fracs_per_block[0] * effective_steps)
                s_hi = effective_steps
            elif b == b_end - 1:
                s_lo = 0
                s_hi = int(transfer_fracs_per_block[1] * effective_steps)
            else:
                s_lo = 0
                s_hi = effective_steps
            s_lo = max(0, min(s_lo, effective_steps))
            s_hi = max(s_lo, min(s_hi, effective_steps))

            for i in range(s_lo, s_hi):
                mask_index = x == mask_id

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask.repeat(2, 1)
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x, attention_mask=attention_mask).logits

                if suppress_tokens:
                    for tok in suppress_tokens:
                        logits[:, :, tok] = -torch.inf
                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if begin_suppress_tokens:
                    for tok in begin_suppress_tokens:
                        logits[:, :, tok] = -torch.inf

                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                for j in range(B):
                    x0_p[j, prompt_lens[j] + (b + 1) * block_size :] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -np.inf))

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    k = int(num_transfer_tokens[j, i].item())
                    if k > 0:
                        _, select_index = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_index] = True

                x[transfer_index] = x0[transfer_index]
                if histories is not None:
                    histories.append(x.clone())

        return x

    # --------------------------------------------------------------------- #
    # Public entry
    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def sample(
        self,
        inputs,
        config: Optional[BranchingMDLMSamplerConfig] = None,
        **kwargs,
    ):
        if config is None:
            config = BranchingMDLMSamplerConfig()

        num_branches = kwargs.get("num_branches", config.num_branches)
        fork_frac = kwargs.get("fork_frac", config.fork_frac)

        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        return_dict = kwargs.get("return_dict", config.return_dict)

        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        assert num_branches >= 1
        assert 0.0 < fork_frac < 1.0

        # --- Normalise inputs ----------------------------------------------------
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]
        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        total = len(inputs)
        assert total % num_branches == 0, (
            f"inputs length {total} must be divisible by num_branches={num_branches}; "
            "BTGRPOTrainer is expected to feed G consecutive copies of each prompt."
        )
        B_unique = total // num_branches

        # Sanity: verify the G consecutive copies really are identical
        for b in range(B_unique):
            g0 = inputs[b * num_branches]
            for g in range(1, num_branches):
                gi = inputs[b * num_branches + g]
                if not torch.equal(g0, gi):
                    raise ValueError(
                        "BranchingMDLMSampler expects each prompt to be repeated "
                        "num_branches consecutive times; got a mismatch."
                    )

        unique_prompts = [inputs[b * num_branches] for b in range(B_unique)]
        prompt_lens_u = [p.shape[0] for p in unique_prompts]
        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens_u)
        else:
            max_new_tokens = max_length - max(prompt_lens_u)
        T = max_length

        # --- Build Phase-1 canvas (B_unique rows) --------------------------------
        device = self.model.device
        x1 = torch.full((B_unique, T), eos_id, dtype=torch.long, device=device)
        attn1 = torch.zeros((B_unique, T), dtype=torch.long, device=device)
        for i, p in enumerate(unique_prompts):
            pl = prompt_lens_u[i]
            x1[i, :pl] = p
            x1[i, pl : pl + max_new_tokens] = mask_id
            attn1[i, : min(pl + max_new_tokens, T)] = 1

        unmasked_index1 = (x1 != mask_id) & attn1.bool()
        if cfg_keep_tokens:
            keep_mask = torch.isin(
                x1, torch.as_tensor(cfg_keep_tokens, device=device)
            )
            unmasked_index1 = unmasked_index1 & ~keep_mask

        num_blocks = math.ceil(max_new_tokens / block_size)
        steps_per_block = math.ceil(steps / num_blocks)

        # --- Phase 1: shared denoising (B_unique copies) -------------------------
        # We express fork_frac in terms of total-block-step-index:
        #   total_steps_effective = num_blocks * steps_per_block
        #   fork_step_idx = floor(fork_frac * total_steps_effective)
        #   fork_block_idx = fork_step_idx // steps_per_block
        #   fork_frac_in_block = (fork_step_idx % steps_per_block) / steps_per_block
        total_eff = num_blocks * steps_per_block
        fork_step_idx = int(fork_frac * total_eff)
        fork_block_idx = fork_step_idx // steps_per_block
        fork_frac_in_block = (fork_step_idx % steps_per_block) / steps_per_block

        histories = [x1.clone()] if return_dict else None
        x1 = self._run_blocks(
            x=x1,
            attention_mask=attn1,
            unmasked_index=unmasked_index1,
            prompt_lens=prompt_lens_u,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            steps_per_block=steps_per_block,
            block_range=(0, fork_block_idx + 1),
            transfer_fracs_per_block=(0.0, fork_frac_in_block if fork_block_idx < num_blocks else 1.0),
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            stochastic_transfer=stochastic_transfer,
            suppress_tokens=suppress_tokens,
            begin_suppress_tokens=begin_suppress_tokens,
            right_shift_logits=right_shift_logits,
            mask_id=mask_id,
            histories=histories,
        )

        # --- Tile x_{fork} across G branches -------------------------------------
        x2 = (
            x1.unsqueeze(1)
            .expand(B_unique, num_branches, T)
            .contiguous()
            .view(B_unique * num_branches, T)
            .clone()
        )
        attn2 = (
            attn1.unsqueeze(1)
            .expand(B_unique, num_branches, T)
            .contiguous()
            .view(B_unique * num_branches, T)
            .clone()
        )
        unmasked_index2 = (
            unmasked_index1.unsqueeze(1)
            .expand(B_unique, num_branches, T)
            .contiguous()
            .view(B_unique * num_branches, T)
            .clone()
        )
        prompt_lens2 = [prompt_lens_u[b] for b in range(B_unique) for _ in range(num_branches)]

        # --- Phase 2: divergent denoising (B_unique * num_branches copies) -------
        # Resume from the fork block; run the remainder of that block plus all
        # downstream blocks.  Different branches now diverge because Gumbel
        # sampling (temperature > 0) and topk-tie-breaking are stochastic.
        x2 = self._run_blocks(
            x=x2,
            attention_mask=attn2,
            unmasked_index=unmasked_index2,
            prompt_lens=prompt_lens2,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            steps_per_block=steps_per_block,
            block_range=(fork_block_idx, num_blocks),
            transfer_fracs_per_block=(fork_frac_in_block, 1.0),
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            stochastic_transfer=stochastic_transfer,
            suppress_tokens=suppress_tokens,
            begin_suppress_tokens=begin_suppress_tokens,
            right_shift_logits=right_shift_logits,
            mask_id=mask_id,
            histories=histories,
        )

        if return_dict:
            return BaseSamplerOutput(sequences=x2, histories=histories)
        return x2
