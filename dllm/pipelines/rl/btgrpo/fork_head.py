"""Learned fork-fraction head for BT-GRPO.

Maps a pooled prompt hidden state to a per-prompt fork_frac in
[fork_frac_min, fork_frac_max] via a Gaussian policy, trained end-to-end
with REINFORCE using the same per-batch reward signal that GRPO produces.

Run training script (in launcher):
    --learn_fork_frac True --fork_head_lr 1e-3 \
    --fork_frac_min 0.2 --fork_frac_max 0.8

The head is tiny (hidden_size + 2 params) and lives outside the main TRL
optimizer.  Per-rank gradient sync is done manually before each step.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ForkHead(nn.Module):
    """Linear projection of pooled prompt hidden state -> Gaussian over fork_frac."""

    def __init__(self, hidden_size: int, lo: float = 0.2, hi: float = 0.8,
                 bottleneck: int = 8):
        super().__init__()
        if not (0.0 < lo < hi < 1.0):
            raise ValueError(f"need 0 < lo < hi < 1, got lo={lo}, hi={hi}")
        self.lo, self.hi = lo, hi
        # LayerNorm + low-rank bottleneck keeps the projection update bounded:
        # without this, a single REINFORCE step on a 4096-d weight can swing
        # the next-prompt projection by O(lr * sqrt(H) * |h|) ~ several units,
        # immediately saturating the [lo, hi] clamp. With bottleneck=8 and
        # post-LN |h_norm|~1, max single-step swing is O(lr * 8) ~ tiny.
        self.norm = nn.LayerNorm(hidden_size)
        self.bottleneck = nn.Linear(hidden_size, bottleneck)
        self.proj = nn.Linear(bottleneck, 1)
        # Value head: predicts E[reward | prompt], acts as a per-prompt baseline
        # for REINFORCE. Without it, advantage = reward - global_EMA is biased
        # by prompt difficulty (easy prompts get high reward regardless of
        # fork_frac → head drifts toward "what I picked on easy prompts" rather
        # than "best fork_frac for this prompt"). See FORK_HEAD.md for details.
        self.value_head = nn.Linear(bottleneck, 1)
        # Init: zero the projection weights so initial mean is determined
        # solely by proj.bias = midpoint (matches old fixed default 0.5).
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.5 * (lo + hi))
        # Value head starts at 0; MSE on reward will pull it toward E[reward].
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        # Bottleneck weights small Gaussian — gives mild per-prompt variation
        # once proj.weight starts moving, but stays bounded.
        nn.init.normal_(self.bottleneck.weight, std=0.02)
        nn.init.zeros_(self.bottleneck.bias)
        # Fixed-ish exploration sigma in raw (pre-clip) Gaussian space.
        # log_sigma is learnable but clamped at use time.
        self.log_sigma = nn.Parameter(torch.tensor(-1.6))  # sigma ≈ 0.2

    def _features(self, h: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(self.norm(h))

    def _mean_sigma(self, z: torch.Tensor):
        raw_mean = self.proj(z).squeeze(-1)
        mean = raw_mean.clamp(self.lo, self.hi)
        sigma = self.log_sigma.exp().clamp(0.05, 0.3)
        return mean, sigma

    @torch.no_grad()
    def predict_mean(self, h: torch.Tensor) -> float:
        z = self._features(h)
        m, _ = self._mean_sigma(z)
        return float(m.mean().item())

    def sample(self, h: torch.Tensor):
        """Sample one fork_frac from N(mean(h), sigma), clipped to [lo, hi].

        Returns a tuple:
            action_value (float, detached):  clipped fork_frac to use in sampler
            log_prob (Tensor, requires_grad): log N(raw; mean, sigma) for REINFORCE
            mean_value (float, detached):    unclipped Gaussian mean (for logging)
            value (Tensor, requires_grad):   V(prompt) baseline, scalar, for
                                              advantage estimation + MSE training
        """
        z = self._features(h.detach())
        mean, sigma = self._mean_sigma(z)
        # If h was a batch of prompt embeddings, average to a single (mean, sigma).
        if mean.ndim > 0:
            mean = mean.mean()
            # sigma is a scalar parameter; no need to average
        # Value baseline: average across the (possibly batched) features.
        value = self.value_head(z).squeeze(-1)
        if value.ndim > 0:
            value = value.mean()
        dist = torch.distributions.Normal(mean, sigma)
        # REINFORCE: action is treated as a constant w.r.t. policy params.
        # Use sample() (no reparameterization) and explicitly detach to be safe.
        # Using rsample() here would cancel the gradient w.r.t. mean to zero
        # (direct path through log_prob and indirect path through raw=mean+σε
        # have equal magnitude and opposite signs).
        raw = dist.sample().detach()
        log_prob = dist.log_prob(raw)
        action = raw.clamp(self.lo + 1e-3, self.hi - 1e-3)
        return float(action.item()), log_prob, float(mean.detach().item()), value
