# Learned Fork Fraction for BT-GRPO

This document explains the design of `ForkHead` and the iterations it went
through during development. It should be read alongside
[`fork_head.py`](./fork_head.py) and the relevant logic in
[`trainer.py`](./trainer.py) (`_generate_and_score_completions`,
`_pool_prompt_hidden`, `_sample_fork_frac`, `_reinforce_fork_head`).

## 1. Background: what is `fork_frac`?

BT-GRPO (Branching Tree GRPO) extends standard GRPO by performing each
diffusion rollout in two phases:

1. **Shared phase** — one prefix rollout for all `G` siblings in a fork
   group (cheap; every member gets the same partial denoising).
2. **Divergent phase** — `G` independent rollouts from that prefix,
   producing `G` different completions used for the GRPO group-relative
   advantage.

The boundary between the two phases is controlled by `fork_frac ∈ (0, 1)`:

```
fork_step_idx = int(fork_frac * total_denoise_steps)
```

- `fork_frac → 0` ≈ standard GRPO (fully independent rollouts).
- `fork_frac → 1` ≈ all siblings share almost everything (very low
  intra-group reward variance, potentially no learning signal).

Picking a good `fork_frac` is a trade-off between **rollout cost**
(higher fork_frac = more shared compute) and **policy exploration**
(lower fork_frac = more divergent behaviour to learn from).

Historically `fork_frac` was a constant hyper-parameter (default `0.5`).
The goal of `ForkHead` is to learn it **per prompt**, online, alongside
the main GRPO loop.

## 2. Goal

Train a tiny policy head that maps the prompt's hidden state to a
`fork_frac` value, such that it maximises the GRPO reward signal we are
already computing anyway — no extra rollouts, no extra reward model.

Constraints:

- **Must not destabilise main training.** The head lives in its own
  optimiser, its own loss, its own gradient synchronisation.
- **Must recover the original fixed-`0.5` behaviour as its initial
  state**, so we can ship it default-on without regressing existing
  runs.
- **Must fit in the existing per-step budget.** No extra rollouts, one
  extra prompt-only forward pass at most.

## 3. Architecture

```
h_pooled  (B × H, mean-pooled prompt hidden state, H = 4096)
   │
   ▼
LayerNorm(H)              # scales / centres h to O(1)
   │
   ▼
Linear(H → 8)  = bottleneck
   │
   ├──► Linear(8 → 1) → clamp(lo, hi)      = μ(h)   (policy mean)
   │
   ├──► Linear(8 → 1)                      = V(h)   (value baseline)
   │
   └──► log_sigma (scalar, clamped [0.05, 0.3])     (exploration σ)

Action  a ~ Normal(μ(h), σ) ,  fork_frac = clamp(a, lo+ε, hi-ε)
```

**Total parameters:** ~`H * 8 + 8*1 + 8*1 + H*2 + 8 + 2` ≈ **~41k** for
`H = 4096`, bottleneck = 8. Negligible vs the 8 B main model.

**Initialisation** (matters a lot — see §5.2):

- `proj.weight = 0`, `proj.bias = (lo + hi)/2` → initial μ exactly
  equals the old fixed default (0.5 for `[lo, hi] = [0.2, 0.8]`).
- `value_head.weight = 0`, `value_head.bias = 0` → V(h) starts at 0 and
  regresses toward observed reward over time.
- `bottleneck.weight ~ N(0, 0.02²)` → random but small; gives gentle
  per-prompt variation once `proj.weight` starts moving.
- `log_sigma = -1.6` → σ ≈ 0.2 initial exploration.

## 4. Training algorithm (per GRPO step)

At every call to `_generate_and_score_completions(inputs)`:

1. **Sample**: pool the prompt-only hidden state → `h ∈ ℝᴴ`. Run
   `head.sample(h)` to obtain `(a, log π(a|h), μ(h), V(h))`. Overwrite
   `sampler_config.fork_frac = a`.
2. **Rollout + score** via `super()._generate_and_score_completions(…)`
   — standard BT-GRPO, using the fork_frac we just chose.
3. **Read back** the batch's mean reward `r` from the parent's metrics,
   all-reduce across ranks.
4. **Actor–critic update**:
   - `advantage = r − V(h).detach()`
   - `policy_loss = −log π(a|h) · advantage`
   - `value_loss = (V(h) − r)²`
   - `loss = policy_loss + value_loss`
   - All-reduce gradients across ranks, then `fork_optim.step()`.

Important details:

- **Action must be detached** when computing `log_prob`. Using
  `rsample()` here is a subtle bug (see §5.1).
- **Head is kept in fp32** even when the main model is bf16. Adam
  moments would otherwise underflow the tiny REINFORCE updates.
- **Separate optimiser** (`torch.optim.Adam`, lr `1e-3`). It does not
  touch LoRA / base-model parameters.
- **Cross-rank sync**: both the reward scalar and the head gradients
  are manually all-reduced. The head params stay identically
  replicated across ranks without any DDP wrapping.

## 5. Why the design looks the way it does — a bug diary

Each subsection corresponds to a real bug we hit; they explain every
non-obvious design choice.

### 5.1 Bug: `rsample()` cancelled the policy gradient to zero

**Symptom:** `μ(h)` stayed **exactly** at 0.5 for many training steps.
`log_sigma` moved fine, `grad_norm` on the head was > 0, Adam was
clearly stepping — but `proj.weight` and `proj.bias` gradients were
**identically zero**.

**Cause:** the original code did

```python
raw = dist.rsample()          # reparameterised: raw = μ + σ·ε
log_prob = dist.log_prob(raw)
```

For a Gaussian, `∂ log p / ∂ μ = (raw − μ)/σ²`. With the reparameterised
`raw = μ + σ·ε`, backprop through `log_prob` produces **two** paths to
μ:

```
direct:    ∂ log p / ∂ μ                   = +(raw − μ)/σ²
indirect:  ∂ log p / ∂ raw · ∂ raw / ∂ μ   = −(raw − μ)/σ² · 1
           = −(raw − μ)/σ²
sum       =                                   0
```

They cancel exactly. REINFORCE ≠ reparameterised sampling — the action
must be treated as a **constant** w.r.t. policy parameters.

**Fix:** `raw = dist.sample().detach()`; `log_prob = dist.log_prob(raw)`.

Confirmed with a 30-step synthetic test: bias stayed at 0.5 with
`rsample()`, cleanly moved with detached `sample()`.

### 5.2 Bug: sigmoid-parameterised mean was 7× undergeared

**First attempt** parameterisation was

```
μ = sigmoid(logit) · (hi − lo) + lo
```

Derivative at `logit = 0` is `σ'(0) · (hi − lo) = 0.25 · 0.6 = 0.15`.
So every 0.01 of movement in `proj.bias` only produces 0.0015 of
movement in μ — it would take hundreds of steps to see the head leave
0.500 on the 3-decimal log display.

**Fix:** direct linear parameterisation with a clamp,

```
μ = clamp(proj(h), lo, hi)
```

Gradient is now 1:1. Combined with the next bug fix (§5.3), this is
fast enough to visibly learn in 20–30 steps.

### 5.3 Bug: 4096-dim weight blew past the clamp in one step

After §5.1 and §5.2 were fixed, μ now moved — but **saturated at 0.8
immediately** on step 3 and stayed pinned there.

**Cause:** with the head a single `Linear(4096 → 1)`, the update to
`proj.weight` after one REINFORCE step is `Δw_i = lr · sign(grad_wᵢ)`
per coordinate (Adam's first-step behaviour). The gradient direction
is perfectly correlated with the current prompt's hidden state, so

```
Δμ_next  ≈  Δw · h_next  ≈  lr · √H · E[|h|]  =  1e-3 · √4096 · 0.5  ≈  0.3
```

— enough to shoot past the clamp in a single step. Lowering `lr` helps
but does not solve it: the problem is the **dimensionality** of the
weight, not the learning rate.

**Fix:** `LayerNorm → Linear(H → 8) → Linear(8 → 1)`. The low-rank
bottleneck caps the effective dimension the gradient can correlate
against. Max per-step swing becomes `O(lr · 8) ≈ 0.008` — two orders of
magnitude smaller. Verified on synthetic task: μ now drifts smoothly
from 0.500 over 50 steps toward the true optimum.

LayerNorm is also important: without it, `h` from LLaDA-8B has scale
that varies widely across prompts, and the "max single-step swing" bound
above would not actually hold.

### 5.4 Bias: constant EMA baseline leaked prompt difficulty into the policy

**Problem:** with `advantage = reward − EMA(reward)`, easy prompts
always have positive advantage and hard prompts always have negative
advantage, **independent of the fork_frac we chose**. REINFORCE then
pushes `log π(· | easy_prompt_features)` up and
`log π(· | hard_prompt_features)` down. The head ends up learning to
identify prompt difficulty rather than learning the actual
fork_frac → reward relationship.

**Fix:** per-prompt learned value baseline `V(h)`, trained jointly by
MSE on the realised reward (§4). `advantage = r − V(h).detach()` strips
the prompt-difficulty component: V(h) absorbs E[reward | prompt], and
what's left is the genuine signal about whether **this** fork_frac
choice was above or below expectation for **this** prompt.

This is plain actor–critic with the prompt embedding as state, the only
subtlety being that we share the 8-d bottleneck feature between the
policy and value heads.

### 5.5 Bug: `clamp` on the policy mean severs the gradient

**Symptom (run11, 243 steps):** with actor-critic in place, `μ(h)` still
drifted upward to 0.800 by step ~10 and then stayed **exactly** at the
upper clamp boundary for the remaining 230+ steps. `log_sigma` also
barely moved. `V(h)` tracked reward correctly, `fork_advantage`
oscillated with mean ~+0.19 and swings of ±1.9, but μ didn't budge.

**Cause:** the old implementation did

```python
raw_mean = self.proj(z).squeeze(-1)
mean     = raw_mean.clamp(self.lo, self.hi)     # ← the culprit
dist     = Normal(mean, sigma)
```

Once `raw_mean > hi`, the clamp produces `mean = hi` (a constant w.r.t.
`raw_mean`), so `∂mean/∂raw_mean = 0` and the gradient from
`log π(a|h)` cannot flow back to `proj.weight` / `proj.bias`. The actor
can **never** pull μ back inside `[lo, hi]` once it exits, regardless of
what advantages the value head is computing.

Why run10 and earlier didn't catch this: with the old EMA baseline,
advantages stayed systematically positive as long as reward was positive,
so there was no mechanism pushing μ *down* and the clamp-gradient issue
wasn't observable. Actor-critic fixed the sign of the advantage (run11
has plenty of negative advantages), which then exposed the lack of a
return path.

**Fix:** never clamp the distribution parameter; only clamp the final
sampled action:

```python
raw_mean = self.proj(z).squeeze(-1)             # unbounded
dist     = Normal(raw_mean, sigma)
raw      = dist.sample().detach()
action   = raw.clamp(self.lo + 1e-3, self.hi - 1e-3)
```

`log_prob` stays differentiable w.r.t. `raw_mean` everywhere, so even if
μ drifts past `hi` temporarily, a negative-advantage batch will reduce
`raw_mean` on the next step.

To keep `raw_mean` from drifting to infinity (and `log_prob` to useful
scale), optionally add a small L2 on `raw_mean` to the policy loss. In
practice the REINFORCE signal itself is strong enough to regularise it
as long as advantages average near zero, which actor-critic ensures.

This bug will be fixed in whatever run re-enables the fork head (run12
keeps it disabled).

## 6. Logged metrics

During training the following series are written to the usual TRL
metrics dict (and picked up by `dashboard.py`):

| Key                      | Meaning                                         |
| ------------------------ | ----------------------------------------------- |
| `btgrpo/fork_frac`       | sampled action `a` actually used by the sampler |
| `btgrpo/fork_frac_mean`  | deterministic μ(h) — what the head would pick   |
| `btgrpo/fork_sigma`      | current exploration σ                           |
| `btgrpo/fork_value`      | V(h) prediction for this step                   |
| `btgrpo/fork_baseline`   | EMA(reward) — **diagnostic only**, not used     |
| `btgrpo/fork_advantage`  | `r − V(h)` used for the REINFORCE update        |
| `btgrpo/fork_loss`       | combined policy + value loss                    |
| `btgrpo/fork_value_loss` | MSE component only                              |

The **Fork Frac (head)** and **Fork Head REINFORCE** dashboard panels
plot these directly.

## 7. How to enable / disable

Default is **off** (legacy static fork_frac). To enable, pass to the
launcher:

```
--learn_fork_frac True      \
--fork_head_lr     1e-3     \
--fork_frac_min    0.2      \
--fork_frac_max    0.8
```

To disable mid-run, just drop `--learn_fork_frac True` — the sampler
will fall back to `--fork_frac` (default 0.5).
