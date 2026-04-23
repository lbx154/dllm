# BT-GRPO Training Run History

Chronological log of all training restarts, the motivation for each
change, and the observed outcome. Compiled at the end of run11. All
runs are on LLaDA-8B-Instruct + LoRA (r=64 from run2 onward) on GSM8K
with 8× GPU, per-device batch 4, `num_generations G=4` (= 1 fork-group
per GPU).

The MA20 numbers below are moving-average over the final 20 steps of
each run, from log re-parsing. "Corr" is `rewards/correctness_reward_func/mean`
(0..2 range). Step counts are the number of logged TRL steps, not
denoising steps.

> run1–3 stats come from the `git log` commit messages — no logs were
> retained. run4's log is archived but was not re-read; numbers are
> from in-session notes. run5–run11 numbers come from log re-parsing.

**Convention going forward:** every training restart is preceded by a
git commit describing the change. The "Commit" column below maps each
run to the exact source-tree state that produced it.

---

## Summary table

| Run     | Commit    | Δ from previous                                                                           | Steps | MA20 loss | MA20 reward | MA20 corr | MA20 grad | fork μ | Outcome                       |
| ------- | --------- | ----------------------------------------------------------------------------------------- | ----: | --------: | ----------: | --------: | --------: | -----: | ----------------------------- |
| run1    | `85576bc` | initial BT-GRPO launch: `num_iter=4, β=0.02, lr=3e-6, eps=0.5, fork=0.5, lora_r=128`       |   n/a |       n/a |         n/a |  **0.47** |       n/a |   0.5  | early corr OK, but grad blowups |
| run2    | `14c1b63` | `num_iter 4→1, β 0.02→0.04, eps 0.5→0.3, lr 3e-6→1.5e-6, fork 0.5→0.35, lora_r 128→64, reward corr×5` |   n/a |       n/a |         n/a |       n/a |       n/a | 0.35   | **KL explosion (~1e9)**       |
| run3    | `0be8638` | code fixes: `kl_ratio_clip=5.0`, 1/f_D advantage, `scale_rewards=True`                      |   n/a |       n/a |         n/a |  **0.18** |       n/a | 0.35   | KL still 1e8, β·KL dominates; corr collapses 0.47→0.18 |
| run4    | `bae7590` | `β 0.04→0`, `fork 0.35→0.5`, `eps 0.3→0.2`, `sync_ref_model=False`                           |  1500+|        ~0 |      ~1.05  |    ~0.20  |      low  |   0.5  | **flat, not learning**        |
| run5.a1 | `bb83e7f`     | `num_iter=2, β=0.02, lr=1e-5, scale_rewards=F`                                        |     7 |    **7e8** |     0.88    |    0.17   |   **5e9** |   0.5  | **KL blew up — killed**       |
| run5.a2 | `bb83e7f`     | `num_iter=1, β=0, lr=3e-6, scale_rewards=T, kl_ratio_clip=2.0`                        |    10 |    **1e7** |     1.17    |    0.23   |   **8e11** |  0.5  | still exploding — killed      |
| run5    | `bb83e7f`     | same as a2 but `apply_divergent_mask=False`, fix per-rank `adv_scale`                  |   159 |   -0.003   |     1.19    |    0.23   |    0.62   |   0.5  | **stable; correctness flat**  |
| run6    | `bb83e7f`     | +learned fork_frac (sigmoid policy, lr=1e-3), `max_completion_length=266`              |    34 |   -0.003   |     0.85    |    0.16   |    0.43   | **0.500** | fork head not moving         |
| run7    | `bb83e7f`     | +fp32 ForkHead, direct linear param, lr=1e-2                                          |    17 |    0.019   |     1.29    |    0.25   |    0.59   | **0.500** | fork head **still** 0.5      |
| run8    | `bb83e7f`     | REINFORCE bug fix: `rsample` → `sample().detach()`                                    |    11 |    0.022   |     1.19    |    0.23   |    0.45   |  0.745 | μ moves, but **saturates 0.8** in 1 step |
| run9    | `bb83e7f`     | lr back to 1e-3                                                                       |    19 |    0.010   |     1.31    |    0.25   |    0.50   |  0.768 | still saturates by step 3     |
| run10   | `bb83e7f`     | +LayerNorm + bottleneck 4096→8 ForkHead                                               |   122 |    0.002   |     0.69    |    0.13   |    0.44   |  0.800 | μ drifts smoothly then saturates; **reward drifts down** |
| run11   | `bb83e7f`     | +value-head baseline `V(h)` (actor–critic)                                            |   243 |    0.002   |     0.90    |    0.22   |    0.45   |  **0.800** | μ 瞬间饱和 clamp、base 学得极慢 |
| run12   | `514731a`     | 深度诊断重启：`max_comp 266→512`, `num_iter 1→4`, `block 32→64`, `G/bs 4→8/8`, strict_format 权重 0，xmlcount 去负奖励，fork head **关** |    14 | 尖峰波动 | ~0.6 | **~0.12** (σ=0.22)| 尖峰 1k-35k | 0.5 关 | xmlcount 正值 (+0.34)、截断 98%→<5%；但 num_iter=4 致 ratio 漂移、grad 尖峰；14 batch 太短、早停上 run13 |
| run13   | `8e0e861`     | 重开学习型 fork head，范围 [0,1]、init 0.5、修 clamp-bug；num_iter 4→2                         |   跑中 | | | | | | |

> Going forward every new run **must** be preceded by a git commit so
> the run ↔ code state mapping is recoverable from `git log`. run5–run11
> don't have this unfortunately (all done in-session before the practice
> was established).

---

## run1 — `85576bc` — initial BT-GRPO launch

**Config:** `num_iterations=4, beta=0.02, epsilon=0.5, lr=3e-6,
fork_frac=0.5, steps=128, max_completion_length=256, lora_r=128,
reward_weights=uniform`.

**Motivation:** vanilla BT-GRPO first run as introduced in the
original commit. Multi-iteration PPO loop (`num_iterations=4`), moderate
KL, relatively loose PPO clip, large LoRA.

**Outcome:** early-training correctness reached ~0.47 (best of any run
to date), but exhibited gradient blow-ups that motivated run2.
Exact step count and end-of-run stats are not preserved (no log file
retained).

---

## run2 — `14c1b63` — tune for on-policy + correctness-weighted

**Changes:** `num_iterations 4→1` (pure on-policy; 4× faster/step),
`steps 128→64`, `max_completion_length 256→200`, `beta 0.02→0.04`
(stronger KL), `epsilon 0.5→0.3` (tighter PPO clip), `lr 3e-6→1.5e-6`
(safer), `lora_r 128→64` (smaller effective step), `fork_frac 0.5→0.35`
(earlier fork, more divergent tokens), `ref_model_mixup_alpha 1.0→0.3`
(slower reference EMA), `reward_weights = [0.25, 0.25, 0.25, 0.25, 5.0]`
(correctness-dominant).

**Motivation (per commit msg):** match BT-GRPO theory (per-step credit
works best on-policy); speed up generation; stronger regularisation to
prevent reward hacking.

**Outcome:** **KL estimator exploded to ~1e9**. The TRL `k3` estimator
(`exp(r) − 1 − r`) overflowed on dLLM masked positions where `|r|` can
reach 50+.

---

## run3 — `0be8638` — math-driven theory fixes

**Code changes (no launcher change):**

1. Clamp `(log π_ref − log π_old)` to `[-5, 5]` at precompute time
   (`kl_ratio_clip=5.0`), bounding KL by `e⁵ − 6 ≈ 142` per token.
2. Multiply the BT-GRPO advantage by `1/max(f_D, 0.1)` (capped 10×) to
   correct for the fact that BT-GRPO sums gradient over only `f_D · T`
   divergent tokens while vanilla GRPO sums over `T`.
3. `scale_rewards=True` by default in `BTGRPOConfig` (per-group-std
   advantage normalisation; parent GRPO default was False).

**Motivation (per commit msg):** three post-mortem fixes for run2's
blow-up, driven by the math of the actual objectives.

**Outcome:** KL still produced ~1e8 per token (the `kl_ratio_clip`
gate was only active when `num_iterations > 1` because `old_logps` are
never computed otherwise). Even where bounded, **`β·KL` dominated the
reward gradient ~1000:1** — the model only minimised KL and couldn't
learn the reward signal. Starting correctness **crashed from 0.47
(run1) to 0.18** — the earlier fork (0.35) also broke chain-of-thought
coherence.

---

## run4 — `bae7590` — kill KL, restore fork, tighten clip

**Changes:** `beta 0.04→0` (classic DeepSeek-GRPO: no KL term),
`fork_frac 0.35→0.5` (restore CoT coherence), `epsilon 0.3→0.2`
(PPO clip now the sole trust region), `sync_ref_model=False` (ref no
longer used).

**Motivation (per commit msg):** two lessons from run3 — (a) k3 KL on
dLLMs is fundamentally unstable and the kl_ratio_clip gate doesn't fire
with num_iter=1; (b) fork_frac=0.35 breaks CoT. Roll back both.

**Config at this point (the baseline we inherited):** `num_iterations=1`,
`beta=0`, `lr=1.5e-6`, `fork_frac=0.5`, `eps=0.2`, `lora_r=64`,
`apply_divergent_mask=True`, `scale_rewards=False` (implicit; run3's
default didn't stick on main).

**Outcome after 1500+ steps:**

- `reward` MA100 flat at ~1.05, `correctness` MA100 flat at ~0.20.
- `loss` ≈ 0 and `grad_norm` small.
- `frac_reward_zero_std` ~0.22 → ~22% of GRPO groups have no variance
  and produce zero advantage.
- `divergent_frac` ~0.21 → ~78% of completion tokens were zeroed by
  the divergent mask, killing gradient on nearly everything.
- `strict_format_reward_func` permanently 0.

**Interpretation:** with `num_iterations=1` + `β=0`, the PPO clip
region and the KL penalty both never activate; the loss reduces to the
on-policy identity `-mean(advantage) ≈ 0`. The only learning signal is
the gradient of `log π` times a sparse, masked advantage. With 78% of
tokens masked out and 22% of groups producing zero-std advantage, the
effective signal-to-noise ratio was far too low to move LoRA at
`lr=1.5e-6`.

---

## run5 attempt 1 — "turn things back on"

**Change:** `num_iterations=2, beta=0.02, lr=1e-5`.

**Motivation:** if `β=0` and `num_iter=1` makes the loss identically 0
on-policy, re-engage the PPO clip (by taking multiple policy steps
between rollouts) and add a KL penalty so there is actually a loss
surface to optimise.

**Outcome:** **exploded in 7 steps.** `kl = 1.6e10`, `loss = 7e8`,
`grad_norm = 5e9`. Killed.

**Root cause:** TRL's KL is `k3 = exp(r) − 1 − r` where
`r = logp_ref − logp_policy`. On dLLM masked positions, `|r|` can be
50+ because both factors are unbounded log-probs of a discrete
denoising step. `kl_ratio_clip=5.0` only clamps the `logp_ref - logp_old`
term at rollout time; on iteration > 0 the ratio becomes
`(logp_ref - logp_old) + (logp_old - logp_policy)` and the second term
is uncapped → `kl` and the loss grow unbounded in one iteration.

---

## run5 attempt 2 — "tighten everything"

**Change:** `lr=3e-6, kl_ratio_clip=2.0, scale_rewards=True`.

**Motivation:** smaller lr + tighter KL clip should slow the blow-up
enough to be usable.

**Outcome:** still exploding. `grad_norm = 8e11` by step 10. Killed.

**Lesson learned:** the KL / PPO re-engagement path on dLLMs is a dead
end without a proper KL estimator that handles masked positions
robustly. Rolled back.

---

## run5 (attempt 3) — stable baseline

**Change (from run4):** `scale_rewards=True`, `lr=3e-6`,
`apply_divergent_mask=False`, fix cross-rank inconsistency in
`adv_scale` (all-reduce the mean divergent-frac).

**Motivation:** keep `num_iter=1, β=0` to sidestep the KL issue, but
unstick learning by (a) increasing lr 2×, (b) removing the
78%-of-tokens mask so gradient can actually flow, and (c) normalising
advantages so the scale isn't dominated by GSM8K's multi-modal reward.

**Outcome:** **stable.** Loss ≈ -0.003, grad_norm ≈ 0.6, reward MA20 =
1.19, correctness MA20 = 0.23. No explosion. But correctness still
doesn't trend up meaningfully over 159 steps — signal is stable but
sparse. This is the reference baseline for everything that follows.

---

## run6 — introduce learned `fork_frac` (ForkHead v1)

**Change:** add a small Gaussian-policy ForkHead (sigmoid
parameterisation over `[0.2, 0.8]`) trained with REINFORCE using
`reward − EMA(reward)` as the advantage. `max_completion_length=266`.
`fork_head_lr=1e-3`.

**Motivation:** replace the hand-picked `fork_frac=0.5` with a
per-prompt learnable choice. The original hypothesis was that different
prompts want different shared-vs-divergent trade-offs.

**Outcome:** `fork_frac_mean` **stuck at 0.500 exactly** for the
entire run. Main GRPO metrics unchanged. Killed early.

**Discovery:** `log_sigma` was moving, so backward and Adam clearly
worked — but `proj.weight.grad` and `proj.bias.grad` were identically
zero. See run7/run8 for the cause.

---

## run7 — ForkHead v2: fp32 + direct linear parameterisation

**Changes:**

1. Force ForkHead to fp32 (previously cast to bf16 with the main
   model). Rationale: Adam moments at bf16 precision can underflow the
   tiny per-step REINFORCE updates.
2. Switch `μ = sigmoid(logit)·(hi−lo)+lo` → `μ = clamp(proj(h), lo, hi)`.
   Rationale: the sigmoid has derivative 0.15 at the origin — every
   unit of bias movement only moves μ by 0.15 units, attenuating
   updates by ~7×. Direct linear gives 1:1.

**Motivation:** both of these were cleanups based on the run6 no-learning
symptom, **before** the real bug was identified.

**Outcome:** `fork_frac_mean` **still exactly 0.5**. Neither fix helped,
which narrowed the diagnosis to a gradient-flow bug rather than a
precision / parameterisation bug.

---

## run8 — the `rsample()` bug

**Change:** in `ForkHead.sample`, replace
`raw = dist.rsample()` with `raw = dist.sample().detach()`.

**Motivation:** REINFORCE requires the action to be a **constant**
w.r.t. policy parameters — the gradient comes through `log π(a|s)`, not
through `a`. With the reparameterised `rsample`, `raw = μ + σ·ε` so
`log p(raw)` has two paths to μ: the direct one and the one through
`raw`. For a Gaussian these have equal magnitude and opposite signs
and cancel exactly. That is why `μ`'s gradient was zero but `σ`'s
wasn't (the σ-path doesn't cancel because `raw` is multiplicative in σ).

**Outcome:** μ finally **moves**. But too aggressively: step 2→3 jump
from 0.5 to 0.8 (clamp upper bound) in a single REINFORCE step; μ then
pinned at 0.8 for the rest of the run.

---

## run9 — try lowering lr

**Change:** `fork_head_lr=1e-2` → `1e-3`.

**Motivation:** the obvious hypothesis: updates are too large.

**Outcome:** still saturates at 0.8 within ~3 steps. **lr alone
cannot fix it.**

**Diagnosis:** the problem is not the lr magnitude, it is the
**dimensionality** of the weight vector. For a single `Linear(4096 → 1)`,
Adam's first-step update has ~lr magnitude **per coordinate**; the
dot-product of that update with the next (correlated) prompt's hidden
state is `O(lr · √H · E[|h|])`. For lr=1e-3 and LLaDA scale
`E[|h|]~0.5`, that's already 0.3 per step — enough to blow past the
[0.2, 0.8] clamp in one or two updates, regardless of how small the
lr is in absolute terms.

---

## run10 — ForkHead v3: LayerNorm + low-rank bottleneck

**Change:** `Linear(4096 → 1)` replaced by
`LayerNorm(4096) → Linear(4096 → 8) → Linear(8 → 1)`. Bottleneck
weights init `N(0, 0.02²)`, `proj.weight=0`, `proj.bias=(lo+hi)/2`.

**Motivation:** cap the effective weight dimension at 8 so the
`√H · |h|` term becomes `√8 · 1 ≈ 3`, two orders of magnitude smaller.
LayerNorm controls input scale so the bound actually holds.

**Outcome:** **μ drifts smoothly.** Over the first 19 steps:
0.500 → 0.506 → 0.519 → 0.541 → 0.572 → 0.601 → 0.679 → **0.775**.
Per-prompt variation in the sampled action preserved (range
0.245 – 0.799).

**But:** μ eventually saturates at 0.8 and pins there. Over 122 steps,
reward MA20 dropped to 0.69 and correctness to 0.13 — **worse than run5's
stable baseline (reward 1.19, corr 0.23)**. The head clearly learned
to pick high fork_frac consistently; high fork_frac = more shared
rollout ⇒ less intra-group variance ⇒ more zero-std groups ⇒ weaker
main-policy learning signal. This is concerning.

**Cause, identified post-hoc:** the REINFORCE advantage `reward − EMA(reward)`
mixes together two very different signals:

1. "This fork_frac was good/bad for this prompt."
2. "This prompt was easy/hard."

Because easy GSM8K prompts give reward ≈ 2.5 and hard ones ≈ 0.1
regardless of fork_frac, the *prompt difficulty* term totally dominates
the *fork_frac quality* term. The head ends up learning to predict
"am I looking at an easy-prompt direction?", then picking whatever
fork_frac it happened to sample on past easy prompts. That's a
**difficulty-detector**, not a fork_frac policy.

---

## run11 — actor-critic: per-prompt value baseline

**Change:** add a `value_head: Linear(8 → 1)` sharing the 8-d
bottleneck features. Train jointly:

```
advantage   = reward − V(h).detach()
policy_loss = -log π(a|h) · advantage
value_loss  = (V(h) − reward)²
total       = policy_loss + value_loss
```

The EMA(reward) stat is kept for dashboard display only.

**Motivation:** `V(h)` absorbs the prompt-difficulty component. The
remaining advantage `r − V(h)` is the genuine signal "was this
fork_frac choice above or below expectation **for this prompt**?" —
which is what the policy head actually needs in order to learn
conditional-on-prompt behaviour rather than a prompt-difficulty
classifier.

**Outcome (243 steps, post-mortem):**

- `V(h)` regressed toward observed reward cleanly; MA100 gap stayed ≤ 0.1.
- `fork_frac_mean` hit 0.800 (the upper clamp) by step ~10 and **never
  came back**. `fork_sigma` also stuck at its init 0.20.
- `reward` MA100 = 0.90, `correctness` MA20 = 0.22.  The run **did
  improve** (first-third corr 0.154 → last-third 0.181, Δ=+0.027), but
  at this slope you'd need 1000+ more steps to reach run1's 0.47.
- Per-step S/N = (last − first) MA5 / σ(MA5) = **−0.50** — the trend
  is buried in the noise floor, which is why the dashboard looked
  "no stable growth".

**Three things run11 exposed that run6–10 hadn't:**

1. **ForkHead `mean` clamp kills gradient.** `mean = clamp(raw_mean,
   lo, hi)` sends `∂mean/∂raw_mean = 0` once `raw_mean` exits `[lo,
   hi]`. Even with V(h) pulling advantages toward 0, `proj.weight/bias`
   can't move because the computational graph is severed. This is a
   5th entry in the `FORK_HEAD.md` bug diary.
2. **98% of completions hit the 266-token cap.** `max_completion_length
   = 266` is too tight for GSM8K CoT + XML tags. Generations get
   truncated mid-`</answer>`, breaking the format regex and silently
   zeroing correctness on completions that *did* solve the math.
3. **`xmlcount_reward_func` had a negative tail-penalty.**
   `count -= len(text_after_</answer>) * 0.001` went NEGATIVE on most
   completions (LLaDA fills a fixed window). Anti-correlated with
   correctness (longer correct answers got more penalty). Its mean was
   −0.05 throughout run11.

These three turn a sympathetic reading of run11 ("V is learning, need
more steps") into a structural one: **the base-GRPO signal has been
starved by reward-function pathologies + truncation + too few gradient
updates per batch + too-small G**. The fork-head story is secondary.

---

## run12 — deep-diagnosis restart after the run11 post-mortem

**Changes (all at once; see `scripts/launch_btgrpo_run12.sh`):**

| change | from | to | why |
| --- | --- | --- | --- |
| `max_completion_length` | 266 | **512** | run11 hit the cap on 98% of steps; B200 has the memory |
| `num_iterations` | 1 | **4** | per-batch only got 1 gradient update; run1 (which reached corr 0.47) had 4. `β=0` stays, so no KL-estimator explosion |
| `block_size` | 32 | **64** | halve outer sequential-block count for B200 parallelism (9 → 5 blocks) |
| `num_generations` (G) | 4 | **8** | 22% of groups had zero-std advantage; bigger G makes group baselines meaningful |
| `per_device_train_batch_size` | 4 | **8** | keep `per_device_bs == G` so each GPU holds exactly 1 fork group. Doubles unique prompts/step → σ per step roughly /√2 |
| `reward_weights` strict_format | 0.25 | **0.0** | `strict_format_reward_func` was permanently 0 — 12.5% dead weight |
| `count_xml` code | tail penalty | **removed** | penalty was anti-correlated with correctness; ran negative in run11 |
| `learn_fork_frac` | True | **False** | disabled until base-GRPO is actually teaching the model; re-enable once the base shows clear correctness growth |

**What this does NOT change (intentionally, to isolate variables):**

- `lr=3e-6`, `beta=0`, `epsilon=0.2`, `lora_r=64, α=32`, `scale_rewards=True`,
  `apply_divergent_mask=False`, `fork_frac=0.5`, `sync_ref_model=False`,
  `steps=64` (denoising), `p_mask_prompt=0.15`, `seed=42`.

**Hypothesis being tested:** run1's corr=0.47 came from the combination
of `num_iter=4` + `lora_r=128` + (some lucky fork/β settings). We never
tried `num_iter=4` with `β=0`. If run12 converges toward corr ≥ 0.4
over ~1000 steps, the "num_iter=1 was starving us" hypothesis wins and
we can safely keep `β=0` going forward.

**Success criteria (MA100):**

- `correctness ≥ 0.35` by step 500 (vs run5's ~0.23 at step 159)
- `reward` trending upward monotonically on MA100 (not flat or drifting)
- no NaN / grad blowups (if multi-iter PPO + bf16 dLLM has any latent
  instability, that's a new finding)

If run12 looks like another run5 (corr flat at ~0.22), we've eliminated
`num_iter`, block_size, G, reward weights, completion length as causes,
and the next suspects are `lora_r` and `lr`.

---

## run12 — early-stopped at 14 batches, informative enough

**What happened:** 14 real batches (56 logged steps with num_iter=4),
then user called it ("这么垃圾").

**What we learned:**

1. **Reward-function fixes validated:**
   - `xmlcount_reward_func/mean`: **-0.05 (run11) → +0.34 (run12)** — the
     tail-penalty removal worked as designed; it's now a pure bonus.
   - `completions/clipped_ratio`: **98% @ 266 (run11) → 0-15% @ 512
     (run12)** — truncation is no longer eating our format/correctness
     signal.

2. **`num_iter=4` on dLLM caused asymmetric PPO blow-up.** Out of 14
   batches, **3** (#1, #3, #5, #8) had `grad_norm` in the 1000-35000
   range; the rest were 1-40. Pre-clip loss on those batches was in
   the 200-230k range. Cause: TRL's PPO clip `min(r·A, clip(r,1±ε)·A)`
   only bounds the objective when `A > 0` and `r` is large. For
   `A < 0` with large `r` the unclipped branch wins and the loss
   magnitude is unbounded. With `num_iter=4`, ratio `r = π_new/π_old`
   accumulates drift across the 4 reuse steps — drift per step is
   small, but for dLLM non-autoregressive log-prob sums over many
   tokens, 4 steps is enough to make r wander to `exp(±5)` tail. The
   global `max_grad_norm=1.0` prevents actual parameter explosion, so
   the run is safe to watch but the gradient signal is dominated by
   clip artefacts rather than real advantages.

3. **`frac_reward_zero_std` averaged 0.72 even with G=8.** Hypothesis:
   base policy is so weak that most hard GSM8K prompts produce 8/8
   wrong answers → zero group variance → zero advantage. Raising G
   helps less than expected; the fix is either a warmer base
   (SFT first) or downsampling hard prompts.

4. **Correctness at 14 batches (= 112 unique prompts) is pure prompt
   noise.** Batch-wise corr ∈ {0, 0.03, 0.13, 0.25, 0.28, 0.75}. First-7
   mean 0.174, last-7 mean 0.111, difference ~0.06 vs per-batch
   σ≈0.22 (S/N = 0.27 — invisible). We don't have a trendable sample
   after 14 batches.

**Inherited fixes carried into run13** (validated): xmlcount patch,
max_comp=512, block=64, G=8, per_device_bs=8, strict_format weight 0,
scale_rewards=True, β=0, ε=0.2.

---

## run13 — learned fork_frac re-enabled, range [0,1], num_iter=4→2

**Two independent changes:**

### (a) Fork head re-enabled with widened range + clamp-bug fixed

Config deltas: `--learn_fork_frac True`, `--fork_frac_min 0.0`,
`--fork_frac_max 1.0` (was [0.2, 0.8] in run6-11 + disabled in run12).
Bias init is `0.5·(lo+hi) = 0.5`, so step-0 behaviour is identical to
hand-picked `fork_frac=0.5` in run12 — if the head misbehaves we
degrade gracefully.

Code fix in `fork_head.py::_mean_sigma` (the §5.5 bug): removed
`mean = raw_mean.clamp(lo, hi)`. The mean is now the raw (unbounded)
linear output; only the *sampled action* is clamped at use time. This
keeps `dπ/dθ` alive everywhere — REINFORCE can pull the mean back in
from either side regardless of saturation.

Widening the range to [0, 1] has two purposes:
- removes the "optimal frac is at the boundary" scenario (in [0.2, 0.8],
  run11 consistently wanted to reach toward 1.0 but couldn't);
- makes the clamp-severing failure mode (if fix regresses) visible
  immediately — μ saturating at 0.999 is obviously broken.

### (b) num_iterations 4 → 2

Direct response to the grad_norm spikes observed in run12. Halving PPO
reuse halves the expected ratio drift per batch while still giving 2×
the gradient density of num_iter=1 (which run5-11 showed to be too
starved to see learning on GSM8K in reasonable wall-clock).

**What we're watching for (first 50 batches):**

- `grad_norm` staying < 100 on > 90% of batches (was ~75% on run12)
- `fork_frac_mean`: does μ *move* from 0.5? in which direction?
  does it keep moving vs run11-style saturation?
- `value_loss`: decreasing, tracks `reward` MA
- `correctness` MA20 trending up from its ~0.12-0.22 run12/11 baseline

**If it still doesn't learn at 200 batches** (MA50 corr < 0.25), next
levers to toggle in order of cheapness: `lr 3e-6 → 1e-5`, then
`lora_r 64 → 128` (matches run1), then reconsider whether SFT warmup
is needed before RL can work from this base.

---

## Cross-cutting lessons

1. **GRPO on dLLMs: keep it on-policy.** `num_iter=1, β=0` is the only
   stable configuration we found. Re-engaging PPO clip or KL penalty
   caused immediate blow-up because TRL's KL estimator is unbounded on
   masked denoising steps.

2. **`apply_divergent_mask=True` kills the signal.** With G=4 fork
   siblings, ~78% of completion tokens get masked. Turning it off was
   necessary to get any learning.

3. **ForkHead / any auxiliary head should use a low-rank bottleneck +
   LayerNorm.** A dense `Linear(H → 1)` REINFORCE head on H=4096 is
   fundamentally unstable: a single Adam step produces updates whose
   dot-product with the next prompt saturates whatever output clamp
   you put on. Bottleneck = 8 works; bottleneck = 1 would also work
   but kills per-prompt conditioning.

4. **REINFORCE ≠ reparameterised sampling.** Use `.sample().detach()`
   for the action. `rsample()` cancels the policy gradient w.r.t. the
   distribution mean.

5. **Always use a per-prompt value baseline** when the reward is
   strongly prompt-dependent. A global EMA baseline makes any auxiliary
   policy converge to a "difficulty detector" rather than a policy.

6. **fp32 for tiny auxiliary heads.** Even if the main model is bf16,
   Adam moments for tiny heads need fp32 to accumulate the small
   REINFORCE updates without underflow / stalling.

7. **Keep legacy default as a recoverable state.** ForkHead is
   initialised so that μ(h) = (lo+hi)/2 exactly at step 0, i.e.
   equivalent to the old hand-picked fork_frac=0.5. If the learned
   head misbehaves, we can always drop `--learn_fork_frac True` and
   recover the previous baseline verbatim.

8. **`clamp` on a policy mean severs the gradient.** `mean = clamp(raw,
   lo, hi)` has `∂mean/∂raw = 0` outside `[lo, hi]`. If the raw
   parameterisation drifts past the boundary once, REINFORCE can never
   pull it back. Clamp only the sampled action, never the distribution
   parameter itself (§5.5 in FORK_HEAD.md).

9. **Always check if you're hitting `max_completion_length`.** run11
   discovery: 98% of completions truncated at the cap. Truncation
   destroys format rewards + answer parsers → the measured correctness
   floor is artificially low even when the underlying policy is fine.
   When in doubt, double the cap.

10. **Reward functions can silently go negative.** `xmlcount`'s
    "penalty for tail text after </answer>" made its mean hover at
    −0.05, cancelling correctness gains. Always eyeball per-component
    reward means early in a run; any persistently-negative component
    is suspect.

11. **Watch for dead-weight reward components.** `strict_format` was
    permanently 0 in every run 4–11 — 12.5% of reward-weight budget
    was just padding. Drop or rewrite any reward that never fires.

12. **Binary correctness × G=4 = huge sampling noise.** Per-step σ(reward)
    ≈ 0.9 mostly from Bernoulli sampling of 16 completions, not from
    optimisation. Trends need MA50+ to surface. Bigger G + bigger
    per-device batch is cheap variance reduction on B200-class GPUs.

13. **TRL's PPO clip is asymmetric.** `min(r·A, clip(r, 1-ε, 1+ε)·A)`
    bounds the objective only for `A > 0` with large `r`. For `A < 0`
    with large `r`, the unclipped branch wins and the logged loss /
    grad-norm can look catastrophic. On dLLMs with `num_iter > 2` the
    ratio drift is enough to trigger this frequently. Mitigation:
    (a) keep `num_iter ≤ 2`, or (b) rely on `max_grad_norm=1.0` to
    bound actual parameter updates and accept that displayed loss is
    cosmetic on spike batches.

14. **`frac_reward_zero_std` is a policy-quality diagnostic.** High
    values (>0.5) mean the group variance reducer can't help because
    most groups have 8 identical rewards. This is mostly a sign that
    the base is too weak for most prompts — warmup (SFT) or curriculum
    (drop hardest 30% of prompts early) addresses it. G alone doesn't.
