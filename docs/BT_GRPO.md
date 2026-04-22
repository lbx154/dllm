# BT-GRPO — Branching Trajectory GRPO for Masked Diffusion LLMs

> Status: v0 proposal. Implementation is pushed alongside this doc.

## 1. TL;DR

All existing RL methods for masked diffusion LLMs (d1/diffu-GRPO, GDPO, SPG, ESPO, Co-GRPO, TIC-GRPO, DiPOD, AWM) treat the dLLM as a black-box AR replacement: each prompt gets **G independent rollouts**, each rollout has a **scalar sequence-level reward**, and the policy update uses a single log-prob surrogate across the full sequence.

**What this throws away.** A denoising trajectory `τ = (x_T → x_{T-1} → … → x_0)` visits a sequence of partially masked states, each of which is a **complete, valid sample-able state** (unlike AR, where intermediate states are strict prefixes). That means **we can share the early portion of a trajectory across multiple rollouts** and let them diverge later. This is structurally impossible in AR, and is the dLLM's main untapped RL-relevant property.

**BT-GRPO** turns this observation into a training algorithm:

1. For each prompt, denoise `x_T → x_{k*}` **once** (shared prefix of the trajectory).
2. From `x_{k*}`, fork G independent continuations, each finishing at a different `x_0^i`.
3. Reward each `x_0^i` as usual. Compute **group-relative advantage within the fork group**.
4. Apply the PPO/GRPO update only to the **post-fork (divergent) positions**: positions where all G branches agree carry zero credit automatically.

This gives **per-fork-step credit assignment** that in AR would require a learned critic or MCTS, but here is obtained critic-free through the group baseline, with **exact prefix-variance cancellation** guaranteed by the shared trajectory.

## 2. Theoretical Positioning

### 2.1 What d1 / diffu-GRPO does (the baseline)

Given a prompt q and G sampled completions {o_i}, d1 computes:
```
Â_i = r_i − mean_j r_j
log π_θ(o_i) ≈ single-step CE under a random mask
```
then applies PPO clip. Two sources of variance enter the advantage:
- **(a) shared-prompt variance**: "this prompt is hard" — cancelled by mean_j r_j. ✔
- **(b) rollout-trajectory variance**: different denoising trajectories produce different o_i — **not cancelled**, contaminates the advantage.

### 2.2 What BT-GRPO does

By sharing the prefix `x_T → x_{k*}` across the G fork-siblings, the trajectory-level variance up to x_{k*} is **also** cancelled in r_i − r̄. Only the variance from x_{k*} → x_0 remains, which is exactly what the update is attributing credit to.

Formally, if reward decomposes as `r_i = f(τ_i_pre) + g(τ_i_post) + ε_i`, then under BT-GRPO's shared-prefix construction `τ_i_pre = τ_pre` is deterministic across i, so
```
Â_i = g(τ_i_post) − mean_j g(τ_j_post) + ε_i − mean_j ε_j
```
stripping out f entirely. This is the diffusion analogue of a **shared-baseline control variate**, and it's free — we save forward passes rather than paying for them (Phase 1 runs once per prompt instead of G times).

### 2.3 Comparison with related work

| Method | Off-policy mask | Per-step credit | Prefix variance cancel | dLLM-specific structure |
|---|---|---|---|---|
| diffu-GRPO (d1) | ❌ | ❌ | ❌ | ❌ |
| TIC-GRPO | ❌ | ❌ | ❌ | ❌ |
| GDPO / SPG / ESPO | ❌ | ❌ | ❌ | ❌ |
| Co-GRPO | ⚠️ (learns scheduler) | ❌ | ❌ | ⚠️ |
| **BT-GRPO (ours)** | ✅ | ✅ | ✅ | ✅ |

The "dLLM-specific structure" column is the key: intermediate denoising states x_k are **non-causal complete states** — forking from them shares bidirectional context, which AR rollouts cannot do.

## 3. Algorithm

```
Input: prompt batch {q_1, …, q_B}, fork_frac f ∈ (0, 1), branches G, total steps T

# Phase 1 — shared trajectory (one per prompt)
for each q_b:
    x_b ← FullyMask(q_b, max_new_tokens)
    run MDLM denoise for ⌊f · T⌋ steps → x_b^{(fork)}

# Phase 2 — fork and diverge
for each q_b:
    for g = 1…G:
        x_{b,g} ← copy of x_b^{(fork)}
        run MDLM denoise for T − ⌊f · T⌋ steps, independent RNG → o_{b,g}

# Credit assignment
for each q_b:
    rewards r_{b,1}, …, r_{b,G} ← reward_fn(q_b, o_{b,g})
    Â_{b,g} ← r_{b,g} − mean_g r_{b,g}

# Policy update: restrict gradient to DIVERGENT positions
for each (b, g):
    divergent_mask_{b,g,t} = 1 iff o_{b,1,t}, …, o_{b,G,t} are not all equal
    loss contribution ∝ Â_{b,g} · ρ_{b,g,t} · divergent_mask_{b,g,t}
```

### Why `divergent_mask` is correct

For positions t where all G branches produced the same token, both the numerator and denominator of the PPO ratio are identical across the group, so their contribution to any group-relative advantage is zero. Explicitly masking them out just makes this exactness manifest (and saves a tiny bit of signal noise).

## 4. v0 implementation scope (this PR)

- **Single fork point** at `k* = ⌊f · T⌋`, f configurable (default 0.5).
- `BranchingMDLMSampler` reuses the MDLM step-loop logic but runs it in two phases with state handoff.
- `BTGRPOTrainer` subclasses `DiffuGRPOTrainer`: swaps sampler and multiplies `completion_mask` by a computed `divergent_mask`. All other machinery (PPO clip, β-KL, ref-sync) is inherited.
- **Not in v0**: multi-level (recursive) forking, adaptive fork depth, per-step TR-GRPO exact log-prob. These are separable follow-ups.

## 5. Hyperparameters chosen for the 4×8×A100-40GB launch

| Knob | Value | Rationale |
|---|---|---|
| `fork_frac` | 0.5 | Middle of the trajectory; half shared denoising, half divergent |
| `num_generations` (G) | 8 | Fork group size |
| `block_size` | 32 | Match d1 |
| `steps` | 128 | Match d1; gives 64 shared + 64 divergent per rollout |
| `p_mask_prompt` | 0.15 | Kept for logp estimate |
| `beta` | 0.02 | Halved vs d1 — BT-GRPO gives lower-variance gradients, KL can be softer |
| `num_iterations` | 4 | Cut from d1's 12 — cleaner signal per step |
| `epsilon` | 0.5 | Match d1 |
| `per_device_train_batch_size` | 2 | 40GB budget w/ ZeRO-3 + LoRA, during summon-full-params |
| `LoRA r / α` | 128 / 64 | Match d1 example |

Total effective batch = 32 GPUs × 2 = 64 completions per opt step = 8 unique prompts × G=8 branches.

## 6. Planned ablations (post-launch)

1. **vs d1 at equal compute**: expect BT-GRPO to win on sample efficiency due to prefix variance reduction.
2. **fork_frac sweep**: {0.25, 0.5, 0.75}; measures where credit assignment matters most.
3. **divergent_mask on vs off**: sanity check on theoretical claim.
4. **BT-GRPO + TR-GRPO-K1 IS**: the two orthogonal improvements should stack.

## 7. Files added

```
dllm/core/samplers/mdlm_branching.py     BranchingMDLMSampler
dllm/pipelines/rl/btgrpo/__init__.py
dllm/pipelines/rl/btgrpo/trainer.py      BTGRPOConfig + BTGRPOTrainer
examples/rl/grpo/llada/train_btgrpo.py   entry
examples/rl/grpo/llada/launch_btgrpo_4node.sh    torchrun-based multi-node launcher (AzureML)
docs/BT_GRPO.md                          this file
```
