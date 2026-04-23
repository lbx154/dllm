# BT-GRPO on LLaDA-8B — Experiment Log

This fork turns the upstream `dllm` repo into a working lab for
**Branching-Trajectory GRPO** (BT-GRPO) on `LLaDA-8B-Instruct` + LoRA,
trained on GSM8K. This README is the living log of every training run
we've done, what was changed, and what happened. Deep-dive docs live
alongside the code:

- [`dllm/pipelines/rl/btgrpo/RUN_HISTORY.md`](dllm/pipelines/rl/btgrpo/RUN_HISTORY.md) — full per-run notes (run1 → run11)
- [`dllm/pipelines/rl/btgrpo/FORK_HEAD.md`](dllm/pipelines/rl/btgrpo/FORK_HEAD.md) — learned `fork_frac` head design + bug diary

---

## Workflow rule (from run12 onward)

**Every training restart is preceded by a git commit.** The commit is
the archival record of the source-tree state for that run; without it
we lose the run ↔ code mapping. Format:

```
runN: <one-line delta summary>

<optional multi-line detail>

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
```

run1–run4 follow this rule already. run5–run11 were done in one session
without commits and were collapsed into a single retroactive commit
`c028d8b` capturing the run11 code state.

---

## Hardware / config baseline

- 8× A100-40G, single node.
- Model: `LLaDA-8B-Instruct` + LoRA (`r=64`, `alpha=32` from run2 onward).
- Data: GSM8K with reasoning prompt; `num_generations G=4`,
  `per_device_train_batch_size=4` → **1 fork-group per GPU**.
- Reward weights: `[xmlcount 0.25, soft_format 0.25, strict_format 0.25,
  int 0.25, correctness 5.0]` from run2 onward.
- Generation: 64 denoising steps, `max_completion_length=266` from run6.

---

## Run summary

All rewards/correctness are MA20 at end of run. "Corr" =
`rewards/correctness_reward_func/mean` in `[0, 2]`. "fork μ" is the
mean of the head's sampled `fork_frac` (always 0.5 for fixed-fork runs).

| Run     | Commit        | Δ from previous                                                                                   | Steps | MA20 loss | MA20 reward | MA20 corr | MA20 grad | fork μ | Outcome                                |
| ------- | ------------- | ------------------------------------------------------------------------------------------------- | ----: | --------: | ----------: | --------: | --------: | -----: | -------------------------------------- |
| run1    | `85576bc`     | initial BT-GRPO launch: `num_iter=4, β=0.02, lr=3e-6, eps=0.5, fork=0.5, lora_r=128`               |   n/a |       n/a |         n/a |  **0.47** |       n/a |    0.5 | early corr OK, grad blowups            |
| run2    | `14c1b63`     | `num_iter 4→1, β 0.02→0.04, eps 0.5→0.3, lr 3e-6→1.5e-6, fork 0.5→0.35, lora_r 128→64, corr×5`     |   n/a |       n/a |         n/a |       n/a |       n/a |   0.35 | **KL explosion ~1e9**                  |
| run3    | `0be8638`     | code fixes: `kl_ratio_clip=5.0`, 1/f_D advantage, `scale_rewards=True`                             |   n/a |       n/a |         n/a |  **0.18** |       n/a |   0.35 | KL still 1e8, β·KL dominates 1000:1    |
| run4    | `bae7590`     | `β 0.04→0`, `fork 0.35→0.5`, `eps 0.3→0.2`, `sync_ref_model=False`                                 | 1500+ |        ~0 |      ~1.05  |    ~0.20  |       low |    0.5 | **flat, not learning**                 |
| run5.a1 | _(c028d8b)_   | `num_iter=2, β=0.02, lr=1e-5, scale_rewards=F`                                                     |     7 |   **7e8** |      0.88   |    0.17   |   **5e9** |    0.5 | **KL blew up — killed**                |
| run5.a2 | _(c028d8b)_   | `num_iter=1, β=0, lr=3e-6, scale_rewards=T, kl_ratio_clip=2.0`                                     |    10 |   **1e7** |      1.17   |    0.23   | **8e11**  |    0.5 | still exploding — killed               |
| run5    | _(c028d8b)_   | same as a2 but `apply_divergent_mask=False`, fix per-rank `adv_scale`                              |   159 |   -0.003  |      1.19   |    0.23   |     0.62  |    0.5 | **stable; correctness flat**           |
| run6    | _(c028d8b)_   | +learned fork_frac (sigmoid policy, lr=1e-3), `max_completion_length=266`                          |    34 |   -0.003  |      0.85   |    0.16   |     0.43  | **0.500** | fork head not moving                |
| run7    | _(c028d8b)_   | +fp32 ForkHead, direct linear param, lr=1e-2                                                       |    17 |    0.019  |      1.29   |    0.25   |     0.59  | **0.500** | fork head **still** 0.5             |
| run8    | _(c028d8b)_   | REINFORCE bug fix: `rsample` → `sample().detach()`                                                 |    11 |    0.022  |      1.19   |    0.23   |     0.45  |  0.745 | μ moves, **saturates 0.8** in 1 step   |
| run9    | _(c028d8b)_   | lr back to 1e-3                                                                                    |    19 |    0.010  |      1.31   |    0.25   |     0.50  |  0.768 | still saturates by step 3              |
| run10   | _(c028d8b)_   | +LayerNorm + bottleneck 4096→8 ForkHead                                                            |   122 |    0.002  |      0.69   |    0.13   |     0.44  |  0.800 | μ drifts smoothly then saturates; reward **drifts down** |
| run11   | _(c028d8b)_   | +value-head baseline `V(h)` (actor-critic replaces EMA baseline)                                   |    25 |   -0.004  |      0.89   |    0.17   |     0.44  |  0.762 | early; V(h) learning; observing        |

See `RUN_HISTORY.md` for the per-run write-up (motivation, config,
diagnosis) and `FORK_HEAD.md` for the fork-head bug diary.

---

## Cross-cutting lessons

1. **`num_iterations=1, β=0` is the only stable GRPO setting on dLLMs.**
   The k3 KL estimator (`exp(r) − 1 − r`) overflows on masked positions
   where `|log π_ref − log π_policy|` can reach 50+, producing KL ~1e9.
   The PPO ratio is also always 1 on the step where `old_logps` were
   collected. PPO clip is the sole trust region.
2. **`apply_divergent_mask=True` kills learning.** It zeroed ~78% of
   completion-token gradients in run4. Leave it off.
3. **Per-rank `adv_scale` must be all-reduced.** Otherwise each rank
   sees a different advantage scale and gradients don't align at the
   reduce step — silently introduces bias.
4. **REINFORCE wants `sample().detach()`, not `rsample()`.** `rsample`
   puts a pathwise gradient through the mean that cancels the
   `log π · A` term analytically; head never updates (runs 6–7).
5. **Dense `Linear(4096→1)` as a REINFORCE head is intrinsically
   unstable.** One Adam step's dot-product with the next prompt is
   `O(lr · √H · |h|)`, which saturates the clamp regardless of lr.
   Use a LayerNorm + low-rank bottleneck `4096→8→1` (run10+).
6. **Global EMA baseline ⇒ difficulty-classifier failure mode.**
   `A = r − EMA(r)` makes the head learn "is this prompt easy?" rather
   than "what fork_frac is best for this prompt?". Use a per-prompt
   value head `V(h)` (actor-critic, run11+).
7. **Keep aux heads in fp32.** bf16 weights + Adam moments underflow.
8. **`strict_format_reward_func` is permanently 0** in our log — 12.5%
   of the reward-weight budget is wasted. Consider dropping or rewriting.

---

## Repo layout (what matters)

```
dllm/pipelines/rl/btgrpo/
  trainer.py         BT-GRPO trainer; ForkHead integration, actor-critic update
  fork_head.py       learned per-prompt fork_frac head (LN + 4096→8→{π, V}, fp32)
  FORK_HEAD.md       head design + bug diary
  RUN_HISTORY.md     full per-run notes
examples/rl/grpo/llada/
  train_btgrpo.py    entry point (TrlParser → BTGRPOConfig)
scripts/
  launch_btgrpo_run{5..11}.sh   per-run launchers
dashboard.py         live training dashboard (10 panels)
monitor.py           lightweight log tail helper
```

---

## Running

**Monitor a live run:**

```bash
python dashboard.py --log .logs/btgrpo-run11.log --refresh 30
```

**Launch a new run (pre-commit required):**

```bash
git add -A
git commit -m "run12: <delta>" \
           -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
bash scripts/launch_btgrpo_run12.sh
```

The launcher emits to `.logs/btgrpo-run<N>.log`. Archived old logs sit
in `.logs/archive/`.
