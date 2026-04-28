"""Failure signatures derived from 16-run BT-GRPO log corpus.

Each signature is a pure function on a list-of-step-dicts (the same dicts the
trainer prints, parsed via ast.literal_eval). Thresholds were calibrated on
historical runs in `session-state/.../files/runs_features.csv`.

Categories (matches docs/RUN_HISTORY.md root-cause taxonomy):

  grad_blowup         : KL/PPO ratio explosion             — run12, run13
  starved_signal      : zero-std groups dominate           — run14-19
  fork_saturated      : fork head pinned at boundary       — run13, run16
  len_collapsing      : completion length monotonic down   — run8, run9, run18
  corr_dead_early     : correctness reward stuck near 0    — (none observed)
  corr_negative_slope : correctness trending downward      — run4, run6, run8 ...
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Iterable, Sequence
import math
import numpy as np


# ---------------------------------------------------------------------------
# Metric keys we read from trainer step dicts. Centralised so a future trainer
# rename only touches this file.
# ---------------------------------------------------------------------------
K_GRAD     = "grad_norm"
K_LOSS     = "loss"
K_REW_STD  = "reward_std"
K_FZS      = "frac_reward_zero_std"
K_LEN      = "completions/length/mean"
K_CORR     = "rewards/correctness_reward_func/mean"
K_FORK_MU  = "btgrpo/fork_frac_mean"
K_FORK_SAMP = "btgrpo/fork_frac"
K_ENTROPY  = "entropy"


# ---------------------------------------------------------------------------
# Thresholds. These came from the 50-step early-window analysis in
# extract_run_features.py; they are intentionally loose (we want recall, not
# precision — Tier 2 is advisory, only Tier 1 aborts).
# ---------------------------------------------------------------------------
@dataclass
class Signatures:
    # ---- Tier 1 hard-abort thresholds (single-step or short-window) -------
    # NOTE: BT-GRPO + dLLM produces large pre-clip grad_norms (1e8–1e11) on
    # mask/padding tokens because exp(r) in the k3 KL estimator can saturate
    # at e^5 ≈ 148 per token. With max_grad_norm=1.0 (HF default), the *applied*
    # update is bounded regardless. So single-step grad spikes are NOT fatal —
    # we only abort on *sustained* explosion, on true loss runaway, or on
    # confirmed reward regression.  See run22 postmortem (steps 1–12 had
    # grad_max=1.3e11 yet correctness rose 0.42→0.52).
    abort_grad_norm:           float = 1.0e13   # raised from 1e4: single step
    abort_grad_blowup_window:  int   = 20
    abort_grad_blowup_thresh:  float = 1.0e10   # raised from 1e3: per-step
    abort_grad_blowup_frac:    float = 0.50     # raised: need MAJORITY exploded
    abort_loss_jump_factor:    float = 1.0e6    # raised from 100: BT-GRPO loss
                                                # routinely jumps 100x between
                                                # steps as KL breathes
    abort_loss_jump_floor:     float = 1.0e6    # raised from 1: only count
                                                # jumps from already-large loss
    # Reward-regression abort: fires when correctness has been monotonically
    # falling for `abort_corr_decline_window` consecutive steps AFTER step
    # `abort_corr_warmup_steps` (don't fire during warmup transient).
    abort_corr_warmup_steps:   int   = 30
    abort_corr_decline_window: int   = 8
    abort_corr_decline_drop:   float = 0.15     # drop from window peak required
    # Note: fork-head saturation is RECOVERABLE (re-init the head, freeze it,
    # drop value coupling). It is NOT a Tier-1 abort condition. The "fork
    # saturated" check lives in the advisory tier instead — see
    # sig_fork_boundary_*.

    # ---- Tier 2 advisory tripwires (window of N steps) -------------------
    window_size:               int   = 50
    sig_grad_blowup_frac:      float = 0.05
    sig_starved_signal_thresh: float = 0.50    # mean(frac_reward_zero_std)
    sig_fork_boundary_dist:    float = 0.45    # |fork_μ-0.5| > this
    sig_fork_boundary_frac:    float = 0.30    # ... in this fraction of window
    sig_len_slope_thresh:      float = -0.5    # tokens / step
    sig_corr_dead_thresh:      float = 0.05    # mean(correctness)
    sig_corr_neg_slope:        float = -1.0e-3 # corr slope per step

    # ---- Tier 3 trend extrapolation --------------------------------------
    target_corr:               float = 1.0     # ≥1.0 means model wraps answers
    trend_min_steps:           int   = 200     # only extrapolate after this
    trend_window:              int   = 100     # MA window for slope estimation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _col(rows: Sequence[dict], key: str) -> np.ndarray:
    """Extract a numeric column. Missing/NaN values become np.nan."""
    out = np.full(len(rows), np.nan, dtype=float)
    for i, r in enumerate(rows):
        v = r.get(key)
        if v is None:
            continue
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def _slope(y: np.ndarray) -> float:
    if len(y) < 5:
        return 0.0
    x = np.arange(len(y), dtype=float)
    m = np.isfinite(y)
    if m.sum() < 5:
        return 0.0
    return float(np.polyfit(x[m], y[m], 1)[0])


def _safe_mean(y: np.ndarray) -> float:
    y = y[np.isfinite(y)]
    return float(y.mean()) if len(y) else float("nan")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def summarise_window(rows: Sequence[dict]) -> dict:
    """Compute the 6 signature features for a window of step-dicts.
    Returns a flat dict of named scalars (NaN where data is unavailable)."""
    grad   = _col(rows, K_GRAD)
    rstd   = _col(rows, K_REW_STD)
    fzs    = _col(rows, K_FZS)
    length = _col(rows, K_LEN)
    corr   = _col(rows, K_CORR)
    fork   = _col(rows, K_FORK_MU)
    if np.isnan(fork).all():
        fork = _col(rows, K_FORK_SAMP)

    return {
        "n_steps":            len(rows),
        "grad_max":           float(np.nanmax(grad)) if np.isfinite(grad).any() else float("nan"),
        "grad_blowup_frac":   float(np.mean(grad > 1e3)) if len(grad) else float("nan"),
        "reward_std_mean":    _safe_mean(rstd),
        "frac_zero_std_mean": _safe_mean(fzs),
        "len_slope":          _slope(length),
        "corr_slope":         _slope(corr),
        "corr_early_mean":    _safe_mean(corr),
        "fork_at_boundary":   float(np.mean(np.abs(fork - 0.5) > 0.45)) if np.isfinite(fork).any() else float("nan"),
        "fork_var":           float(np.nanvar(fork)) if np.isfinite(fork).any() else float("nan"),
    }


def evaluate_signatures(
    rows: Sequence[dict],
    sigs: Signatures | None = None,
) -> dict[str, dict]:
    """Evaluate Tier 1 (abort) and Tier 2 (advisory) signatures on a window.

    Returns
    -------
    {
      "abort": {
          "grad_norm_step":    bool,   # any step > sigs.abort_grad_norm
          "grad_blowup_window": bool,  # last N steps blowup-frac > thresh
          "loss_jump":         bool,   # 5-step OoM jump (gated by abs-loss floor)
      },
      "advisory": {
          "grad_blowup":         bool,
          "starved_signal":      bool,
          "fork_saturated":      bool,
          "len_collapsing":      bool,
          "corr_dead_early":     bool,
          "corr_negative_slope": bool,
          "n_fired":             int,
      },
      "features": summarise_window(rows),
    }
    """
    sigs = sigs or Signatures()
    feat = summarise_window(rows)

    # --- abort checks --------------------------------------------------
    grad = _col(rows, K_GRAD)
    loss = _col(rows, K_LOSS)

    abort = {
        "grad_norm_step":      bool(np.any(grad > sigs.abort_grad_norm)),
        "grad_blowup_window":  False,
        "loss_jump":           False,
    }
    if len(rows) >= sigs.abort_grad_blowup_window:
        recent = grad[-sigs.abort_grad_blowup_window:]
        recent = recent[np.isfinite(recent)]
        if len(recent):
            abort["grad_blowup_window"] = (
                float(np.mean(recent > sigs.abort_grad_blowup_thresh))
                > sigs.abort_grad_blowup_frac
            )
    if len(rows) >= 6:
        # 5-step OoM jump: ratio between current step and the median of the
        # previous 5. Gated by abort_loss_jump_floor to avoid firing on the
        # early-training noise regime where |loss| ~ 0.01.
        finite = loss[np.isfinite(loss)]
        if len(finite) >= 6:
            recent = np.abs(finite[-6:])
            ref = max(np.median(recent[:-1]), 1e-6)
            if recent[-1] > sigs.abort_loss_jump_floor and recent[-1] / ref > sigs.abort_loss_jump_factor:
                abort["loss_jump"] = True

    # Reward-regression check (advisory): correctness has fallen by
    # `decline_drop` from its peak in the recent window. NOT a Tier-1 abort
    # because (a) noisy small-batch correctness can swing without true
    # regression, and (b) recoverable via LR/β tuning. The watcher should
    # surface this as a warning to the operator, not auto-kill.
    corr_regression = False
    corr = _col(rows, K_CORR)
    if len(rows) >= sigs.abort_corr_warmup_steps + sigs.abort_corr_decline_window:
        finite_corr = corr[np.isfinite(corr)]
        if len(finite_corr) >= sigs.abort_corr_decline_window + 5:
            tail = finite_corr[-sigs.abort_corr_decline_window:]
            peak = float(np.max(finite_corr[: -sigs.abort_corr_decline_window]))
            tail_mean = float(np.mean(tail))
            if peak > 0.3 and tail_mean < peak - sigs.abort_corr_decline_drop:
                corr_regression = True

    # --- advisory checks (window-level) -------------------------------
    advisory = {
        "grad_blowup":          (feat["grad_blowup_frac"] or 0)         > sigs.sig_grad_blowup_frac,
        "starved_signal":       (feat["frac_zero_std_mean"] or 0)       > sigs.sig_starved_signal_thresh,
        "fork_saturated":       (feat["fork_at_boundary"] or 0)         > sigs.sig_fork_boundary_frac,
        "len_collapsing":        feat["len_slope"]                      < sigs.sig_len_slope_thresh,
        "corr_dead_early":      (feat["corr_early_mean"] or 1)          < sigs.sig_corr_dead_thresh,
        "corr_negative_slope":   feat["corr_slope"]                     < sigs.sig_corr_neg_slope,
        "corr_regression":       corr_regression,
    }
    # Replace NaN-derived comparisons with explicit False
    for k, v in list(advisory.items()):
        if isinstance(v, float) and math.isnan(v):
            advisory[k] = False
        else:
            advisory[k] = bool(v)
    advisory["n_fired"] = int(sum(1 for k, v in advisory.items() if k != "n_fired" and v))

    return {"abort": abort, "advisory": advisory, "features": feat}
