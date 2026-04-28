"""Tier 3 — trend regression for the "slow flat plateau" failure mode.

Tests on runs that pass Tier 1 (no blowup) and Tier 2 (no signature gone bad)
but never actually learn — run5/run7/run10 in our corpus. We fit a simple
linear trend on the running mean of correctness_reward and ask:

    "Extrapolating to max_steps, will we cross target_corr?"

If not, abort. Also detects the reward-hacking three-way:
   d(corr)/dt ≤ 0  ∧  d(len)/dt < 0  ∧  d(format_r)/dt > 0
which is run10/run19's late-stage signature.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Sequence

from .signatures import (
    K_CORR, K_LEN, _slope, _col, Signatures,
)


@dataclass
class TrendResult:
    n_steps_seen:    int
    corr_slope:      float
    corr_current:    float
    extrapolated:    float       # value at max_steps if slope continues
    ok_will_reach:   bool        # extrapolated >= target_corr
    reward_hacking:  bool        # corr↓ + len↓ + format↑
    details:         str


class TrendExtrapolator:
    def __init__(
        self,
        max_steps: int,
        sigs: Signatures | None = None,
    ):
        self.max_steps = max_steps
        self.sigs = sigs or Signatures()

    def evaluate(self, rows: Sequence[dict]) -> TrendResult | None:
        sigs = self.sigs
        n = len(rows)
        if n < sigs.trend_min_steps:
            return None

        # Use the trailing window to estimate slope (less biased by warmup).
        win = rows[-min(sigs.trend_window, n):]
        corr   = _col(win, K_CORR)
        length = _col(win, K_LEN)
        # any "format_*" reward — pick xmlcount as the canonical proxy
        format_r = _col(win, "rewards/xmlcount_reward_func/mean")

        s_corr = _slope(corr)
        s_len  = _slope(length)
        s_fmt  = _slope(format_r)

        cur = float(np.nanmean(corr[-min(20, len(corr)):])) if np.isfinite(corr).any() else float("nan")
        steps_left = max(self.max_steps - n, 0)
        extr = cur + s_corr * steps_left

        will_reach = bool(extr >= sigs.target_corr)
        hacking = bool((s_corr <= 0) and (s_len < 0) and (s_fmt > 0))

        detail = (
            f"step {n}/{self.max_steps}: corr_now={cur:.3f}, "
            f"slope_corr={s_corr:+.2e}/step, slope_len={s_len:+.2f}/step, "
            f"slope_fmt={s_fmt:+.2e}/step, extrapolated_at_max={extr:.3f}, "
            f"target={sigs.target_corr}"
        )
        return TrendResult(
            n_steps_seen=n,
            corr_slope=s_corr,
            corr_current=cur,
            extrapolated=extr,
            ok_will_reach=will_reach,
            reward_hacking=hacking,
            details=detail,
        )
