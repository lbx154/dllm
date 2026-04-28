"""Replay all historical runs through the canary watcher and verify that
the signatures fire on the runs we know failed.

Run with:
    cd /root/dllm && python -m pytest dllm/pipelines/rl/canary/tests/test_replay.py -q

Or directly:
    cd /root/dllm && python dllm/pipelines/rl/canary/tests/test_replay.py
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]      # /root/dllm
sys.path.insert(0, str(REPO))

import pandas as pd
from dllm.pipelines.rl.canary import Watcher, Signatures
from dllm.pipelines.rl.canary.trend import TrendExtrapolator

# Use the per-step CSV produced by extract_run_features.py
SESSION_FILES = Path("/root/.copilot/session-state/e0760e9f-dd31-44a0-9a31-10dead3cff60/files")
PER_STEP_CSV  = SESSION_FILES / "runs_per_step.csv"


def load_rows_by_run() -> dict[str, list[dict]]:
    df = pd.read_csv(PER_STEP_CSV, low_memory=False)
    out: dict[str, list[dict]] = {}
    for run, sub in df.groupby("run"):
        sub = sub.sort_values("step")
        rows = []
        for _, r in sub.iterrows():
            rows.append({k: v for k, v in r.items()
                         if pd.notna(v) and k not in ("run",)})
        out[run] = rows
    return out


# Expected signature triggers within the first 50 steps.
# These come from session-state/.../files/canary_rl.md §1 (rule-based table).
EXPECTED_FIRES_FIRST_50 = {
    "run4":  {"corr_negative_slope"},
    "run5":  set(),
    "run6":  {"corr_negative_slope"},
    "run7":  set(),
    "run8":  {"len_collapsing", "corr_negative_slope"},
    "run9":  {"len_collapsing", "corr_negative_slope"},
    "run10": set(),
    "run11": {"corr_negative_slope"},
    "run12": {"grad_blowup", "starved_signal", "len_collapsing", "corr_negative_slope"},
    "run13": {"grad_blowup", "starved_signal", "fork_saturated", "len_collapsing"},
    "run14": {"starved_signal"},
    "run15": {"starved_signal"},
    "run16": {"starved_signal", "fork_saturated"},
    "run17": {"starved_signal"},
    "run18": {"starved_signal", "len_collapsing", "corr_negative_slope"},
    "run19": {"starved_signal"},
}


def first_n_eval(rows, n=50, sigs=None):
    """Compute the signatures over the first `n` rows of a run, IGNORING any
    Tier-1 abort that would normally short-circuit the watcher. Used by the
    recall test below: we want to know "would the signatures fire if we let
    the run go to step n?" not "what was the last evaluation before abort?"."""
    from dllm.pipelines.rl.canary.signatures import evaluate_signatures
    sigs = sigs or Signatures()
    window = rows[:n]
    return evaluate_signatures(window, sigs), None


def test_signatures_match_history():
    by_run = load_rows_by_run()
    failures = []
    for run, expected in EXPECTED_FIRES_FIRST_50.items():
        if run not in by_run:
            failures.append(f"  missing logs: {run}")
            continue
        ev, _ = first_n_eval(by_run[run], n=50)
        if ev is None:
            failures.append(f"  {run}: no steps parsed")
            continue
        fired = {k for k, v in ev["advisory"].items() if k != "n_fired" and v}
        # We require *recall*: every expected signature must fire. We allow
        # extra signatures since thresholds are tuned for sensitivity.
        missing = expected - fired
        if missing:
            failures.append(f"  {run}: missing fires {missing} (got {fired or '∅'})")
    assert not failures, "Signature mismatches:\n" + "\n".join(failures)


def test_tier1_aborts_on_grad_blowup():
    # NOTE: default Signatures thresholds are tuned for BT-GRPO+dLLM where
    # pre-clip grad_norm of 1e8–1e11 is normal (k3 KL on mask tokens). The
    # historical run12/run13 used the legacy AR trainer with much smaller
    # natural grad scale, so we override to the legacy thresholds here.
    from dllm.pipelines.rl.canary.signatures import Signatures
    legacy = Signatures(
        abort_grad_norm=1.0e4,
        abort_grad_blowup_thresh=1.0e3,
        abort_grad_blowup_frac=0.05,
        abort_loss_jump_factor=100.0,
        abort_loss_jump_floor=1.0,
    )
    by_run = load_rows_by_run()
    for run in ("run12", "run13"):
        w = Watcher(path="/dev/null", sigs=legacy)
        w.replay_rows(by_run[run])
        assert w.state.aborted, f"{run}: expected Tier-1 abort but did not"
        # Either a direct grad signal or the loss-jump that accompanies it
        # is acceptable — both are correct Tier-1 fires.
        assert w.state.abort_reason in {
            "grad_norm_step", "grad_blowup_window", "loss_jump"
        }, f"{run}: unexpected reason {w.state.abort_reason}"


def test_no_false_abort_on_clean_runs():
    """run4/5/6/7/8/9/10/11 had no grad explosion; Tier-1 must NOT abort them."""
    by_run = load_rows_by_run()
    clean = ["run4", "run5", "run6", "run7", "run8", "run9", "run10", "run11"]
    for run in clean:
        w = Watcher(path="/dev/null")
        w.replay_rows(by_run[run])
        assert not w.state.aborted, f"{run}: false Tier-1 abort ({w.state.abort_reason})"


def test_trend_catches_run17_collapse():
    """run17 is a 4400-step monotonic decay; trend extrapolator should see
    a non-positive slope and report extrapolated < target."""
    by_run = load_rows_by_run()
    rows = by_run["run17"]
    tr = TrendExtrapolator(max_steps=8000)
    res = tr.evaluate(rows)
    assert res is not None
    # Slope must not be strongly positive; with ~corr_now small extrapolation
    # cannot reach target_corr=1.0.
    assert not res.ok_will_reach, f"run17 trend says will reach target: {res.details}"


if __name__ == "__main__":
    print("Running canary replay tests on historical runs...")
    test_signatures_match_history()
    print("  ok signatures_match_history")
    test_tier1_aborts_on_grad_blowup()
    print("  ok tier1_aborts_on_grad_blowup")
    test_no_false_abort_on_clean_runs()
    print("  ok no_false_abort_on_clean_runs")
    test_trend_catches_run17_collapse()
    print("  ok trend_catches_run17_collapse")
    print("\nAll canary tests passed.")
