"""CLI: live-tail a BT-GRPO log and abort on canary signal.

Usage:
    python scripts/canary_watcher.py \
        --log .logs/btgrpo-run20.log \
        --target-pid $TRAIN_PID \
        --refresh 2

If --target-pid is omitted, the watcher only prints to stderr; you can use it
as a passive observer alongside dashboard.py.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Make the repo importable when run as `python scripts/canary_watcher.py`
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dllm.pipelines.rl.canary import Watcher, Signatures


def _fmt_eval(ev: dict) -> str:
    a = ev["advisory"]
    sig_names = ["grad_blowup", "starved_signal", "fork_saturated",
                 "len_collapsing", "corr_dead_early", "corr_negative_slope"]
    lights = "".join("●" if a[k] else "○" for k in sig_names)
    feat = ev["features"]
    return (
        f"step {ev['step']:>5d}  [{lights}]  "
        f"grad_max={feat['grad_max']:.2f}  "
        f"fzs={feat['frac_zero_std_mean']:.2f}  "
        f"len_slope={feat['len_slope']:+.2f}  "
        f"corr_slope={feat['corr_slope']:+.4f}  "
        f"fired={a['n_fired']}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, type=Path)
    p.add_argument("--target-pid", type=int, default=None,
                   help="PID to send SIGINT on abort (default: print only)")
    p.add_argument("--refresh", type=float, default=2.0,
                   help="Seconds between log polls")
    p.add_argument("--once", action="store_true",
                   help="Replay the existing log once and exit (no follow)")
    args = p.parse_args()

    sigs = Signatures()
    w = Watcher(args.log, sigs=sigs, target_pid=args.target_pid)

    legend = "legend: signatures = grad/starved/fork/len/corrDead/corrDown"
    sys.stderr.write(legend + "\n")

    if args.once:
        st = w.replay()
        if st.last_eval:
            print(_fmt_eval(st.last_eval))
        if st.aborted:
            print(f"ABORTED: {st.abort_reason}")
            sys.exit(2)
        return

    last_print = 0
    for ev in w.follow(poll=args.refresh):
        # Don't spam: print every step but flush.
        print(_fmt_eval(ev), flush=True)
        if w.state.aborted:
            print(f"ABORTED: {w.state.abort_reason}", flush=True)
            sys.exit(2)


if __name__ == "__main__":
    main()
