#!/usr/bin/env python
"""Live monitor for BT-GRPO 4-node training.

Tails the latest btgrpo-*-node0.log, parses trl/transformers logging_steps
lines (json-like dicts emitted at every step), and writes an updating PNG
dashboard with the key metrics every refresh interval.

Metrics tracked (whichever are present in the log):
    loss, reward, reward_std, kl, completion_length, grad_norm,
    learning_rate, clip_ratio, epoch

Output:
    /home/aiscuser/dllm/.logs/monitor_dashboard.png   (overwritten every tick)
    /home/aiscuser/dllm/.logs/monitor_metrics.csv     (full history, appended)

Also prints per-node GPU utilization + memory to stdout.
"""
from __future__ import annotations

import csv
import glob
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = Path("/home/aiscuser/dllm/.logs")
PNG_OUT = LOG_DIR / "monitor_dashboard.png"
CSV_OUT = LOG_DIR / "monitor_metrics.csv"
REFRESH = 20  # seconds

# matches "{'loss': 1.234, 'reward': 0.5, ...}" as emitted by Trainer.log
DICT_RE = re.compile(r"\{[^{}]*'loss'[^{}]*\}")
NUM_RE = re.compile(r"'([a-zA-Z_][\w/]*)'\s*:\s*([-+0-9.eE]+)")

NODES = ["node-0", "node-1", "node-2", "node-3"]
TRACK_KEYS = [
    "reward",
    "rewards/correctness_reward_func/mean",
    "rewards/soft_format_reward_func/mean",
    "rewards/xmlcount_reward_func/mean",
    "rewards/int_reward_func/mean",
    "reward_std",
    "btgrpo/divergent_frac",
    "completions/length/mean",
    "kl",
    "grad_norm",
    "loss",
    "learning_rate",
]


def latest_node0_log() -> Path | None:
    files = sorted(LOG_DIR.glob("btgrpo-*-node0.log"), key=os.path.getmtime)
    return files[-1] if files else None


def parse_log(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            m = DICT_RE.search(line)
            if not m:
                continue
            d = {}
            for k, v in NUM_RE.findall(m.group(0)):
                try:
                    d[k] = float(v)
                except ValueError:
                    pass
            if d:
                rows.append(d)
    return rows


def gpu_snapshot():
    out = {}
    for n in NODES:
        try:
            r = subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", n,
                 "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=8,
            )
            if r.returncode == 0:
                utils, mems = [], []
                for ln in r.stdout.strip().splitlines():
                    u, m = ln.split(",")
                    utils.append(int(u.strip()))
                    mems.append(int(m.strip()))
                out[n] = (sum(utils) / len(utils), sum(mems) / len(mems))
            else:
                out[n] = (None, None)
        except Exception:
            out[n] = (None, None)
    return out


def plot_dashboard(rows, gpus):
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "waiting for first training log line...",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        fig.savefig(PNG_OUT, dpi=90, bbox_inches="tight")
        plt.close(fig)
        return

    series = defaultdict(list)
    steps = list(range(len(rows)))
    for i, r in enumerate(rows):
        for k in TRACK_KEYS:
            if k in r:
                series[k].append((i, r[k]))

    # Grid: 8 metrics + GPU panel
    keys = [k for k in TRACK_KEYS if series[k]]
    n = len(keys) + 1
    cols = 3
    ro = (n + cols - 1) // cols
    fig, axes = plt.subplots(ro, cols, figsize=(5 * cols, 3.2 * ro))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, key in zip(axes, keys):
        xs, ys = zip(*series[key])
        ax.plot(xs, ys, lw=1.2)
        ax.set_title(key)
        ax.grid(alpha=0.3)
        ax.set_xlabel("step")

    # GPU usage panel
    ax = axes[len(keys)]
    ax.set_title(f"GPU util / mem (MiB) @ {time.strftime('%H:%M:%S')}")
    ax.axis("off")
    lines = [f"latest step: {len(rows)}"]
    for n_, (u, m) in gpus.items():
        if u is None:
            lines.append(f"  {n_}: ssh fail")
        else:
            lines.append(f"  {n_}: util={u:5.1f}%   mem={m:6.0f} MiB/gpu")
    last = rows[-1]
    lines.append("")
    lines.append("latest metrics:")
    for k in TRACK_KEYS:
        if k in last:
            lines.append(f"  {k}: {last[k]:.4g}")
    ax.text(0.02, 0.98, "\n".join(lines), fontfamily="monospace",
            va="top", ha="left", fontsize=10, transform=ax.transAxes)

    for ax in axes[len(keys) + 1:]:
        ax.axis("off")

    fig.suptitle(f"BT-GRPO training  —  steps logged: {len(rows)}",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(PNG_OUT, dpi=90, bbox_inches="tight")
    plt.close(fig)


def write_csv(rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with CSV_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    print(f"monitor: watching {LOG_DIR}, refresh={REFRESH}s, out={PNG_OUT}")
    last_n = -1
    while True:
        log = latest_node0_log()
        rows = parse_log(log) if log else []
        gpus = gpu_snapshot()
        plot_dashboard(rows, gpus)
        write_csv(rows)
        if len(rows) != last_n:
            last_n = len(rows)
            last = rows[-1] if rows else {}
            short = " ".join(f"{k}={last[k]:.3g}" for k in ("loss", "reward", "kl") if k in last)
            print(f"[{time.strftime('%H:%M:%S')}] log={log.name if log else 'None'}  "
                  f"steps={len(rows)}  {short}")
        sys.stdout.flush()
        time.sleep(REFRESH)


if __name__ == "__main__":
    main()
