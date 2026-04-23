#!/usr/bin/env python
"""Single-node BT-GRPO training monitor.

Reads trainer_state.json from the output directory, plots key metrics,
and saves an auto-refreshing PNG dashboard.

Usage:
    python monitor.py [--output_dir /root/dllm/.models/llada-btgrpo-run4] [--refresh 15]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTPUT_DIR = Path("/root/dllm/.models/llada-btgrpo-run4")
PNG_OUT = Path("/root/dllm/dashboard.png")
REFRESH = 15

METRIC_GROUPS = {
    "Loss & Reward": ["loss", "reward", "reward_std"],
    "Reward Breakdown": [
        "rewards/correctness_reward_func/mean",
        "rewards/soft_format_reward_func/mean",
        "rewards/xmlcount_reward_func/mean",
        "rewards/int_reward_func/mean",
    ],
    "KL & Clip": ["kl", "clip_ratio"],
    "Completion": ["completions/length/mean", "btgrpo/divergent_frac"],
    "Training": ["grad_norm", "learning_rate"],
}

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def read_trainer_state(output_dir: Path) -> list[dict]:
    state_file = output_dir / "trainer_state.json"
    if not state_file.exists():
        return []
    try:
        with state_file.open() as f:
            state = json.load(f)
        return state.get("log_history", [])
    except (json.JSONDecodeError, KeyError):
        return []


def gpu_snapshot() -> list[dict]:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0:
            return []
        gpus = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            gpus.append({
                "idx": int(parts[0]),
                "util": int(parts[1]),
                "mem_used": int(parts[2]),
                "mem_total": int(parts[3]),
                "temp": int(parts[4]),
            })
        return gpus
    except Exception:
        return []


def plot_dashboard(logs: list[dict], gpus: list[dict], png_path: Path):
    if not logs:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "⏳ Waiting for first training step...\n"
                "(model loading & first generation in progress)",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        fig.savefig(png_path, dpi=100, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    # Collect all available metrics
    available_groups = {}
    for group_name, keys in METRIC_GROUPS.items():
        present = [k for k in keys if any(k in log for log in logs)]
        if present:
            available_groups[group_name] = present

    n_panels = len(available_groups) + 1  # +1 for GPU panel
    cols = 3
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows))
    axes = axes.flatten()

    panel_idx = 0
    for group_name, keys in available_groups.items():
        ax = axes[panel_idx]
        for i, key in enumerate(keys):
            steps, vals = [], []
            for log in logs:
                if key in log and "step" in log:
                    steps.append(log["step"])
                    vals.append(log[key])
            if steps:
                label = key.split("/")[-1] if "/" in key else key
                ax.plot(steps, vals, lw=1.3, color=COLORS[i % len(COLORS)],
                        label=label, alpha=0.85)
        ax.set_title(group_name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
        ax.set_xlabel("step", fontsize=9)
        ax.tick_params(labelsize=8)
        if group_name == "Training" and "learning_rate" in keys:
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))
        panel_idx += 1

    # GPU panel
    ax = axes[panel_idx]
    ax.set_title("GPU Status", fontsize=11, fontweight="bold")
    ax.axis("off")

    last_step = max((log.get("step", 0) for log in logs), default=0)
    last_log = [l for l in logs if l.get("step") == last_step]
    last = last_log[-1] if last_log else logs[-1]

    lines = [f"Step: {last_step}  |  Time: {time.strftime('%H:%M:%S')}", ""]

    if gpus:
        lines.append("GPU  Util%  Mem(GB)   Temp")
        lines.append("─" * 32)
        for g in gpus:
            lines.append(
                f" [{g['idx']}]  {g['util']:3d}%   "
                f"{g['mem_used']/1024:5.1f}/{g['mem_total']/1024:.0f}  "
                f"{g['temp']}°C"
            )
    lines.append("")
    lines.append("Latest metrics:")
    lines.append("─" * 32)
    for key in ["loss", "reward", "reward_std", "kl", "grad_norm",
                "completions/length/mean", "btgrpo/divergent_frac"]:
        if key in last:
            short = key.split("/")[-1] if "/" in key else key
            lines.append(f"  {short:18s}: {last[key]:.4g}")

    ax.text(0.02, 0.98, "\n".join(lines), fontfamily="monospace",
            va="top", ha="left", fontsize=9, transform=ax.transAxes)
    panel_idx += 1

    for i in range(panel_idx, len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        f"BT-GRPO Training Dashboard  —  {last_step} steps",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--refresh", type=int, default=REFRESH)
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"📊 Monitor: watching {output_dir}/trainer_state.json")
    print(f"   Dashboard: {PNG_OUT}")
    print(f"   Refresh: {args.refresh}s")

    last_step = -1
    while True:
        logs = read_trainer_state(output_dir)
        gpus = gpu_snapshot()
        plot_dashboard(logs, gpus, PNG_OUT)

        cur_step = max((log.get("step", 0) for log in logs), default=0) if logs else 0
        if cur_step != last_step:
            last_step = cur_step
            last = logs[-1] if logs else {}
            short = " | ".join(
                f"{k}={last[k]:.3g}" for k in ("loss", "reward", "kl", "grad_norm")
                if k in last
            )
            print(f"[{time.strftime('%H:%M:%S')}] step={cur_step}  {short}")

        if args.once:
            break

        sys.stdout.flush()
        time.sleep(args.refresh)


if __name__ == "__main__":
    main()
