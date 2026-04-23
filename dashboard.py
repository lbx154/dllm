#!/usr/bin/env python
"""Beautiful dark-themed live dashboard for BT-GRPO training.

Parses metrics directly from the training log, plots smoothed curves with
raw scatter, and writes a single auto-refreshing PNG.

Usage:
    python dashboard.py --log /root/dllm/.logs/btgrpo-run10.log [--once] [--refresh 30]
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

LOG_FILE = Path("/root/dllm/.logs/btgrpo-run10.log")
PNG_OUT = Path("/root/dllm/dashboard.png")
REFRESH = 30

DICT_RE = re.compile(r"\{[^{}]*'loss'[^{}]*\}")

# ------ theme ------
BG = "#0e1117"
FG = "#e6edf3"
GRID = "#30363d"
ACCENT = "#58a6ff"
MUTED = "#8b949e"

# (row, col, rowspan, colspan) grid is 4 rows x 4 cols
PANELS = [
    # (gridspec slice, title, metric_keys, ylabel)
    ("loss",      "Loss",                  ["loss"],                                          None),
    ("reward",    "Reward (total)",        ["reward"],                                        None),
    ("entropy",   "Policy Entropy",        ["entropy"],                                       None),
    ("gradnorm",  "Gradient Norm",         ["grad_norm"],                                     None),
    ("rewards",   "Reward Breakdown",      ["rewards/correctness_reward_func/mean",
                                            "rewards/soft_format_reward_func/mean",
                                            "rewards/xmlcount_reward_func/mean",
                                            "rewards/int_reward_func/mean"],                  None),
    ("lengths",   "Completion Length",     ["completions/length/mean",
                                            "completions/clipped_ratio"],                     None),
    ("btgrpo",    "BT-GRPO Signals",       ["btgrpo/divergent_frac",
                                            "frac_reward_zero_std"],                          None),
    ("forkfrac",  "Fork Frac (head)",      ["btgrpo/fork_frac",
                                            "btgrpo/fork_frac_mean",
                                            "btgrpo/fork_baseline"],                          None),
    ("forkrl",    "Fork Head REINFORCE",   ["btgrpo/fork_advantage",
                                            "btgrpo/fork_loss",
                                            "btgrpo/fork_sigma"],                             None),
    ("lr",        "Learning Rate",         ["learning_rate"],                                 None),
]

SERIES_COLORS = {
    "loss":                                      "#f97583",
    "reward":                                    "#7ee787",
    "entropy":                                   "#d2a8ff",
    "grad_norm":                                 "#ffa657",
    "learning_rate":                             "#79c0ff",
    "rewards/correctness_reward_func/mean":      "#7ee787",
    "rewards/soft_format_reward_func/mean":      "#79c0ff",
    "rewards/xmlcount_reward_func/mean":         "#ffa657",
    "rewards/int_reward_func/mean":              "#d2a8ff",
    "completions/length/mean":                   "#79c0ff",
    "completions/clipped_ratio":                 "#f97583",
    "btgrpo/divergent_frac":                     "#7ee787",
    "frac_reward_zero_std":                      "#f97583",
    "btgrpo/fork_frac":                          "#79c0ff",
    "btgrpo/fork_frac_mean":                     "#7ee787",
    "btgrpo/fork_baseline":                      "#ffa657",
    "btgrpo/fork_advantage":                     "#7ee787",
    "btgrpo/fork_loss":                          "#f97583",
    "btgrpo/fork_sigma":                         "#d2a8ff",
}

PRETTY = {
    "rewards/correctness_reward_func/mean": "correctness",
    "rewards/soft_format_reward_func/mean": "soft_format",
    "rewards/xmlcount_reward_func/mean":    "xmlcount",
    "rewards/int_reward_func/mean":         "int",
    "completions/length/mean":              "mean length",
    "completions/clipped_ratio":            "clipped ratio",
    "btgrpo/divergent_frac":                "divergent frac",
    "frac_reward_zero_std":                 "zero-std frac",
    "btgrpo/fork_frac":                     "sampled action",
    "btgrpo/fork_frac_mean":                "head μ",
    "btgrpo/fork_baseline":                 "EMA baseline",
    "btgrpo/fork_advantage":                "advantage",
    "btgrpo/fork_loss":                     "REINFORCE loss",
    "btgrpo/fork_sigma":                    "head σ",
    "grad_norm":                            "grad_norm",
    "learning_rate":                        "lr",
    "loss":                                 "loss",
    "reward":                               "reward",
    "entropy":                              "entropy",
}


def parse_log(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            m = DICT_RE.search(line)
            if not m:
                continue
            try:
                d = ast.literal_eval(m.group(0))
                if isinstance(d, dict):
                    rows.append(d)
            except (ValueError, SyntaxError):
                pass
    return rows


def gpu_snapshot():
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0:
            return []
        out = []
        for line in r.stdout.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            out.append(dict(idx=int(p[0]), util=int(p[1]),
                            mem_used=int(p[2]), mem_total=int(p[3]),
                            temp=int(p[4])))
        return out
    except Exception:
        return []


def ema(xs, alpha=0.15):
    if not xs:
        return xs
    out = [xs[0]]
    for x in xs[1:]:
        out.append(alpha * x + (1 - alpha) * out[-1])
    return out


def style_ax(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=11, fontweight="bold",
                 loc="left", pad=8)
    ax.grid(color=GRID, linestyle="-", linewidth=0.5, alpha=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(GRID)


def plot_panel(ax, title, rows, keys):
    style_ax(ax, title)
    if not rows:
        return False
    plotted = False
    for k in keys:
        pts = [(i + 1, r[k]) for i, r in enumerate(rows) if k in r]
        if not pts:
            continue
        xs, ys = zip(*pts)
        color = SERIES_COLORS.get(k, ACCENT)
        label = PRETTY.get(k, k.split("/")[-1])
        # raw scatter (faint) + EMA smooth (bold)
        ax.plot(xs, ys, color=color, alpha=0.22, lw=0.9, zorder=2)
        if len(ys) >= 3:
            ax.plot(xs, ema(list(ys), alpha=0.18), color=color,
                    lw=2.0, label=label, zorder=3)
        else:
            ax.plot(xs, ys, color=color, lw=2.0, label=label, zorder=3)
        plotted = True
    if plotted and len(keys) > 1:
        leg = ax.legend(loc="best", frameon=False, fontsize=8,
                        labelcolor=FG)
    ax.set_xlabel("step", color=MUTED, fontsize=8)
    return plotted


def draw_header(fig, rows, gpus):
    last = rows[-1] if rows else {}
    last_step = len(rows)

    # Header area
    ax = fig.add_axes([0.0, 0.945, 1.0, 0.05])
    ax.set_facecolor(BG)
    ax.axis("off")

    # Derive a human label from the log filename, e.g. "run10" from "btgrpo-run10.log"
    import re as _re
    m = _re.search(r"(run\d+)", Path(getattr(draw_header, "_log_path", "")).stem)
    label = m.group(1) if m else "run?"
    ax.text(0.015, 0.55, f"BT-GRPO  ·  {label}",
            color=FG, fontsize=18, fontweight="bold",
            va="center", ha="left", transform=ax.transAxes)
    subtitle = (f"LLaDA-8B-Instruct  ·  GSM8K  ·  "
                f"β=0  fork=0.5  ε=0.2  lr=1.5e-6  LoRA r=64")
    ax.text(0.015, 0.1, subtitle, color=MUTED, fontsize=9,
            va="center", ha="left", transform=ax.transAxes)

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    ax.text(0.985, 0.55, f"step  {last_step} / 8000",
            color=ACCENT, fontsize=16, fontweight="bold",
            va="center", ha="right", transform=ax.transAxes)
    ax.text(0.985, 0.1, now, color=MUTED, fontsize=9,
            va="center", ha="right", transform=ax.transAxes)


def draw_kpis(fig, rows):
    """Big-number strip directly below the header."""
    ax = fig.add_axes([0.0, 0.87, 1.0, 0.075])
    ax.set_facecolor(BG)
    ax.axis("off")

    if not rows:
        return

    last = rows[-1]
    # Moving averages (last 20 for stability)
    tail = rows[-20:] if len(rows) >= 20 else rows
    def avg(k):
        vs = [r[k] for r in tail if k in r]
        return sum(vs) / len(vs) if vs else None

    kpis = [
        ("reward",                                   "Reward",      "{:.3f}"),
        ("rewards/correctness_reward_func/mean",     "Correct",     "{:.3f}"),
        ("entropy",                                  "Entropy",     "{:.3f}"),
        ("loss",                                     "Loss",        "{:.4f}"),
        ("grad_norm",                                "Grad norm",   "{:.3f}"),
        ("btgrpo/fork_frac_mean",                    "Fork μ",      "{:.3f}"),
        ("btgrpo/divergent_frac",                    "Divergent",   "{:.3f}"),
    ]
    n = len(kpis)
    for i, (key, label, fmt) in enumerate(kpis):
        x0 = 0.015 + i * (0.97 / n)
        w = 0.97 / n - 0.01
        val = avg(key)
        # card background
        ax.add_patch(FancyBboxPatch(
            (x0, 0.05), w, 0.9,
            boxstyle="round,pad=0.0,rounding_size=0.02",
            linewidth=1, edgecolor=GRID, facecolor="#161b22",
            transform=ax.transAxes, zorder=1,
        ))
        ax.text(x0 + 0.012, 0.72, label, color=MUTED, fontsize=9,
                va="center", ha="left", transform=ax.transAxes)
        if val is None:
            s = "—"
            color = MUTED
        else:
            s = fmt.format(val)
            color = FG
        ax.text(x0 + 0.012, 0.30, s, color=color, fontsize=18,
                fontweight="bold", va="center", ha="left",
                transform=ax.transAxes, family="monospace")


def draw_gpu_strip(fig, gpus):
    """Compact GPU usage strip at the bottom."""
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.07])
    ax.set_facecolor(BG)
    ax.axis("off")
    if not gpus:
        ax.text(0.5, 0.5, "no GPU data", color=MUTED,
                ha="center", va="center", transform=ax.transAxes)
        return

    n = len(gpus)
    for i, g in enumerate(gpus):
        x0 = 0.015 + i * (0.97 / n)
        w = 0.97 / n - 0.008
        ax.add_patch(FancyBboxPatch(
            (x0, 0.15), w, 0.75,
            boxstyle="round,pad=0.0,rounding_size=0.015",
            linewidth=1, edgecolor=GRID, facecolor="#161b22",
            transform=ax.transAxes, zorder=1,
        ))
        # util bar
        util_frac = g["util"] / 100.0
        bar_w = (w - 0.02) * util_frac
        bar_color = "#7ee787" if g["util"] > 60 else ("#ffa657" if g["util"] > 20 else "#8b949e")
        ax.add_patch(FancyBboxPatch(
            (x0 + 0.01, 0.22), max(bar_w, 0.002), 0.12,
            boxstyle="round,pad=0.0,rounding_size=0.005",
            linewidth=0, facecolor=bar_color,
            transform=ax.transAxes, zorder=2,
        ))
        ax.text(x0 + 0.012, 0.70,
                f"GPU {g['idx']}", color=FG, fontsize=9,
                fontweight="bold", transform=ax.transAxes)
        ax.text(x0 + w - 0.012, 0.70,
                f"{g['util']}%  {g['temp']}°C",
                color=MUTED, fontsize=8, ha="right",
                transform=ax.transAxes)
        ax.text(x0 + 0.012, 0.43,
                f"{g['mem_used']/1024:.1f} / {g['mem_total']/1024:.0f} GB",
                color=MUTED, fontsize=8, transform=ax.transAxes,
                family="monospace")


def render(rows, gpus, png_path: Path):
    fig = plt.figure(figsize=(18, 11), facecolor=BG)

    draw_header(fig, rows, gpus)
    draw_kpis(fig, rows)
    draw_gpu_strip(fig, gpus)

    # plots region: rows 0.09..0.86
    if not rows:
        ax = fig.add_axes([0.05, 0.3, 0.9, 0.5])
        ax.set_facecolor(BG)
        ax.axis("off")
        ax.text(0.5, 0.5, "Waiting for first training step…",
                ha="center", va="center", color=MUTED, fontsize=18,
                transform=ax.transAxes)
        fig.savefig(png_path, dpi=110, facecolor=BG)
        plt.close(fig)
        return

    n = len(PANELS)
    cols = 4
    rws = (n + cols - 1) // cols  # 2

    # Grid inside [0.03, 0.09, 0.94, 0.77] — normalized figure coords
    gx0, gy0, gw, gh = 0.035, 0.095, 0.935, 0.755
    hgap, vgap = 0.035, 0.085
    cell_w = (gw - hgap * (cols - 1)) / cols
    cell_h = (gh - vgap * (rws - 1)) / rws

    for idx, (_, title, keys, _) in enumerate(PANELS):
        r = idx // cols
        c = idx % cols
        x = gx0 + c * (cell_w + hgap)
        # rows go top-to-bottom
        y = gy0 + (rws - 1 - r) * (cell_h + vgap)
        ax = fig.add_axes([x, y, cell_w, cell_h])
        plot_panel(ax, title, rows, keys)

    fig.savefig(png_path, dpi=110, facecolor=BG)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default=str(LOG_FILE))
    parser.add_argument("--refresh", type=int, default=REFRESH)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    log_path = Path(args.log)
    print(f"dashboard: {log_path}  ->  {PNG_OUT}  (refresh {args.refresh}s)")
    draw_header._log_path = str(log_path)

    last_n = -1
    while True:
        rows = parse_log(log_path)
        gpus = gpu_snapshot()
        render(rows, gpus, PNG_OUT)

        if len(rows) != last_n:
            last_n = len(rows)
            last = rows[-1] if rows else {}
            short = " | ".join(
                f"{k}={last[k]:.3g}"
                for k in ("loss", "reward", "entropy", "grad_norm")
                if k in last
            )
            print(f"[{time.strftime('%H:%M:%S')}] steps={len(rows)}  {short}")

        if args.once:
            break
        sys.stdout.flush()
        time.sleep(args.refresh)


if __name__ == "__main__":
    main()
