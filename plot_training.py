#!/usr/bin/env python
"""
Plot training loss from train_log.jsonl.
Run at any time — works on a live or finished log.

Usage:
    /opt/anaconda3/envs/lerobot/bin/python plot_training.py
    /opt/anaconda3/envs/lerobot/bin/python plot_training.py --suite libero_object
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

parser = argparse.ArgumentParser()
parser.add_argument("--suite", default="libero_spatial")
parser.add_argument("--out",   default=None, help="output PNG path")
args = parser.parse_args()

LOG_PATH = f"./checkpoints/{args.suite}/train_log.jsonl"
OUT_PATH = args.out or f"./checkpoints/{args.suite}/loss_curve.png"

if not os.path.exists(LOG_PATH):
    print(f"No log found at {LOG_PATH}. Start training first.")
    raise SystemExit(1)

# ── Load log ──────────────────────────────────────────────────────────────────
steps, losses, lrs = [], [], []
with open(LOG_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        steps.append(d["step"])
        losses.append(d["loss"])
        lrs.append(d.get("lr", float("nan")))

if not steps:
    print("Log is empty — training may not have started yet.")
    raise SystemExit(1)

steps  = np.array(steps)
losses = np.array(losses)
lrs    = np.array(lrs)

# ── Smoothed loss (exponential moving average) ────────────────────────────────
def ema(x, alpha=0.3):
    out = np.zeros_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out

loss_smooth = ema(losses, alpha=0.2)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]})
fig.suptitle(f"SmolVLA training — {args.suite}  ({steps[-1]+1} steps logged)", fontsize=13)

# Loss panel
ax1.plot(steps, losses,      color="#aad4f5", linewidth=0.8, alpha=0.6, label="loss (raw)")
ax1.plot(steps, loss_smooth, color="#1f77b4", linewidth=2.0,             label="loss (EMA-0.2)")
ax1.set_ylabel("Flow-matching loss")
ax1.set_xlabel("Step")
ax1.legend(loc="upper right")
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax1.grid(True, alpha=0.3)

# Annotate min loss
min_idx = np.argmin(loss_smooth)
ax1.scatter(steps[min_idx], loss_smooth[min_idx], color="red", zorder=5, s=60)
ax1.annotate(
    f"min={loss_smooth[min_idx]:.3f}\n@ step {steps[min_idx]}",
    xy=(steps[min_idx], loss_smooth[min_idx]),
    xytext=(10, 20), textcoords="offset points",
    fontsize=8, color="red",
    arrowprops=dict(arrowstyle="->", color="red", lw=1),
)

# Show current / final loss
ax1.axhline(losses[-1], color="orange", linestyle="--", linewidth=1, alpha=0.7,
            label=f"latest={losses[-1]:.3f}")
ax1.legend(loc="upper right")

# Learning-rate panel
ax2.semilogy(steps, lrs, color="#2ca02c", linewidth=1.5)
ax2.set_ylabel("LR")
ax2.set_xlabel("Step")
ax2.grid(True, alpha=0.3)

# Stats box
total_steps_planned = 2000  # matches train script default
pct_done = (steps[-1] + 1) / total_steps_planned * 100
elapsed_min = None
try:
    with open(LOG_PATH) as f:
        lines = [l for l in f if l.strip()]
    last = json.loads(lines[-1])
    first = json.loads(lines[0])
    elapsed_min = (last["elapsed"] - first.get("elapsed", 0)) / 60
except Exception:
    pass

stats_lines = [
    f"Steps logged : {len(steps)}",
    f"Latest loss  : {losses[-1]:.4f}",
    f"Min loss     : {losses.min():.4f} @ step {steps[np.argmin(losses)]}",
    f"Progress     : {pct_done:.1f}% of {total_steps_planned}",
]
if elapsed_min is not None:
    stats_lines.append(f"Elapsed      : {elapsed_min:.1f} min")

ax1.text(
    0.02, 0.05, "\n".join(stats_lines),
    transform=ax1.transAxes, fontsize=8.5,
    verticalalignment="bottom",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=130)
print(f"Saved: {OUT_PATH}")
print(f"Steps: {len(steps)}  |  Latest loss: {losses[-1]:.4f}  |  Min: {losses.min():.4f}")
