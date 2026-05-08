#!/usr/bin/env python
"""
Comparison evaluation plot for SmolVLA LIBERO fine-tuning.

Reads eval_output/openloop_{suite}_{model}.npz for any available models and
produces a multi-panel figure comparing them side-by-side.

Usage:
    /opt/anaconda3/envs/lerobot/bin/python plot_eval.py
    /opt/anaconda3/envs/lerobot/bin/python plot_eval.py --suite libero_object
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
parser.add_argument("--out",   default=None)
args = parser.parse_args()

SUITE    = args.suite
EVAL_DIR = "./eval_output"
OUT_PATH = args.out or f"./checkpoints/{SUITE}/eval_comparison.png"

# Models to compare, in display order
MODEL_SPECS = [
    ("random",         "Random weights",        "#d62728", "//"),
    ("trained",        "Trained (no VLM)",       "#ff7f0e", "\\\\"),
    ("trained_vlm",    "Trained (pretrained VLM)","#2ca02c", ""),
]

ACTION_LABELS = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]

# ── Load evaluation results ───────────────────────────────────────────────────

def load_eval(model_tag: str) -> dict | None:
    path = os.path.join(EVAL_DIR, f"openloop_{SUITE}_{model_tag}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return {
        "tasks": list(d["tasks"]),
        "mae":   d["mae"],
        "l2":    d["l2"],
        "grip":  d["grip"] * 100,  # → percentage
    }

results = {}
for tag, label, color, hatch in MODEL_SPECS:
    r = load_eval(tag)
    if r is not None:
        results[tag] = (r, label, color, hatch)

if not results:
    print("No eval results found. Run eval_openloop.py with MODEL=random/trained/trained_vlm first.")
    raise SystemExit(1)

# ── Load training logs ────────────────────────────────────────────────────────

def load_log(log_path: str):
    steps, losses = [], []
    if not os.path.exists(log_path):
        return np.array([]), np.array([])
    with open(log_path) as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                steps.append(d["step"])
                losses.append(d["loss"])
            except Exception:
                pass
    return np.array(steps), np.array(losses)

def ema(x, alpha=0.2):
    out = np.zeros_like(x)
    if len(x) == 0:
        return out
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out

steps_noVLM, losses_noVLM = load_log(f"./checkpoints/{SUITE}/train_log.jsonl")
steps_VLM,   losses_VLM   = load_log(f"./checkpoints/{SUITE}_vlm/train_log.jsonl")

# ── Figure layout ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(17, 13))
fig.suptitle(
    f"SmolVLA on LIBERO — {SUITE.replace('_', ' ').title()}\n"
    f"Random baseline  vs  Trained (random VLM)  vs  Trained (pretrained VLM)",
    fontsize=13, fontweight="bold", y=0.99
)

gs = fig.add_gridspec(3, 3, hspace=0.52, wspace=0.35,
                      left=0.07, right=0.97, top=0.93, bottom=0.06)

ax_loss   = fig.add_subplot(gs[0, :])   # full-width training curves
ax_mae    = fig.add_subplot(gs[1, :2])  # MAE per task (wide)
ax_sum    = fig.add_subplot(gs[1, 2])   # overall summary stats
ax_l2     = fig.add_subplot(gs[2, 0])
ax_grip   = fig.add_subplot(gs[2, 1])
ax_dim    = fig.add_subplot(gs[2, 2])

# ── 1. Training loss curves ───────────────────────────────────────────────────
if len(losses_noVLM) > 0:
    ax_loss.plot(steps_noVLM, ema(losses_noVLM), color="#ff7f0e", lw=2,
                 label="Trained — random VLM")
if len(losses_VLM) > 0:
    ax_loss.plot(steps_VLM, ema(losses_VLM), color="#2ca02c", lw=2,
                 label="Trained — pretrained VLM")

ax_loss.set_title("Training Loss (flow-matching, EMA-0.2)", fontsize=11)
ax_loss.set_xlabel("Step")
ax_loss.set_ylabel("Loss")
ax_loss.legend(fontsize=9)
ax_loss.grid(True, alpha=0.3)
ax_loss.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
if len(losses_noVLM) > 0 or len(losses_VLM) > 0:
    ax_loss.axvspan(0, 100, alpha=0.07, color="gold", label="warmup (100 steps)")

# ── Helper ────────────────────────────────────────────────────────────────────

def short_task(name: str, maxlen: int = 28) -> str:
    name = name.replace("pick up the black bowl ", "").replace("pick up the ", "")
    return name[:maxlen]

available = [(tag, r, label, color, hatch)
             for tag, (r, label, color, hatch) in results.items()]

tasks = available[0][1]["tasks"]
tasks_short = [short_task(t) for t in tasks]
n = len(tasks_short)
x = np.arange(n)
bar_w = 0.8 / max(len(available), 1)

# ── 2. MAE per task ───────────────────────────────────────────────────────────
for i, (tag, r, label, color, hatch) in enumerate(available):
    offset = (i - len(available)/2 + 0.5) * bar_w
    ax_mae.bar(x + offset, r["mae"], bar_w, color=color, hatch=hatch,
               alpha=0.82, label=f"{label}  (mean={r['mae'].mean():.3f})",
               edgecolor="white", linewidth=0.4)

ax_mae.set_xticks(x)
ax_mae.set_xticklabels(tasks_short, rotation=50, ha="right", fontsize=7.5)
ax_mae.set_ylabel("Mean Absolute Error ↓")
ax_mae.set_title("MAE per Task (open-loop prediction)", fontsize=11)
ax_mae.legend(fontsize=8.5, loc="upper right")
ax_mae.grid(True, axis="y", alpha=0.3)

# ── 3. Overall summary table ──────────────────────────────────────────────────
lines = []
for tag, r, label, color, hatch in available:
    lines.append(f"{'—'*26}")
    lines.append(label)
    lines.append(f"  MAE  : {r['mae'].mean():.3f}")
    lines.append(f"  L2   : {r['l2'].mean():.3f}")
    lines.append(f"  Grip : {r['grip'].mean():.1f}%")

# Improvement vs random if both available
if "random" in results and "trained_vlm" in results:
    r_base = results["random"][0]
    r_vlm  = results["trained_vlm"][0]
    imp = (r_base["mae"].mean() - r_vlm["mae"].mean()) / r_base["mae"].mean() * 100
    lines.append(f"{'—'*26}")
    lines.append(f"VLM MAE improv:")
    lines.append(f"  {imp:.1f}% vs random")
elif "random" in results and "trained" in results:
    r_base = results["random"][0]
    r_tr   = results["trained"][0]
    imp = (r_base["mae"].mean() - r_tr["mae"].mean()) / r_base["mae"].mean() * 100
    lines.append(f"{'—'*26}")
    lines.append(f"Trained improv:")
    lines.append(f"  {imp:.1f}% vs random")

ax_sum.text(0.05, 0.95, "\n".join(lines),
            ha="left", va="top", transform=ax_sum.transAxes,
            fontsize=8.5, fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))
ax_sum.set_title("Overall Summary", fontsize=11)
ax_sum.axis("off")

# ── 4. L2 distance per task ───────────────────────────────────────────────────
for i, (tag, r, label, color, hatch) in enumerate(available):
    offset = (i - len(available)/2 + 0.5) * bar_w
    ax_l2.bar(x + offset, r["l2"], bar_w, color=color, hatch=hatch,
              alpha=0.82, label=label, edgecolor="white", linewidth=0.4)

ax_l2.set_xticks(x)
ax_l2.set_xticklabels(tasks_short, rotation=50, ha="right", fontsize=7.5)
ax_l2.set_ylabel("L2 Distance ↓")
ax_l2.set_title("L2 Distance per Task", fontsize=11)
ax_l2.legend(fontsize=7.5)
ax_l2.grid(True, axis="y", alpha=0.3)

# ── 5. Gripper accuracy per task ──────────────────────────────────────────────
for i, (tag, r, label, color, hatch) in enumerate(available):
    offset = (i - len(available)/2 + 0.5) * bar_w
    ax_grip.bar(x + offset, r["grip"], bar_w, color=color, hatch=hatch,
                alpha=0.82, label=label, edgecolor="white", linewidth=0.4)

ax_grip.set_xticks(x)
ax_grip.set_xticklabels(tasks_short, rotation=50, ha="right", fontsize=7.5)
ax_grip.set_ylabel("Gripper Accuracy % ↑")
ax_grip.set_title("Gripper Sign Accuracy", fontsize=11)
ax_grip.set_ylim(0, 115)
ax_grip.legend(fontsize=7.5)
ax_grip.grid(True, axis="y", alpha=0.3)
ax_grip.axhline(100, color="gray", ls=":", lw=1, alpha=0.5)

# ── 6. Per-dimension aggregate MAE ───────────────────────────────────────────
# Use per-dim data saved by eval script if present; otherwise skip
per_dim_data = []
for tag, r, label, color, hatch in available:
    path = os.path.join(EVAL_DIR, f"openloop_{SUITE}_{tag}_perdim.npz")
    if os.path.exists(path):
        pd = np.load(path)["mae_per_dim"]
        per_dim_data.append((tag, pd, label, color, hatch))

if per_dim_data:
    xi = np.arange(len(ACTION_LABELS))
    dw = 0.8 / max(len(per_dim_data), 1)
    for i, (tag, pd, label, color, hatch) in enumerate(per_dim_data):
        offset = (i - len(per_dim_data)/2 + 0.5) * dw
        ax_dim.bar(xi + offset, pd, dw, color=color, hatch=hatch,
                   alpha=0.82, label=label, edgecolor="white", linewidth=0.4)
    ax_dim.set_xticks(xi)
    ax_dim.set_xticklabels(ACTION_LABELS, rotation=25, ha="right", fontsize=9)
    ax_dim.set_ylabel("MAE ↓")
    ax_dim.set_title("Per-Action-Dim MAE", fontsize=11)
    ax_dim.legend(fontsize=7.5)
    ax_dim.grid(True, axis="y", alpha=0.3)
else:
    # Show a simple bar chart of overall MAE
    bar_labels = [label for _, _, label, _, _ in available]
    bar_vals   = [r["mae"].mean() for _, r, _, _, _ in available]
    bar_colors = [color for _, _, _, color, _ in available]
    ax_dim.bar(bar_labels, bar_vals, color=bar_colors, alpha=0.85, edgecolor="white")
    for i, v in enumerate(bar_vals):
        ax_dim.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_dim.set_ylabel("Overall MAE ↓")
    ax_dim.set_title("Overall MAE Comparison", fontsize=11)
    ax_dim.set_xticklabels(bar_labels, rotation=15, ha="right", fontsize=8.5)
    ax_dim.grid(True, axis="y", alpha=0.3)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")

print("\n=== Evaluation Comparison ===")
for tag, r, label, color, hatch in available:
    print(f"  {label:<35}  MAE={r['mae'].mean():.3f}  L2={r['l2'].mean():.3f}  Grip={r['grip'].mean():.1f}%")
