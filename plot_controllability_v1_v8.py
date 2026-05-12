#!/usr/bin/env python
"""
Comprehensive controllability progression — all 8 variants (Stage 3 → Stage 6/v8).

Each variant differs by a small set of design decisions; the figure isolates the
contribution of each change.

  Variant   Stage  Injection   Encoder           Steps  Bal   Aux  Pass
  v1        3      additive    EEGNet 40.7%      1000   ✗     ✗    1/4
  v2        5a     token       EEGNet 40.7%      1000   ✗     ✗    2/4
  v3        5b     token       EEGNet 40.7%      2000   ✓     ✗    2/4
  v4        5c     token       EEGNet 51.6%      3000   ✓     ✗    3/4 ★
  v5        6      token       ATCNet 57.7%      3000   ✓     0.2  1/4
  v6        6      token       ATCNet 57.7%      3000   ✓     ✗    2/4
  v7        7      token       EEGNet 51.6%      5000   ✓     ✗    2/4
  v8        8      token       ATCNet 57.7% (scratch) 5000 ✓ ✗     2/4
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = "./eval_output/controllability_v1_v8.png"

# (label, n_pass, [pass-mark per class L/R/F/Fwd])
variants = [
    ("v1 · Stage 3\nadditive · 1k\nunbal · enc 40.7%",            1, ["✗","✗","✓","✗"]),
    ("v2 · Stage 5a\ntoken · 1k\nunbal · enc 40.7%",              2, ["✗","✓","✓","✗"]),
    ("v3 · Stage 5b\ntoken · 2k\n★BAL · enc 40.7%",               2, ["✓","✗","✓","✗"]),
    ("v4 · Stage 5c ★\ntoken · 3k\n★BAL · enc 51.6%",             3, ["✗","✓","✓","✓"]),
    ("v5 · Stage 6a\ntoken · 3k · BAL\nATCNet 57.7% + aux 0.2",   1, ["✗","✗","✗","✓"]),
    ("v6 · Stage 6b\ntoken · 3k · BAL\nATCNet 57.7%",             2, ["✗","✓","✗","✓"]),
    ("v7 · Stage 7\ntoken · 5k · BAL\nEEGNet 51.6%",              2, ["✓","✓","✗","✗"]),
    ("v8 · Stage 8\ntoken · 5k · BAL\nATCNet+expert-scratch",     2, ["✗","✓","✗","✓"]),
]
classes = ["left_fist (Δx<0)", "right_fist (Δx>0)", "both_fists (grip<0)", "both_feet (Δy>0)"]
class_colors_pass = ["#2ca02c", "#3aae45", "#4cb96a", "#1c8a32"]
COL_PASS = "#2ca02c"
COL_FAIL = "#d62728"

fig = plt.figure(figsize=(15, 8))
gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.30)
ax  = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

n_var = len(variants)
n_cls = len(classes)
bar_w = 0.18
x = np.arange(n_var)

# ── Upper: per-class pass/fail grid ─────────────────────────────────────────
for j, cls in enumerate(classes):
    offset = (j - n_cls/2 + 0.5) * bar_w
    cols   = [COL_PASS if v[2][j] == "✓" else COL_FAIL for v in variants]
    heights = [1.0] * n_var
    bars = ax.bar(x + offset, heights, bar_w,
                  color=cols, edgecolor="white", linewidth=1.4)
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width()/2, 0.5,
                variants[i][2][j],
                ha="center", va="center",
                fontsize=18, color="white", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([v[0] for v in variants], fontsize=8.5)
ax.set_yticks([])
ax.set_ylim(0, 1.15)
ax.set_title("Controllability progression — v1 through v8\n"
             "Each cell = one EEG condition · green ✓ = passes directional test  ·  red ✗ = fails",
             fontsize=12, fontweight="bold")

# Class legend in the upper right
legend_h = [
    mpatches.Patch(facecolor=COL_PASS, edgecolor="white", label="Passed directional test"),
    mpatches.Patch(facecolor=COL_FAIL, edgecolor="white", label="Failed"),
]
ax.legend(handles=legend_h, loc="upper right", fontsize=9, framealpha=0.95)

# Class column labels inside the plot (top)
for j, cls in enumerate(classes):
    offset = (j - n_cls/2 + 0.5) * bar_w
    ax.text(0 + offset - bar_w, 1.10, cls.split(" ")[0],
            ha="left", va="bottom", fontsize=8, color="#555555")

# ── Lower: total pass count bar chart ────────────────────────────────────────
n_pass = [v[1] for v in variants]
bars2 = ax2.bar(x, n_pass, 0.55,
                color=["#888888","#888888","#888888","#2ca02c",
                       "#888888","#888888","#888888","#888888"],
                edgecolor="white")
ax2.set_xticks(x)
ax2.set_xticklabels([f"v{i+1}" for i in range(n_var)], fontsize=10)
ax2.set_ylim(0, 4.5)
ax2.set_yticks([0,1,2,3,4])
ax2.set_ylabel("# passed / 4", fontsize=10)
ax2.grid(axis="y", alpha=0.25)
ax2.set_title("Aggregate score per variant — v4 still the only run that broke 3/4",
              fontsize=10, fontweight="bold")
for b, p in zip(bars2, n_pass):
    ax2.text(b.get_x() + b.get_width()/2, p + 0.12,
             f"{p}/4", ha="center", fontsize=10, fontweight="bold")
ax2.axhline(4, color="#444444", ls=":", lw=0.8)
ax2.text(n_var - 0.5, 4.0 + 0.05, "uniform 4/4 target", fontsize=8, color="#444444",
         ha="right", va="bottom")

plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"Saved: {OUT}")
