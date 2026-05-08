#!/usr/bin/env python
"""
Show how controllability evolved across 4 architecture / training variants.

Each variant differs by ONE design decision; results isolate the contribution
of each change.

Variant         Injection      Steps  Class-bal  Encoder   Pass
v1 (Stage 3)    additive       1000   ✗          40.7%     1/4
v2 (Stage 5a)   token-level    1000   ✗          40.7%     2/4
v3 (Stage 5b)   token-level    2000   ✓          40.7%     2/4
v4 (Stage 5c)   token-level    3000   ✓          51.6%     3/4
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "./eval_output/controllability_progression.png"

variants = [
    ("v1: Additive\n1000 steps, unbal,\nencoder 40.7%",       1, ["✗","✗","✓","✗"]),
    ("v2: Token-level\n1000 steps, unbal,\nencoder 40.7%",    2, ["✗","✓","✓","✗"]),
    ("v3: Token-level\n2000 steps, BALANCED,\nencoder 40.7%", 2, ["✓","✗","✓","✗"]),
    ("v4: Token-level\n3000 steps, BALANCED,\nencoder 51.6%", 3, ["✗","✓","✓","✓"]),
]
classes = ["left_fist", "right_fist", "both_fists", "both_feet"]
colors  = {"✓": "#2ca02c", "✗": "#d62728"}

fig, ax = plt.subplots(figsize=(11, 6))
n_var = len(variants)
n_cls = len(classes)

bar_w = 0.18
x = np.arange(n_var)
for j, cls in enumerate(classes):
    offset = (j - n_cls/2 + 0.5) * bar_w
    heights = [1.0 for _ in variants]   # all bars same height
    cols    = [colors[v[2][j]] for v in variants]
    bars = ax.bar(x + offset, heights, bar_w,
                  color=cols, edgecolor="white", linewidth=1.5,
                  label=cls if j < n_cls else None)
    # Mark each cell
    for i, (b, mark) in enumerate(zip(bars, [v[2][j] for v in variants])):
        ax.text(b.get_x() + b.get_width()/2, 0.5, mark,
                ha="center", va="center", fontsize=20, color="white",
                fontweight="bold")
    # Class label below first bar of each class group
    ax.text(0 + offset, -0.08, cls, ha="center", va="top", fontsize=8,
            color="#444")

# Total pass annotation per variant
for i, v in enumerate(variants):
    n_pass = v[1]
    ax.text(i, 1.18, f"{n_pass}/4", ha="center", fontsize=15, fontweight="bold",
            color="#1f77b4")

ax.set_xticks(x)
ax.set_xticklabels([v[0] for v in variants], fontsize=9)
ax.set_yticks([])
ax.set_ylim(0, 1.4)
ax.set_title("EEG Controllability Progression — isolating each design choice",
             fontsize=13, fontweight="bold", pad=20)
# Custom legend showing color meaning
from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor=colors["✓"], edgecolor="white", label="passed"),
    Patch(facecolor=colors["✗"], edgecolor="white", label="failed"),
]
ax.legend(handles=legend_elems, loc="upper left", fontsize=10, frameon=True)

# Highlight design changes
ax.annotate("", xy=(0.7, 1.32), xytext=(0.3, 1.32),
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5))
ax.text(0.5, 1.36, "deeper VLM injection", ha="center", fontsize=8, color="#1f77b4")
ax.annotate("", xy=(1.7, 1.32), xytext=(1.3, 1.32),
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5))
ax.text(1.5, 1.36, "+ class balancing", ha="center", fontsize=8, color="#1f77b4")
ax.annotate("", xy=(2.7, 1.32), xytext=(2.3, 1.32),
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5))
ax.text(2.5, 1.36, "+ stronger encoder\n+ longer training", ha="center",
        fontsize=8, color="#1f77b4")

plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"Saved: {OUT}")
