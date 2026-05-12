#!/usr/bin/env python
"""
Focused per-class comparison: EEGNet vs ATCNet.

Two panels:
  Left:  Per-class linear separability bar chart (5-fold LR CV)
  Right: Controllability test pass/fail across variants that used each encoder

This is the "why does the worse-on-paper encoder win" plot referenced in the README.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "./eval_output/atcnet_vs_eegnet.png"

# Per-class LR accuracy (from compare_eeg_embeddings.py output)
classes = ["left_fist", "right_fist", "both_fists", "both_feet"]
eegnet  = [68.9, 65.1, 72.0, 69.6]      # overall 67.2%
atcnet  = [64.3, 61.8, 73.9, 70.2]      # overall 65.3%

# Controllability — which encoder passed which class, aggregated across runs
# v4 (EEGNet 3000):      right_fist, both_fists, both_feet
# v7 (EEGNet 5000):      left_fist, right_fist
# Union for EEGNet best :  left, right, both_fists, both_feet  (covers all, never in one run)
# v6 (ATCNet 3000):      right_fist, both_feet
# v8 (ATCNet 5000 scratch): right_fist, both_feet
eegnet_pass = ["✓ (v7)", "✓ (v4/v7)", "✓ (v4)",    "✓ (v4)"]
atcnet_pass = ["✗",      "✓ (v6/v8)",  "✗",         "✓ (v6/v8)"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.3, 1.0]})

# ─── Left: per-class linear separability ──────────────────────────────────────
x = np.arange(len(classes))
w = 0.38
b1 = ax1.bar(x - w/2, eegnet, w, label="EEGNet (LR 67.2%)", color="#1f77b4", edgecolor="white")
b2 = ax1.bar(x + w/2, atcnet, w, label="ATCNet (LR 65.3%)", color="#d62728", edgecolor="white")

for b, v in zip(b1, eegnet):
    ax1.text(b.get_x() + b.get_width()/2, v + 0.4, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
for b, v in zip(b2, atcnet):
    ax1.text(b.get_x() + b.get_width()/2, v + 0.4, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")

# Spread annotation
spread_e = max(eegnet) - min(eegnet)
spread_a = max(atcnet) - min(atcnet)
ax1.text(0.02, 0.97,
         f"EEGNet spread: {spread_e:.1f} pp (uniform)\nATCNet spread: {spread_a:.1f} pp (uneven)",
         transform=ax1.transAxes, va="top", ha="left",
         fontsize=10, fontweight="bold",
         bbox=dict(facecolor="white", edgecolor="#888", alpha=0.92))

ax1.set_xticks(x)
ax1.set_xticklabels(classes, fontsize=10)
ax1.set_ylim(55, 80)
ax1.set_ylabel("Per-class linear separability (%)")
ax1.set_title("Per-class embedding quality (5-fold LR CV)\n"
              "EEGNet is more uniform; ATCNet is sharper but uneven",
              fontsize=11, fontweight="bold")
ax1.grid(True, axis="y", alpha=0.3)
ax1.legend(loc="lower right", fontsize=9)
ax1.axhline(np.mean(eegnet), color="#1f77b4", ls="--", lw=0.8, alpha=0.6)
ax1.axhline(np.mean(atcnet), color="#d62728", ls="--", lw=0.8, alpha=0.6)

# ─── Right: controllability table ─────────────────────────────────────────────
ax2.axis("off")
cell_text = []
for i, c in enumerate(classes):
    cell_text.append([c, eegnet_pass[i], atcnet_pass[i]])

table = ax2.table(
    cellText=cell_text,
    colLabels=["MI class", "EEGNet runs", "ATCNet runs"],
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

# Color cells by pass/fail
for i in range(4):
    for j in (1, 2):
        cell = table[(i + 1, j)]
        txt  = cell.get_text().get_text()
        if "✓" in txt:
            cell.set_facecolor("#d6f5d6")
        elif "✗" in txt:
            cell.set_facecolor("#f9d6d6")

# Header style
for j in range(3):
    cell = table[(0, j)]
    cell.set_facecolor("#404040")
    cell.set_text_props(color="white", fontweight="bold")

ax2.set_title("Which encoder passes which directional test?\n"
              "ATCNet never passes left_fist or both_fists, in any run.",
              fontsize=11, fontweight="bold", pad=20)

# Annotation explaining the takeaway
ax2.text(0.5, 0.05,
         "ATCNet's 'extra' MI accuracy (57.7% vs 51.6%) does not translate\n"
         "to controllability — it shows up as sharpness on 2 classes only.",
         transform=ax2.transAxes, ha="center", va="bottom",
         fontsize=9.5, style="italic", color="#444")

plt.suptitle("Why the 'stronger' encoder controls less:  ATCNet vs EEGNet",
             fontsize=13, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"Saved: {OUT}")
