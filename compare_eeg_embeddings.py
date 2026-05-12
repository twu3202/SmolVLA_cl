#!/usr/bin/env python
"""
Side-by-side 2D embedding comparison: EEGNet vs ATCNet.

2×2 grid:
  Row 1: EEGNet  — PCA | t-SNE
  Row 2: ATCNet  — PCA | t-SNE

Each panel is annotated with:
  - Overall linear separability (5-fold LR CV accuracy)
  - Per-class recall so we can see which classes ATCNet underserves

Usage:
    cd /Users/r/Projects/SmolVLA_cl
    /opt/anaconda3/envs/lerobot/bin/python compare_eeg_embeddings.py
"""

import os, sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.decomposition   import PCA
from sklearn.manifold        import TSNE
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics         import confusion_matrix

sys.path.insert(0, "/Users/r/Projects/SmolVLA_cl")
from eeg_encoder import EEGNet
from eeg_encoder_atcnet import ATCNet
from utils import get_device

# ── Config ────────────────────────────────────────────────────────────────────
EEG_DATA_PATH   = "./data/eeg_physionet/epochs.npz"
EEGNET_CKPT     = "./checkpoints/eeg_encoder/encoder_only.pt"
ATCNET_CKPT     = "./checkpoints/eeg_encoder_atcnet/encoder_only.pt"
OUTPUT_PATH     = "./eval_output/compare_eeg_embeddings.png"
TSNE_PERPLEXITY = 40
TSNE_N_ITER     = 1000
SUBSAMPLE       = 2000          # keep manageable; use None for all 9377
SEED            = 42
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES  = ["left_fist", "right_fist", "both_fists", "both_feet"]
CLASS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def load_data(subsample=None):
    data = np.load(EEG_DATA_PATH)
    X, y = data["X"].astype(np.float32), data["y"].astype(np.int64)
    if subsample is not None and subsample < len(X):
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X), subsample, replace=False)
        X, y = X[idx], y[idx]
    print(f"Data: {X.shape}  classes {np.unique(y, return_counts=True)}")
    return X, y


def build_eegnet(device):
    model = EEGNet(n_channels=64, n_timepoints=320, n_classes=4, embed_dim=64)
    ckpt  = torch.load(EEGNET_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["backbone_state"], strict=False)
    return model.to(device).eval()


def build_atcnet(device):
    model = ATCNet(n_channels=64, n_timepoints=320, n_classes=4, embed_dim=64)
    ckpt  = torch.load(ATCNET_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["backbone_state"], strict=False)
    return model.to(device).eval()


@torch.no_grad()
def get_embeddings(model, X, device, batch_size=256):
    embs = []
    tensor = torch.from_numpy(X).unsqueeze(1)      # (N, 1, 64, 320)
    for i in range(0, len(tensor), batch_size):
        xb = tensor[i:i+batch_size].to(device)
        embs.append(model.encode(xb).cpu().numpy())
    return np.concatenate(embs, axis=0)             # (N, 64)


def linear_separability(embs, labels):
    """5-fold stratified LR, returns overall accuracy + per-class recall."""
    sc  = StandardScaler().fit(embs)
    Z   = sc.transform(embs)
    lr  = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                             random_state=SEED)
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    acc = cross_val_score(lr, Z, labels, cv=cv, scoring="accuracy").mean()

    # Per-class recall via single fit on all data (for display only)
    lr.fit(Z, labels)
    preds  = lr.predict(Z)
    cm     = confusion_matrix(labels, preds)
    recall = cm.diagonal() / cm.sum(axis=1)
    return float(acc), recall


def project_pca(embs):
    sc = StandardScaler().fit(embs)
    Z  = sc.transform(embs)
    pc = PCA(n_components=2, random_state=SEED).fit_transform(Z)
    return pc


def project_tsne(embs):
    sc = StandardScaler().fit(embs)
    Z  = sc.transform(embs)
    ts = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY,
              max_iter=TSNE_N_ITER, random_state=SEED, verbose=0)
    return ts.fit_transform(Z)


def scatter_panel(ax, proj, labels, title, recall, acc):
    for c_idx, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        mask = labels == c_idx
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=color, s=6, alpha=0.45, linewidths=0, label=name)

    # Recall annotation box
    recall_str = "\n".join(
        f"{name}: {r*100:.1f}%" for name, r in zip(CLASS_NAMES, recall)
    )
    ax.text(0.02, 0.98, recall_str,
            transform=ax.transAxes, fontsize=7.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#aaa", alpha=0.85))

    ax.set_title(f"{title}\nLR acc={acc*100:.1f}%", fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])


def main():
    device = get_device()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Loading data ...")
    X, y = load_data(subsample=SUBSAMPLE)

    print("Building encoders ...")
    eegnet_model = build_eegnet(device)
    atcnet_model = build_atcnet(device)

    print("Computing EEGNet embeddings ...")
    emb_eeg = get_embeddings(eegnet_model, X, device)
    print("Computing ATCNet embeddings ...")
    emb_atc = get_embeddings(atcnet_model, X, device)

    print("Linear separability (EEGNet) ...")
    acc_eeg, recall_eeg = linear_separability(emb_eeg, y)
    print(f"  EEGNet  LR acc={acc_eeg*100:.1f}%  per-class: "
          + "  ".join(f"{n}:{r*100:.1f}%" for n,r in zip(CLASS_NAMES, recall_eeg)))

    print("Linear separability (ATCNet) ...")
    acc_atc, recall_atc = linear_separability(emb_atc, y)
    print(f"  ATCNet  LR acc={acc_atc*100:.1f}%  per-class: "
          + "  ".join(f"{n}:{r*100:.1f}%" for n,r in zip(CLASS_NAMES, recall_atc)))

    print("PCA projections ...")
    pca_eeg = project_pca(emb_eeg)
    pca_atc = project_pca(emb_atc)

    print("t-SNE projections (may take ~1 min) ...")
    tsne_eeg = project_tsne(emb_eeg)
    print("  EEGNet t-SNE done")
    tsne_atc = project_tsne(emb_atc)
    print("  ATCNet t-SNE done")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(
        "EEG Embedding 2D Comparison — EEGNet (51.6% MI) vs ATCNet (57.7% MI)\n"
        f"n={len(X)} epochs, subsample={SUBSAMPLE}",
        fontsize=13, fontweight="bold", y=0.99,
    )

    scatter_panel(axes[0, 0], pca_eeg,  y, "EEGNet — PCA",   recall_eeg, acc_eeg)
    scatter_panel(axes[0, 1], tsne_eeg, y, "EEGNet — t-SNE", recall_eeg, acc_eeg)
    scatter_panel(axes[1, 0], pca_atc,  y, "ATCNet — PCA",   recall_atc, acc_atc)
    scatter_panel(axes[1, 1], tsne_atc, y, "ATCNet — t-SNE", recall_atc, acc_atc)

    # Row labels
    for row, label in enumerate(["EEGNet", "ATCNet"]):
        fig.text(0.01, 0.75 - row * 0.5, label,
                 va="center", ha="left", fontsize=13, fontweight="bold",
                 rotation=90, color="#333")

    # Shared legend
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=c, markersize=8, label=n)
        for n, c in zip(CLASS_NAMES, CLASS_COLORS)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.01))

    # Per-class recall difference annotation (ATCNet − EEGNet)
    diff_lines = []
    for name, r_atc, r_eeg in zip(CLASS_NAMES, recall_atc, recall_eeg):
        diff = (r_atc - r_eeg) * 100
        sign = "+" if diff >= 0 else ""
        diff_lines.append(f"{name}: {sign}{diff:.1f}%")
    diff_text = "ATCNet − EEGNet recall:\n" + "\n".join(diff_lines)
    fig.text(0.76, 0.48, diff_text, fontsize=9, va="center",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff9e6",
                       edgecolor="#ccc", alpha=0.9))

    plt.tight_layout(rect=[0.04, 0.06, 1.0, 0.97])
    plt.savefig(OUTPUT_PATH, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
