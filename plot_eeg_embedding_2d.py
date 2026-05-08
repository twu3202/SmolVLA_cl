#!/usr/bin/env python
"""
Standalone 2D embedding visualization of EEGNet's 64-dim outputs.
Produces side-by-side PCA and t-SNE projections, color-coded by MI class.

Usage:
    /opt/anaconda3/envs/lerobot/bin/python plot_eeg_embedding_2d.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition   import PCA
from sklearn.manifold        import TSNE
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import cross_val_score

from eeg_encoder import EEGNet
from utils import get_device

EEG_DATA_PATH = "./data/eeg_physionet/epochs.npz"
EEG_ENC_PT    = "./checkpoints/eeg_encoder/encoder_only.pt"
OUT_PATH      = "./eval_output/eeg_embedding_2d.png"
SEED          = 42

CLASS_NAMES  = {0: "left_fist", 1: "right_fist", 2: "both_fists", 3: "both_feet"}
CLASS_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}


def main():
    device = get_device()
    print(f"Device: {device}")

    # Load epochs
    data = np.load(EEG_DATA_PATH)
    X, y = data["X"], data["y"]
    print(f"EEG epochs: {X.shape}")

    # Encode
    enc = EEGNet(n_channels=64, n_timepoints=320, n_classes=4, embed_dim=64)
    enc.load_state_dict(
        torch.load(EEG_ENC_PT, map_location="cpu",
                   weights_only=False)["backbone_state"], strict=False)
    enc = enc.to(device).eval()

    embeds = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            batch = torch.from_numpy(X[i:i+64]).unsqueeze(1).to(device)
            embeds.append(enc.encode(batch).cpu().numpy())
    embeds = np.concatenate(embeds)
    print(f"Embeddings: {embeds.shape}")

    # Standardize for projection
    Xz = StandardScaler().fit_transform(embeds)

    # Linear separability over the FULL embedding space
    sep = cross_val_score(LogisticRegression(max_iter=2000),
                          embeds, y, cv=5, scoring="accuracy").mean()
    print(f"Linear separability (5-fold CV): {sep*100:.1f}%")

    # 2D projections
    pca  = PCA(n_components=2, random_state=SEED).fit_transform(Xz)
    tsne = TSNE(n_components=2, perplexity=30, init="pca",
                learning_rate="auto", random_state=SEED).fit_transform(Xz)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        f"EEGNet 64-D Embedding Projected to 2D — 4 Motor-Imagery Classes\n"
        f"Linear separability (logistic regression, 5-fold CV): {sep*100:.1f}% (chance = 25%)",
        fontsize=12, fontweight="bold",
    )

    for ax, proj, title in [(axes[0], pca,  "PCA — first 2 principal components"),
                            (axes[1], tsne, "t-SNE — non-linear embedding (perplexity=30)")]:
        for c in range(4):
            m = (y == c)
            ax.scatter(proj[m, 0], proj[m, 1],
                       c=CLASS_COLORS[c], label=CLASS_NAMES[c],
                       alpha=0.6, s=22, edgecolor="white", linewidth=0.4)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=10, markerscale=1.4, loc="best", frameon=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
