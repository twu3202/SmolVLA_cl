#!/usr/bin/env python
"""
Representation-learning analysis for the SmolVLA+EEG project.

Three experiments to probe whether EEG and the robot-action modality have
learned anything discriminative or shared:

1. CKA (Centered Kernel Alignment) — paired EEG embeddings vs LIBERO actions
   compared against a randomly-shuffled control. Tells us how aligned the two
   representational spaces are.

2. t-SNE / PCA visualization — project EEGNet's 64-dim embeddings to 2D and
   color by motor-imagery class. If clusters separate cleanly, the encoder has
   learned discriminative features for the 4 MI classes.

3. Linear probing — fit a tiny classifier from EEG embedding to:
   (a) MI class (sanity check, should match pretraining val accuracy)
   (b) LIBERO action class via synthetic mapping (does EEG carry transferable
       motor information, vs random baseline)

Outputs:
  ./eval_output/cka_analysis.png
  ./eval_output/eeg_tsne.png
  ./eval_output/probing_results.png
  ./eval_output/representation_summary.json

Usage:
    /opt/anaconda3/envs/lerobot/bin/python analyze_representations.py
"""

import glob
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

LIBERO_PATH = "/Users/r/LIBERO"
if LIBERO_PATH not in sys.path:
    sys.path.insert(0, LIBERO_PATH)

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition       import PCA
from sklearn.manifold            import TSNE
from sklearn.linear_model        import LogisticRegression, Ridge
from sklearn.model_selection     import train_test_split
from sklearn.preprocessing       import StandardScaler
from sklearn.metrics             import accuracy_score

from eeg_encoder import EEGNet
from train_smolvla_eeg import action_to_eeg_class
from utils import get_device

# ── Config ────────────────────────────────────────────────────────────────────
SUITE          = os.environ.get("SUITE", "libero_spatial")
EEG_DATA_PATH  = "./data/eeg_physionet/epochs.npz"
EEG_ENC_PT     = "./checkpoints/eeg_encoder/encoder_only.pt"
STATS_PATH     = f"./checkpoints/{SUITE}/dataset_stats.pt"
DATASET_DIR    = f"/Users/r/LIBERO/libero/datasets/{SUITE}"
OUTPUT_DIR     = "./eval_output"
N_LIBERO       = 2000        # action samples to draw from LIBERO
SEED           = 42

CLASS_NAMES = {0: "left_fist", 1: "right_fist", 2: "both_fists", 3: "both_feet"}
CLASS_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}
# ──────────────────────────────────────────────────────────────────────────────


# ── CKA implementation ───────────────────────────────────────────────────────

def gram_linear(X: np.ndarray) -> np.ndarray:
    """Linear Gram matrix XX^T."""
    return X @ X.T


def center_gram(K: np.ndarray) -> np.ndarray:
    """HKH centering."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Centered Kernel Alignment between representations X (n×d1) and Y (n×d2).
    Returns scalar in [0, 1]. Invariant to scaling and orthogonal transforms.
    """
    Kx = center_gram(gram_linear(X))
    Ky = center_gram(gram_linear(Y))
    hsic_xy = (Kx * Ky).sum()
    hsic_xx = (Kx * Kx).sum()
    hsic_yy = (Ky * Ky).sum()
    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-12))


# ── Data loading ─────────────────────────────────────────────────────────────

def load_eeg_embeddings(device: str) -> tuple[np.ndarray, np.ndarray]:
    """Run all PhysioNet epochs through EEGNet → (N, 64) embeddings + labels."""
    data = np.load(EEG_DATA_PATH)
    X, y = data["X"], data["y"]
    print(f"  EEG epochs: {X.shape}, classes: {np.bincount(y)}")

    enc_ckpt = torch.load(EEG_ENC_PT, map_location="cpu", weights_only=False)
    model = EEGNet(n_channels=64, n_timepoints=320, n_classes=4, embed_dim=64)
    model.load_state_dict(enc_ckpt["backbone_state"], strict=False)
    model = model.to(device).eval()

    embeds = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            batch = torch.from_numpy(X[i:i+64]).unsqueeze(1).to(device)   # (B,1,C,T)
            emb   = model.encode(batch).cpu().numpy()
            embeds.append(emb)
    return np.concatenate(embeds), y


def load_libero_actions(stats: dict, n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Sample n actions from LIBERO demos. Returns (raw_actions, action_classes)."""
    a_mean = stats["action"]["mean"].numpy()
    a_std  = stats["action"]["std"].numpy()
    rng = np.random.default_rng(SEED)
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.hdf5")))

    all_a = []
    for f_path in files:
        with h5py.File(f_path, "r") as f:
            for dk in sorted(f["data"].keys()):
                a = f["data"][dk]["actions"][:].astype(np.float32)
                all_a.append(a)
    all_a = np.concatenate(all_a, axis=0)
    print(f"  LIBERO total actions: {len(all_a)}")

    # Random subset
    idx = rng.choice(len(all_a), n, replace=False)
    sample_raw = all_a[idx]
    sample_norm = (sample_raw - a_mean) / (a_std + 1e-8)

    # Compute class via the same mapping used in training
    classes = np.array([
        action_to_eeg_class(torch.from_numpy(a)) for a in sample_raw
    ])
    print(f"  Action class distribution: {np.bincount(classes, minlength=4)}")
    return sample_raw, sample_norm, classes


# ── Experiment 1: CKA ────────────────────────────────────────────────────────

def experiment_cka(eeg_embeds, eeg_labels, action_norm, action_classes):
    """Compare CKA(EEG, action) under synthetic pairing vs random shuffle."""
    print("\n[1/3] CKA analysis ...")
    # Build synthetic pairing: pick EEG epochs whose label matches action_class
    rng = np.random.default_rng(SEED)
    eeg_by_class = {c: np.where(eeg_labels == c)[0] for c in range(4)}

    matched_eeg_idx = np.array([
        rng.choice(eeg_by_class[c]) for c in action_classes
    ])
    paired_eeg = eeg_embeds[matched_eeg_idx]      # (N, 64)
    # Random shuffle control
    shuffled_eeg = eeg_embeds[rng.permutation(matched_eeg_idx)]

    # Standardize for fair comparison
    paired_eeg_z   = StandardScaler().fit_transform(paired_eeg)
    shuffled_eeg_z = StandardScaler().fit_transform(shuffled_eeg)
    action_z       = StandardScaler().fit_transform(action_norm)

    # Sub-sample for CKA computation cost (n^2 memory)
    n_cka = min(800, len(action_z))
    sel = rng.choice(len(action_z), n_cka, replace=False)

    cka_paired = cka(paired_eeg_z[sel],   action_z[sel])
    cka_random = cka(shuffled_eeg_z[sel], action_z[sel])

    # Also compute self-CKA as upper bound
    cka_self_eeg    = cka(paired_eeg_z[sel],   paired_eeg_z[sel])
    cka_self_action = cka(action_z[sel],       action_z[sel])

    # Per-component CKA: which action dim aligns most with EEG?
    per_dim = []
    for d in range(7):
        cka_d = cka(paired_eeg_z[sel], action_z[sel, d:d+1])
        per_dim.append(cka_d)

    print(f"  CKA(EEG_paired, action)   = {cka_paired:.4f}")
    print(f"  CKA(EEG_random, action)   = {cka_random:.4f}  (control)")
    print(f"  CKA self-similarity check : EEG={cka_self_eeg:.3f}, action={cka_self_action:.3f}")
    print(f"  Per-action-dim CKA: {[f'{v:.3f}' for v in per_dim]}")

    return {
        "cka_paired": cka_paired,
        "cka_random": cka_random,
        "per_dim_cka": per_dim,
    }


# ── Experiment 2: t-SNE / PCA ────────────────────────────────────────────────

def experiment_tsne(eeg_embeds, eeg_labels):
    """Project EEG embeddings to 2D with both PCA and t-SNE; color by class."""
    print("\n[2/3] t-SNE + PCA visualization ...")
    rng_state = SEED

    # Subsample for t-SNE speed
    n = min(800, len(eeg_embeds))
    rng = np.random.default_rng(SEED)
    sel = rng.choice(len(eeg_embeds), n, replace=False)
    X = StandardScaler().fit_transform(eeg_embeds[sel])
    y = eeg_labels[sel]

    pca = PCA(n_components=2, random_state=rng_state).fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=30, random_state=rng_state,
                init="pca", learning_rate="auto").fit_transform(X)

    # Linear separability: train k-fold logistic regression on full embeddings
    from sklearn.model_selection import cross_val_score
    clf = LogisticRegression(max_iter=2000)
    scores = cross_val_score(clf, eeg_embeds, eeg_labels, cv=5, scoring="accuracy")
    sep_acc = scores.mean()
    print(f"  Linear separability (5-fold CV): {sep_acc*100:.1f}% (chance=25%)")

    return {
        "pca": pca, "tsne": tsne, "labels": y,
        "linear_separability": sep_acc,
    }


# ── Experiment 3: Linear probing ─────────────────────────────────────────────

def experiment_probing(eeg_embeds, eeg_labels, action_norm, action_classes):
    """
    (a) MI class probe (sanity)
    (b) LIBERO action regression: can EEG predict ANY useful info about actions?
        - Synthetic pairing baseline
        - Random pairing control
    """
    print("\n[3/3] Linear probing ...")
    rng = np.random.default_rng(SEED)

    # ── (a) Self-probe: EEG → its own MI class ────────────────────────────
    Xtr, Xte, ytr, yte = train_test_split(eeg_embeds, eeg_labels,
                                          test_size=0.2, random_state=SEED,
                                          stratify=eeg_labels)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    mi_acc = accuracy_score(yte, clf.predict(sc.transform(Xte)))
    print(f"  (a) EEG → MI class:  test acc = {mi_acc*100:.1f}% (chance=25%)")

    # ── (b) Cross-modal: EEG → LIBERO action class via synthetic pairing ──
    # Build paired dataset
    eeg_by_class = {c: np.where(eeg_labels == c)[0] for c in range(4)}
    paired_eeg_idx = np.array([rng.choice(eeg_by_class[c]) for c in action_classes])
    paired_X = eeg_embeds[paired_eeg_idx]    # (N, 64)
    paired_y = action_classes                 # (N,)

    Xtr, Xte, ytr, yte = train_test_split(paired_X, paired_y, test_size=0.2,
                                          random_state=SEED, stratify=paired_y)
    clf = LogisticRegression(max_iter=2000).fit(
        StandardScaler().fit_transform(Xtr), ytr,
    )
    paired_acc = accuracy_score(yte, clf.predict(StandardScaler().fit(Xtr).transform(Xte)))

    # Random control: shuffle EEG-action pairing → should be ~chance
    shuffled_idx = rng.permutation(paired_eeg_idx)
    shuf_X = eeg_embeds[shuffled_idx]
    Xtr, Xte, ytr, yte = train_test_split(shuf_X, paired_y, test_size=0.2,
                                          random_state=SEED, stratify=paired_y)
    clf = LogisticRegression(max_iter=2000).fit(
        StandardScaler().fit_transform(Xtr), ytr,
    )
    random_acc = accuracy_score(yte, clf.predict(StandardScaler().fit(Xtr).transform(Xte)))

    print(f"  (b) EEG → action class (paired): {paired_acc*100:.1f}%")
    print(f"      EEG → action class (random): {random_acc*100:.1f}%  (chance=25%)")

    # ── (c) Regression: EEG → continuous action (translation only) ────────
    # Use paired data
    a_xyz = action_norm[:, :3]   # (N, 3) translation deltas
    Xtr, Xte, ytr, yte = train_test_split(paired_X, a_xyz, test_size=0.2,
                                          random_state=SEED)
    sc = StandardScaler().fit(Xtr)
    reg = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
    ypred = reg.predict(sc.transform(Xte))
    r2_paired = 1 - ((yte - ypred)**2).sum() / ((yte - yte.mean(0))**2).sum()

    # Random pairing control
    Xtr, Xte, ytr, yte = train_test_split(shuf_X, a_xyz, test_size=0.2,
                                          random_state=SEED)
    sc = StandardScaler().fit(Xtr)
    reg = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
    ypred = reg.predict(sc.transform(Xte))
    r2_random = 1 - ((yte - ypred)**2).sum() / ((yte - yte.mean(0))**2).sum()

    print(f"  (c) EEG → Δxyz regression R²:")
    print(f"      paired:  {r2_paired:+.4f}")
    print(f"      random:  {r2_random:+.4f}  (control, expects ~0)")

    return {
        "mi_self_acc":        mi_acc,
        "action_paired_acc":  paired_acc,
        "action_random_acc":  random_acc,
        "regression_paired_r2": r2_paired,
        "regression_random_r2": r2_random,
    }


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_all(cka_res, tsne_res, probe_res):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                          left=0.06, right=0.97, top=0.92, bottom=0.07)

    # ── Panel 1: CKA bar ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar(["paired", "random\n(control)"],
                  [cka_res["cka_paired"], cka_res["cka_random"]],
                  color=["#2ca02c", "#aaaaaa"], alpha=0.85)
    for b, v in zip(bars, [cka_res["cka_paired"], cka_res["cka_random"]]):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("CKA")
    ax.set_title("CKA(EEG embedding, action)\n— how aligned are the spaces?", fontsize=10)
    ax.set_ylim(0, max(cka_res["cka_paired"], cka_res["cka_random"]) * 1.4 + 0.05)
    ax.grid(True, axis="y", alpha=0.3)

    # ── Panel 2: per-action-dim CKA ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    labels = ["Δx","Δy","Δz","Δroll","Δpitch","Δyaw","gripper"]
    ax.bar(labels, cka_res["per_dim_cka"], color="#1f77b4", alpha=0.85)
    ax.set_ylabel("CKA")
    ax.set_title("CKA(EEG, single action dim)\n— which dim aligns most?", fontsize=10)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # ── Panel 3: probing accuracies ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    probe_names = ["EEG→MI\n(self)",
                   "EEG→action\n(paired)",
                   "EEG→action\n(random)"]
    probe_vals  = [probe_res["mi_self_acc"]*100,
                   probe_res["action_paired_acc"]*100,
                   probe_res["action_random_acc"]*100]
    cols = ["#1f77b4","#2ca02c","#aaaaaa"]
    bars = ax.bar(probe_names, probe_vals, color=cols, alpha=0.85)
    ax.axhline(25, color="red", ls="--", lw=1, alpha=0.6, label="chance (25%)")
    for b, v in zip(bars, probe_vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{v:.1f}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("classification accuracy %")
    ax.set_title("Linear probing accuracy", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(probe_vals) * 1.25)

    # ── Panel 4: PCA scatter ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    for c in range(4):
        m = tsne_res["labels"] == c
        ax.scatter(tsne_res["pca"][m, 0], tsne_res["pca"][m, 1],
                   c=CLASS_COLORS[c], label=CLASS_NAMES[c],
                   alpha=0.55, s=14, edgecolor="none")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("PCA of EEG embeddings\n(linear projection)", fontsize=10)
    ax.legend(fontsize=8, markerscale=1.5, loc="best")
    ax.grid(True, alpha=0.3)

    # ── Panel 5: t-SNE scatter ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    for c in range(4):
        m = tsne_res["labels"] == c
        ax.scatter(tsne_res["tsne"][m, 0], tsne_res["tsne"][m, 1],
                   c=CLASS_COLORS[c], label=CLASS_NAMES[c],
                   alpha=0.55, s=14, edgecolor="none")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    sep = tsne_res["linear_separability"] * 100
    ax.set_title(f"t-SNE of EEG embeddings\n(linear sep CV acc: {sep:.1f}%)",
                 fontsize=10)
    ax.legend(fontsize=8, markerscale=1.5, loc="best")
    ax.grid(True, alpha=0.3)

    # ── Panel 6: regression R² ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    r2_p = probe_res["regression_paired_r2"]
    r2_r = probe_res["regression_random_r2"]
    bars = ax.bar(["paired", "random\n(control)"],
                  [r2_p, r2_r],
                  color=["#2ca02c", "#aaaaaa"], alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    for b, v in zip(bars, [r2_p, r2_r]):
        ax.text(b.get_x()+b.get_width()/2,
                b.get_height() + 0.01 if v >= 0 else b.get_height() - 0.04,
                f"{v:+.4f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("R²")
    ax.set_title("EEG → Δxyz regression\n— continuous action prediction", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("EEG Representation Analysis — discriminativity & alignment with LIBERO actions",
                 fontsize=13, fontweight="bold")

    out = os.path.join(OUTPUT_DIR, "representation_analysis.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nPlot saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()
    print(f"=== Representation Learning Analysis ===\nDevice: {device}\n")

    # Validate
    for path in [EEG_DATA_PATH, EEG_ENC_PT, STATS_PATH]:
        if not os.path.exists(path):
            print(f"Missing: {path}")
            raise SystemExit(1)

    print("[setup] Loading EEG embeddings ...")
    eeg_embeds, eeg_labels = load_eeg_embeddings(device)

    print("[setup] Loading LIBERO actions ...")
    stats = torch.load(STATS_PATH, map_location="cpu", weights_only=False)
    action_raw, action_norm, action_classes = load_libero_actions(stats, n=N_LIBERO)

    cka_res   = experiment_cka(eeg_embeds, eeg_labels, action_norm, action_classes)
    tsne_res  = experiment_tsne(eeg_embeds, eeg_labels)
    probe_res = experiment_probing(eeg_embeds, eeg_labels, action_norm, action_classes)

    # Save numeric summary
    summary = {
        "n_eeg_epochs":      int(len(eeg_embeds)),
        "n_libero_actions":  int(len(action_norm)),
        "cka": {
            "paired":     float(cka_res["cka_paired"]),
            "random":     float(cka_res["cka_random"]),
            "per_dim":    [float(v) for v in cka_res["per_dim_cka"]],
        },
        "tsne_linear_separability": float(tsne_res["linear_separability"]),
        "probing": {
            "mi_self_acc":          float(probe_res["mi_self_acc"]),
            "action_paired_acc":    float(probe_res["action_paired_acc"]),
            "action_random_acc":    float(probe_res["action_random_acc"]),
            "regression_paired_r2": float(probe_res["regression_paired_r2"]),
            "regression_random_r2": float(probe_res["regression_random_r2"]),
        },
    }
    with open(os.path.join(OUTPUT_DIR, "representation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary JSON: {OUTPUT_DIR}/representation_summary.json")

    plot_all(cka_res, tsne_res, probe_res)

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("INTERPRETATION")
    print(f"{'='*72}")
    cka_ratio = cka_res["cka_paired"] / max(cka_res["cka_random"], 1e-6)
    if cka_res["cka_paired"] < 0.05:
        print("• EEG and action spaces are essentially ORTHOGONAL — EEG carries")
        print("  information not present in the action representation. Whether this")
        print("  is useful depends on if the model can fuse them (it currently can't).")
    elif cka_ratio < 1.5:
        print("• Pairing didn't materially raise CKA above random — the EEG encoder")
        print("  hasn't learned action-relevant structure (or no such structure exists).")
    else:
        print("• Synthetic pairing increased CKA notably above random — EEG embeddings")
        print("  do contain class-discriminative information aligned with action class.")

    sep = tsne_res["linear_separability"]
    if sep > 0.50:
        print(f"• EEG embeddings are LINEARLY SEPARABLE for MI class ({sep*100:.1f}%) —")
        print("  EEGNet has learned discriminative features. The encoder works.")
    elif sep > 0.30:
        print(f"• EEG embeddings show MODEST class separation ({sep*100:.1f}%) — ")
        print("  encoder learned something but signal is weak (small training set).")
    else:
        print(f"• EEG embeddings barely beat chance ({sep*100:.1f}%) — encoder under-trained.")

    if probe_res["regression_paired_r2"] > 0.3:
        print("• EEG can predict robot action translation directly — strong cross-modal signal.")
    elif probe_res["regression_paired_r2"] - probe_res["regression_random_r2"] > 0.1:
        print("• Modest signal exists from EEG to action via the synthetic mapping.")
    else:
        print("• EEG provides little continuous action information beyond the mapping itself.")


if __name__ == "__main__":
    main()
