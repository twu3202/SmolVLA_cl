#!/usr/bin/env python
"""
Pretrain EEGNet on PhysioNet EEGMMIDB (4-class motor imagery).

Trains a compact EEGNet to classify:
  0 = left fist MI
  1 = right fist MI
  2 = both fists MI
  3 = both feet MI

The trained encoder backbone (without the classification head) is then frozen
and used as an input modality in SmolVLA.

Usage:
    /opt/anaconda3/envs/lerobot/bin/python train_eeg_encoder.py

Output:
    ./checkpoints/eeg_encoder/eeg_encoder.pt   — full model (encoder + head)
    ./checkpoints/eeg_encoder/encoder_only.pt  — backbone only (for SmolVLA)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from eeg_encoder import EEGNet
from utils import get_device

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = "./data/eeg_physionet/epochs.npz"
OUT_DIR    = "./checkpoints/eeg_encoder"
EPOCHS     = 80
BATCH_SIZE = 64
LR         = 1e-3
DROPOUT    = 0.5
EMBED_DIM  = 64
VAL_SPLIT  = 0.15
SEED       = 42
# ──────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)


class EEGDataset(Dataset):
    """Wrap raw epochs.npz into a PyTorch Dataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X       = torch.from_numpy(X)      # (N, 64, 320)
        self.y       = torch.from_numpy(y)      # (N,) int64
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]                         # (64, 320)
        if self.augment:
            # Gaussian noise augmentation
            x = x + torch.randn_like(x) * 0.05
            # Random temporal shift (up to 10 samples)
            shift = np.random.randint(-10, 10)
            x = torch.roll(x, shift, dims=-1)
        return x.unsqueeze(0), self.y[idx]      # (1, 64, 320), int64


def train():
    device = get_device()
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print(f"Data not found at {DATA_PATH}.")
        print("Run:  /opt/anaconda3/envs/lerobot/bin/python download_eeg_data.py")
        raise SystemExit(1)

    # ── Load data ─────────────────────────────────────────────────────────────
    data = np.load(DATA_PATH)
    X, y = data["X"], data["y"]
    print(f"Loaded {len(X)} epochs  shape={X.shape}  classes={np.unique(y)}")

    n_channels, n_timepoints = X.shape[1], X.shape[2]
    n_classes = len(np.unique(y))
    counts = {int(c): int((y == c).sum()) for c in np.unique(y)}
    print(f"Class distribution: {counts}")

    # ── Datasets & loaders ────────────────────────────────────────────────────
    full_ds  = EEGDataset(X, y, augment=False)
    n_val    = int(len(full_ds) * VAL_SPLIT)
    n_train  = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(SEED))
    # Re-wrap train split with augmentation
    train_ds.dataset = EEGDataset(X, y, augment=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)

    print(f"Train: {n_train}  Val: {n_val}  Batch: {BATCH_SIZE}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EEGNet(
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        n_classes=n_classes,
        embed_dim=EMBED_DIM,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nEEGNet: {total_params/1e3:.1f}K parameters")

    # ── Training ──────────────────────────────────────────────────────────────
    # Class-balanced loss weights
    weights = torch.tensor(
        [n_train / (n_classes * max(counts.get(i, 1), 1)) for i in range(n_classes)],
        dtype=torch.float32, device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    log_rows = []

    print(f"\nTraining for {EPOCHS} epochs on {device} ...\n")

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            _, logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss    += loss.item() * len(yb)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total   += len(yb)

        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                _, logits = model(xb)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total   += len(yb)

        train_acc = train_correct / train_total * 100
        val_acc   = val_correct   / val_total   * 100
        avg_loss  = train_loss    / train_total

        log_rows.append({"epoch": epoch, "loss": avg_loss,
                         "train_acc": train_acc, "val_acc": val_acc})

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{EPOCHS} | "
                  f"loss={avg_loss:.4f} | "
                  f"train_acc={train_acc:.1f}% | "
                  f"val_acc={val_acc:.1f}%")

        # ── Save best ──────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     val_acc,
                "n_channels":  n_channels,
                "n_timepoints": n_timepoints,
                "n_classes":   n_classes,
                "embed_dim":   EMBED_DIM,
            }, os.path.join(OUT_DIR, "eeg_encoder.pt"))

    # ── Save encoder-only weights (backbone without classification head) ────
    best = torch.load(os.path.join(OUT_DIR, "eeg_encoder.pt"),
                      map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state"])

    # Strip head — save just block1, block2, embed
    backbone_state = {k: v for k, v in model.state_dict().items()
                      if not k.startswith("head.")}
    torch.save({
        "backbone_state": backbone_state,
        "n_channels":     n_channels,
        "n_timepoints":   n_timepoints,
        "embed_dim":      EMBED_DIM,
        "best_val_acc":   best_val_acc,
        "classes": {
            "0": "left_fist_MI",
            "1": "right_fist_MI",
            "2": "both_fists_MI",
            "3": "both_feet_MI",
        },
    }, os.path.join(OUT_DIR, "encoder_only.pt"))

    # Save training log
    with open(os.path.join(OUT_DIR, "train_log.json"), "w") as f:
        json.dump(log_rows, f, indent=2)

    print(f"\n=== EEG Encoder Training Complete ===")
    print(f"  Best val acc  : {best_val_acc:.1f}%  (4-class MI, chance=25%)")
    print(f"  Checkpoint    : {OUT_DIR}/eeg_encoder.pt")
    print(f"  Backbone      : {OUT_DIR}/encoder_only.pt")
    print(f"\nNext: run train_smolvla_eeg.py to integrate into SmolVLA")


if __name__ == "__main__":
    train()
