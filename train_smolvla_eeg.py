#!/usr/bin/env python
"""
Fine-tune SmolVLA + EEG encoder on LIBERO demonstration data.

Architecture:
  - SmolVLA (from existing checkpoint) — VLM backbone frozen, action expert trainable
  - EEGNet encoder (pretrained on PhysioNet MI) — frozen
  - EEG projection layer (new, trainable): Linear(embed_dim=64, state_dim=14)

Injection: the EEG projection is *added* to the normalized robot state before
it enters SmolVLA's state_proj layer.  This keeps the SmolVLA weight shapes
unchanged (no architecture surgery), while allowing EEG to modulate the
effective robot-state representation.

  augmented_state = robot_state_norm + eeg_proj(eeg_embedding)

During training, EEG epochs are randomly sampled from the PhysioNet dataset
(no ground-truth pairing exists) and dropped out 50% of the time so the model
continues to work without EEG at inference.

At evaluation time, specific MI classes (left, right, both fists, both feet)
can be selected to test directional conditioning.

Usage:
    cd /Users/r/Projects/SmolVLA_cl
    PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python train_smolvla_eeg.py

Environment overrides:
    SUITE         libero_spatial
    STEPS         1000       fine-tuning steps on top of existing checkpoint
    LR            5e-5       learning rate for action expert + EEG proj
    EEG_DROPOUT   0.5        probability of zeroing EEG signal per sample
    SAVE_EVERY    200
"""

import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

LIBERO_PATH = "/Users/r/LIBERO"
if LIBERO_PATH not in sys.path:
    sys.path.insert(0, LIBERO_PATH)

import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_STATE

from libero_smolvla_config import make_libero_smolvla_config, STATE_DIM
from dataset_libero import LiberoDataset, collate_fn
from eeg_encoder import EEGNet
from utils import get_device

# ── Config ────────────────────────────────────────────────────────────────────
SUITE             = os.environ.get("SUITE", "libero_spatial")
STEPS             = int(os.environ.get("STEPS", "1000"))
LR                = float(os.environ.get("LR", "5e-5"))
BATCH_SIZE        = int(os.environ.get("BATCH_SIZE", "4"))
GRAD_ACCUM        = int(os.environ.get("GRAD_ACCUM", "4"))
SAVE_EVERY        = int(os.environ.get("SAVE_EVERY", "200"))
SYNTHETIC_PAIRING = bool(int(os.environ.get("SYNTHETIC_PAIRING", "0")))

# When SYNTHETIC_PAIRING=1, force EEG dropout to 0 so the model HAS to use EEG
DEFAULT_DROPOUT = "0.0" if SYNTHETIC_PAIRING else "0.5"
EEG_DROPOUT     = float(os.environ.get("EEG_DROPOUT", DEFAULT_DROPOUT))

# Use a separate output dir for synthetic mode so it doesn't overwrite the random run
RUN_TAG = "_eeg_synth" if SYNTHETIC_PAIRING else "_eeg"

EEG_DATA_PATH   = "./data/eeg_physionet/epochs.npz"
EEG_ENCODER_PT  = "./checkpoints/eeg_encoder/encoder_only.pt"
BASE_CHECKPOINT = f"./checkpoints/{SUITE}/policy_final.pt"
OUTPUT_DIR      = f"./checkpoints/{SUITE}{RUN_TAG}"
LOG_PATH        = f"{OUTPUT_DIR}/train_log.jsonl"
STATS_PATH      = f"./checkpoints/{SUITE}/dataset_stats.pt"

VLM_NAME       = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
CHUNK_SIZE     = 50
MAX_GRAD_NORM  = 10.0
EMBED_DIM      = 64     # must match EEGNet embed_dim
MAX_DEMOS      = None
# ──────────────────────────────────────────────────────────────────────────────


# ── EEG injection wrapper ─────────────────────────────────────────────────────

class SmolVLAWithEEG(nn.Module):
    """
    Thin wrapper around SmolVLAPolicy that injects an EEG signal.

    At each forward pass:
      1. Encode a (B, 1, C, T) EEG tensor → (B, embed_dim) embedding
      2. Project to state_dim and add to batch[OBS_STATE]
      3. Delegate to base_policy.forward / select_action
    """

    def __init__(self, base_policy: SmolVLAPolicy, eeg_encoder: EEGNet,
                 state_dim: int = STATE_DIM, eeg_dropout: float = 0.5):
        super().__init__()
        self.base_policy = base_policy
        self.eeg_encoder = eeg_encoder
        self.eeg_proj    = nn.Linear(EMBED_DIM, state_dim)
        self.eeg_dropout = eeg_dropout

    def _inject_eeg(self, batch: dict, eeg: torch.Tensor | None,
                    training: bool = True) -> dict:
        """Add EEG projection to the state in-place (copy)."""
        batch = dict(batch)           # shallow copy — don't mutate caller's dict
        if eeg is None:
            return batch

        with torch.set_grad_enabled(training and
                                    any(p.requires_grad for p in self.eeg_proj.parameters())):
            emb = self.eeg_encoder.encode(eeg)   # (B, embed_dim)
            proj = self.eeg_proj(emb)             # (B, state_dim)

        # 50% dropout per sample during training
        if training and self.eeg_dropout > 0:
            mask = (torch.rand(len(proj), 1, device=proj.device) > self.eeg_dropout)
            proj = proj * mask.float()

        batch[OBS_STATE] = batch[OBS_STATE] + proj
        return batch

    def forward(self, batch: dict, eeg: torch.Tensor | None = None):
        batch = self._inject_eeg(batch, eeg, training=self.training)
        return self.base_policy.forward(batch)

    def select_action(self, batch: dict, eeg: torch.Tensor | None = None):
        batch = self._inject_eeg(batch, eeg, training=False)
        return self.base_policy.select_action(batch)

    def reset(self):
        self.base_policy.reset()


# ── EEG epoch sampler ─────────────────────────────────────────────────────────

class EEGSampler:
    """
    Random sampler over the PhysioNet epoch pool.
    Call .sample(batch_size, device) to get a (B, 1, C, T) tensor.
    Can also restrict to a specific class for eval.
    """
    def __init__(self, data_path: str):
        data = np.load(data_path)
        self.X = torch.from_numpy(data["X"])   # (N, 64, 320)
        self.y = torch.from_numpy(data["y"])   # (N,)
        self.class_indices = {
            int(c): (self.y == c).nonzero(as_tuple=True)[0]
            for c in self.y.unique()
        }
        print(f"EEGSampler: {len(self.X)} epochs, "
              f"classes {sorted(self.class_indices.keys())}")

    def sample(self, batch_size: int, device: str,
               cls: int | None = None) -> torch.Tensor:
        """Return (B, 1, 64, 320) float32 on device.
        If cls is given, all returned epochs come from that single class."""
        if cls is not None and cls in self.class_indices:
            pool = self.class_indices[cls]
        else:
            pool = torch.arange(len(self.X))
        idx = pool[torch.randint(len(pool), (batch_size,))]
        return self.X[idx].unsqueeze(1).to(device)   # (B, 1, C, T)

    def sample_per_class(self, classes: list[int], device: str) -> torch.Tensor:
        """Return (B, 1, 64, 320) where element i comes from class[i]."""
        out = []
        for cls in classes:
            if cls is not None and cls in self.class_indices:
                pool = self.class_indices[cls]
            else:
                pool = torch.arange(len(self.X))
            idx = pool[torch.randint(len(pool), (1,))].item()
            out.append(self.X[idx])
        return torch.stack(out).unsqueeze(1).to(device)


# ── Action → EEG class mapping (synthetic pairing) ────────────────────────────

def action_to_eeg_class(action_raw: torch.Tensor) -> int:
    """
    Deterministic mapping from a 7-D un-normalized action vector → EEG class.

    Class layout:
      0 — left_fist   : dominant -x motion (delta_x < 0, no gripper close)
      1 — right_fist  : dominant +x motion (delta_x > 0, no gripper close)
      2 — both_fists  : gripper closing OR dominant -y motion
      3 — both_feet   : dominant +y motion (forward)
    """
    dx, dy, _, _, _, _, gripper = action_raw.tolist()

    # Gripper close has highest priority — most distinctive intent
    if gripper < -0.3:
        return 2

    # Dominant translation axis
    if abs(dx) >= abs(dy):
        return 0 if dx < 0 else 1
    else:
        return 3 if dy > 0 else 2


def batch_actions_to_classes(action_batch: torch.Tensor,
                             a_mean: torch.Tensor,
                             a_std:  torch.Tensor) -> list[int]:
    """
    action_batch: (B, chunk_size, 7) — normalized actions from the dataset
    Returns list of B class indices, computed from the FIRST action of each chunk.
    """
    first   = action_batch[:, 0, :]                       # (B, 7) normalized
    raw     = first * a_std + a_mean                      # de-normalize
    return [action_to_eeg_class(raw[i]) for i in range(raw.shape[0])]


# ── Utilities ─────────────────────────────────────────────────────────────────

def batch_to_device(batch: dict, device: str) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def log_step(path: str, entry: dict):
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    device = get_device()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mode_str = "SYNTHETIC PAIRING (controllability test)" if SYNTHETIC_PAIRING else "RANDOM EEG (pipeline test)"
    print(f"=== SmolVLA + EEG Fine-Tuning ===")
    print(f"Mode           : {mode_str}")
    print(f"Suite          : {SUITE}")
    print(f"Steps          : {STEPS}  (effective batch = {BATCH_SIZE}×{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM})")
    print(f"LR             : {LR}")
    print(f"EEG dropout    : {EEG_DROPOUT:.0%}")
    print(f"Device         : {device}")
    print(f"Base checkpoint: {BASE_CHECKPOINT}")
    print(f"EEG encoder    : {EEG_ENCODER_PT}")
    print(f"Output         : {OUTPUT_DIR}")

    for path, name in [(BASE_CHECKPOINT, "SmolVLA checkpoint"),
                       (EEG_ENCODER_PT,  "EEG encoder"),
                       (STATS_PATH,      "Dataset stats"),
                       (EEG_DATA_PATH,   "EEG epochs")]:
        if not os.path.exists(path):
            print(f"\n  Missing {name}: {path}")
            if path == EEG_DATA_PATH:
                print("  Run: python download_eeg_data.py")
            if path == EEG_ENCODER_PT:
                print("  Run: python train_eeg_encoder.py")
            raise SystemExit(1)

    # ── Load stats ────────────────────────────────────────────────────────────
    stats = torch.load(STATS_PATH, map_location="cpu", weights_only=False)

    # ── Build dataloader ──────────────────────────────────────────────────────
    print("\n[1/4] Building dataloader ...")
    tokenizer = AutoTokenizer.from_pretrained(VLM_NAME)
    dataset   = LiberoDataset(
        suite=SUITE, chunk_size=CHUNK_SIZE, tokenizer=tokenizer,
        stats=stats, max_token_len=48, max_demos=MAX_DEMOS,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, collate_fn=collate_fn,
                        drop_last=True, pin_memory=False)
    loader_iter = iter(loader)

    # ── Load base SmolVLA ─────────────────────────────────────────────────────
    print("\n[2/4] Loading SmolVLA from checkpoint ...")
    cfg = make_libero_smolvla_config(device)
    cfg.load_vlm_weights = False
    base_policy = SmolVLAPolicy(cfg).to(device)
    state_dict  = torch.load(BASE_CHECKPOINT, map_location=device, weights_only=False)
    # policy_final.pt is weights-only; step_*.pt has a "policy_state" key
    if isinstance(state_dict, dict) and "policy_state" in state_dict:
        state_dict = state_dict["policy_state"]
    base_policy.load_state_dict(state_dict)
    print("  SmolVLA loaded OK")

    # ── Load pretrained EEG encoder (frozen) ──────────────────────────────────
    print("\n[3/4] Loading pretrained EEG encoder ...")
    enc_ckpt = torch.load(EEG_ENCODER_PT, map_location="cpu", weights_only=False)
    eeg_enc  = EEGNet(
        n_channels=64, n_timepoints=320,
        n_classes=4, embed_dim=EMBED_DIM,
    )
    eeg_enc.load_state_dict(enc_ckpt["backbone_state"], strict=False)
    eeg_enc = eeg_enc.to(device)
    for p in eeg_enc.parameters():
        p.requires_grad = False
    print(f"  EEG encoder loaded (val_acc={enc_ckpt.get('best_val_acc', '?'):.1f}%)")

    # ── Build combined model ───────────────────────────────────────────────────
    model = SmolVLAWithEEG(base_policy, eeg_enc,
                           state_dim=STATE_DIM, eeg_dropout=EEG_DROPOUT)
    model = model.to(device)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\n  Trainable params: {trainable/1e6:.2f}M / {total/1e6:.1f}M total")
    print(f"  (EEG proj: {sum(p.numel() for p in model.eeg_proj.parameters())/1e3:.1f}K new params)")

    # ── EEG sampler ───────────────────────────────────────────────────────────
    print("\n[4/4] Building EEG sampler ...")
    eeg_sampler = EEGSampler(EEG_DATA_PATH)

    # Pre-compute action denorm constants for synthetic pairing
    a_mean = stats["action"]["mean"].to(device)
    a_std  = stats["action"]["std"].to(device)

    # Stats for class distribution diagnostic
    class_count = {0: 0, 1: 0, 2: 0, 3: 0}

    # ── Optimizer ─────────────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, betas=(0.9, 0.95),
                                  weight_decay=1e-10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"\nFine-tuning for {STEPS} steps ...")
    t_start    = time.time()
    loss_window = []
    optimizer.zero_grad()

    for step in range(STEPS):
        # Get next LIBERO batch
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        batch = batch_to_device(batch, device)

        # Sample EEG epochs
        if SYNTHETIC_PAIRING:
            # Deterministic pairing: class chosen from each sample's first action
            classes = batch_actions_to_classes(batch["action"], a_mean, a_std)
            for c in classes:
                class_count[c] = class_count.get(c, 0) + 1
            eeg = eeg_sampler.sample_per_class(classes, device)
        else:
            # Random — no semantic correspondence
            eeg = eeg_sampler.sample(BATCH_SIZE, device)

        # Forward
        model.train()
        loss, loss_dict = model(batch, eeg=eeg)
        loss = loss / GRAD_ACCUM
        loss.backward()

        loss_window.append(loss_dict["loss"])
        if len(loss_window) > 50:
            loss_window.pop(0)

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Logging
        if step % 20 == 0 or step == STEPS - 1:
            elapsed = time.time() - t_start
            lr_now  = optimizer.param_groups[0]["lr"]
            avg_loss = np.mean(loss_window)
            sps      = (step + 1) / max(elapsed, 1e-6)
            extra = ""
            if SYNTHETIC_PAIRING and sum(class_count.values()) > 0:
                tot = sum(class_count.values())
                pct = {k: v/tot*100 for k,v in class_count.items()}
                extra = f" | classes(L/R/F/Fwd)={pct[0]:.0f}/{pct[1]:.0f}/{pct[2]:.0f}/{pct[3]:.0f}%"
            print(f"  step {step:4d}/{STEPS} | loss={avg_loss:.4f} | "
                  f"lr={lr_now:.2e} | {sps:.1f} steps/s | "
                  f"ETA {(STEPS-step-1)/max(sps,1e-6)/60:.1f}min{extra}")
            log_step(LOG_PATH, {
                "step": step, "loss": avg_loss, "lr": lr_now, "elapsed": elapsed,
            })

        # Checkpoint
        if (step + 1) % SAVE_EVERY == 0 or step == STEPS - 1:
            ckpt_path = os.path.join(OUTPUT_DIR, f"step_{step+1:06d}.pt")
            torch.save({
                "step":              step + 1,
                "eeg_proj_state":    model.eeg_proj.state_dict(),
                "policy_state":      base_policy.state_dict(),
                "optimizer_state":   optimizer.state_dict(),
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    elapsed_total = time.time() - t_start
    print(f"\n=== EEG Fine-Tuning Complete ===")
    print(f"  Time         : {elapsed_total/60:.1f} min")
    print(f"  Final loss   : {np.mean(loss_window):.4f}")
    print(f"  Output       : {OUTPUT_DIR}")
    print(f"\nRun evaluation:")
    print(f"  python eval_openloop_eeg.py")


if __name__ == "__main__":
    train()
