#!/usr/bin/env python
"""
Stage 5: Token-level VLM injection — the architectural fix.

Instead of adding EEG to robot state at the very end (Stages 2/3), this projects
the EEG embedding into SmolVLM2's hidden dimension (960) and injects it as a
TOKEN inside the language input sequence. The VLM's transformer attends to it
alongside image patches and language tokens.

Mechanism:
  EEG (64ch×320) → EEGNet → 64-dim embedding
                          → Linear → GELU → Linear → LayerNorm → 960-dim token
                          → replaces position 0 of language embeddings via a
                            forward hook on vlm.get_input_embeddings()
                          → flows through 16 VLM transformer layers via attention
                          → emerges as part of the context tokens that the
                            action expert reads

This requires gradients to flow through the (frozen) VLM back to the projection
layer — increasing memory ~3-5× vs additive injection but still fitting on M5.

We start from the synthetic-pairing recipe (deterministic action→EEG mapping,
0% dropout) so the controllability test is directly comparable to Stage 3.

Usage:
    SUITE=libero_spatial PYTHONPATH=/Users/r/LIBERO \
        /opt/anaconda3/envs/lerobot/bin/python -u train_smolvla_eeg_token.py
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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from libero_smolvla_config import make_libero_smolvla_config
from dataset_libero import LiberoDataset, collate_fn
from eeg_encoder import EEGNet
from train_smolvla_eeg import (
    EEGSampler, batch_actions_to_classes, batch_to_device, log_step,
    action_to_eeg_class,
)
from utils import get_device


# ── Class-balanced sample weights ────────────────────────────────────────────

def compute_sample_weights(dataset: LiberoDataset, stats: dict) -> torch.Tensor:
    """
    For each (demo, t) sample in the dataset, compute its action class
    (via action_to_eeg_class on the first action of the chunk) and assign
    a sampling weight = 1 / class_count.  Returned tensor is fed to a
    WeightedRandomSampler so each class is drawn equally often.

    This is run once at startup (~1-2 min for 62k samples) by reading
    actions directly from the underlying HDF5 files.
    """
    import h5py
    a_mean = stats["action"]["mean"].numpy()
    a_std  = stats["action"]["std"].numpy()

    # Walk the dataset's flat index, opening each demo HDF5 once
    classes = np.zeros(len(dataset), dtype=np.int64)
    last_path = None
    cur_actions = None
    print("  Computing per-sample class for balanced sampling ...", flush=True)
    for idx, (demo_idx, t) in enumerate(dataset._flat):
        path, demo_key, _, _ = dataset._index[demo_idx]
        if path != last_path:
            last_path = path
            cur_h5 = h5py.File(path, "r")
        first = cur_h5["data"][demo_key]["actions"][t].astype(np.float32)
        # action_to_eeg_class expects raw (un-normalized) action
        classes[idx] = action_to_eeg_class(torch.from_numpy(first))

    counts = np.bincount(classes, minlength=4).astype(np.float64)
    print(f"  Action-class counts: L={int(counts[0])} R={int(counts[1])} "
          f"F={int(counts[2])} Fwd={int(counts[3])}")

    # Inverse-frequency weights (each class contributes equally)
    inv = 1.0 / np.maximum(counts, 1)
    weights = inv[classes]
    return torch.from_numpy(weights), classes

# ── Config ────────────────────────────────────────────────────────────────────
SUITE             = os.environ.get("SUITE", "libero_spatial")
STEPS             = int(os.environ.get("STEPS", "1000"))
LR                = float(os.environ.get("LR", "5e-5"))
BATCH_SIZE        = int(os.environ.get("BATCH_SIZE", "4"))
GRAD_ACCUM        = int(os.environ.get("GRAD_ACCUM", "4"))
SAVE_EVERY        = int(os.environ.get("SAVE_EVERY", "200"))
SYNTHETIC_PAIRING = bool(int(os.environ.get("SYNTHETIC_PAIRING", "1")))
CLASS_BALANCED    = bool(int(os.environ.get("CLASS_BALANCED", "0")))
AUX_LOSS_WEIGHT   = float(os.environ.get("AUX_LOSS_WEIGHT", "0.0"))   # 0 = disabled
ENCODER_ARCH      = os.environ.get("ENCODER_ARCH", "eegnet")   # "eegnet" | "atcnet"
ENCODER_CKPT      = os.environ.get("ENCODER_CKPT", "")        # override path
RUN_TAG           = os.environ.get("RUN_TAG", "")     # extra suffix for output dir
EEG_DROPOUT       = float(os.environ.get("EEG_DROPOUT",
                                          "0.0" if SYNTHETIC_PAIRING else "0.5"))

EEG_DATA_PATH    = os.environ.get("EEG_DATA_PATH", "./data/eeg_physionet/epochs.npz")
DEFAULT_ENC_PT   = ("./checkpoints/eeg_encoder_atcnet/encoder_only.pt"
                   if ENCODER_ARCH == "atcnet"
                   else "./checkpoints/eeg_encoder/encoder_only.pt")
EEG_ENCODER_PT   = ENCODER_CKPT or DEFAULT_ENC_PT
BASE_CHECKPOINT  = f"./checkpoints/{SUITE}/policy_final.pt"
STATS_PATH       = f"./checkpoints/{SUITE}/dataset_stats.pt"
OUTPUT_DIR       = f"./checkpoints/{SUITE}_eeg_token{RUN_TAG}"
LOG_PATH         = f"{OUTPUT_DIR}/train_log.jsonl"

VLM_NAME         = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VLM_HIDDEN_DIM   = 960          # SmolVLM2-500M text hidden size
EEG_EMBED_DIM    = 64           # EEGNet output
CHUNK_SIZE       = 50
MAX_GRAD_NORM    = 10.0
MAX_DEMOS        = None
# ──────────────────────────────────────────────────────────────────────────────


# ── Token-level VLM injection wrapper ────────────────────────────────────────

class SmolVLATokenLevelEEG(nn.Module):
    """
    Inject EEG into the VLM's input token stream by replacing position 0 of the
    text embeddings via a forward hook on the embedding layer.

    The VLM is frozen; gradients flow through its activations back to eeg_proj.
    """

    def __init__(
        self,
        base_policy: SmolVLAPolicy,
        eeg_encoder,                          # EEGNet or ATCNet (same API)
        vlm_hidden_dim: int = VLM_HIDDEN_DIM,
        eeg_dropout: float = 0.0,
        n_aux_classes: int = 4,
    ):
        super().__init__()
        self.base_policy = base_policy
        self.eeg_encoder = eeg_encoder
        self.vlm_hidden_dim = vlm_hidden_dim
        self.eeg_dropout = eeg_dropout

        # EEG embedding (64) → VLM hidden (960)
        self.eeg_proj = nn.Sequential(
            nn.Linear(EEG_EMBED_DIM, 256),
            nn.GELU(),
            nn.Linear(256, vlm_hidden_dim),
            nn.LayerNorm(vlm_hidden_dim),
        )

        # Auxiliary classification head: projected EEG token → 4-class logits
        # Used only during training to keep the token discriminative across
        # the 4 EEG classes after projection into the VLM's hidden space.
        self.aux_classifier = nn.Linear(vlm_hidden_dim, n_aux_classes)

        # Per-forward state — set before forward, read by the hook
        self._current_eeg_token: torch.Tensor | None = None

        # Register hook on the VLM's text embedding layer
        embed_layer = base_policy.model.vlm_with_expert.vlm.get_input_embeddings()
        self._hook = embed_layer.register_forward_hook(self._inject_hook)

    def _inject_hook(self, module, inputs, output):
        """
        After token embedding lookup, replace position 0 with the EEG-derived
        token for every sample in the batch. The output shape is (B, T, H).
        """
        if self._current_eeg_token is None:
            return output
        # Only inject when output looks like a language sequence
        # (B matches our EEG batch and T >= 1)
        if output.dim() != 3 or output.shape[0] != self._current_eeg_token.shape[0]:
            return output
        out = output.clone()
        out[:, 0, :] = self._current_eeg_token.to(out.dtype)
        return out

    def _compute_eeg_token(self, eeg: torch.Tensor) -> torch.Tensor:
        """EEG raw → projected 960-dim token, with optional dropout."""
        emb   = self.eeg_encoder.encode(eeg)        # (B, 64)
        token = self.eeg_proj(emb)                  # (B, 960)
        if self.training and self.eeg_dropout > 0:
            mask = (torch.rand(len(token), 1, device=token.device)
                    > self.eeg_dropout).float()
            token = token * mask
        return token

    def forward(self, batch: dict, eeg: torch.Tensor | None = None,
                aux_targets: torch.Tensor | None = None,
                aux_weight: float = 0.0):
        """
        Returns (loss, loss_dict) like base_policy.forward.
        If aux_targets and aux_weight > 0, adds cross-entropy loss on the
        projected EEG token to keep it 4-class discriminative.
        """
        if eeg is not None:
            self._current_eeg_token = self._compute_eeg_token(eeg)
        else:
            self._current_eeg_token = None
        try:
            main_loss, loss_dict = self.base_policy.forward(batch)
        finally:
            saved_token = self._current_eeg_token
            self._current_eeg_token = None

        if aux_targets is not None and aux_weight > 0 and saved_token is not None:
            aux_logits = self.aux_classifier(saved_token)              # (B, 4)
            aux_loss = torch.nn.functional.cross_entropy(aux_logits, aux_targets)
            with torch.no_grad():
                aux_acc = (aux_logits.argmax(-1) == aux_targets).float().mean().item()
            loss_dict = dict(loss_dict)
            loss_dict["aux_loss"] = float(aux_loss.item())
            loss_dict["aux_acc"]  = aux_acc
            loss_dict["main_loss"] = float(main_loss.item())
            return main_loss + aux_weight * aux_loss, loss_dict

        return main_loss, loss_dict

    def select_action(self, batch: dict, eeg: torch.Tensor | None = None):
        if eeg is not None:
            self._current_eeg_token = self._compute_eeg_token(eeg)
        else:
            self._current_eeg_token = None
        try:
            return self.base_policy.select_action(batch)
        finally:
            self._current_eeg_token = None

    def reset(self):
        self.base_policy.reset()


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    device = get_device()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mode_str = ("SYNTHETIC PAIRING (controllability)" if SYNTHETIC_PAIRING
                else "RANDOM EEG (pipeline)")
    print(f"=== SmolVLA + EEG TOKEN-LEVEL VLM Injection ===")
    print(f"Mode           : {mode_str}")
    print(f"Suite          : {SUITE}")
    print(f"Steps          : {STEPS}  (effective batch = {BATCH_SIZE}×{GRAD_ACCUM})")
    print(f"LR             : {LR}")
    print(f"EEG dropout    : {EEG_DROPOUT:.0%}")
    print(f"Class-balanced : {CLASS_BALANCED}")
    print(f"Aux loss weight: {AUX_LOSS_WEIGHT}")
    print(f"Encoder arch   : {ENCODER_ARCH}")
    print(f"Encoder ckpt   : {EEG_ENCODER_PT}")
    print(f"VLM hidden dim : {VLM_HIDDEN_DIM}")
    print(f"Device         : {device}")
    print(f"Base ckpt      : {BASE_CHECKPOINT}")
    print(f"Output         : {OUTPUT_DIR}")

    for path, name in [(BASE_CHECKPOINT,  "SmolVLA checkpoint"),
                       (EEG_ENCODER_PT,   "EEG encoder"),
                       (STATS_PATH,       "Dataset stats"),
                       (EEG_DATA_PATH,    "EEG epochs")]:
        if not os.path.exists(path):
            print(f"\nMissing {name}: {path}")
            raise SystemExit(1)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1/4] Building dataloader ...")
    stats     = torch.load(STATS_PATH, map_location="cpu", weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(VLM_NAME)
    dataset   = LiberoDataset(suite=SUITE, chunk_size=CHUNK_SIZE,
                              tokenizer=tokenizer, stats=stats,
                              max_token_len=48, max_demos=MAX_DEMOS)

    if CLASS_BALANCED:
        weights, _ = compute_sample_weights(dataset, stats)
        # num_samples = STEPS * BATCH_SIZE * GRAD_ACCUM ensures coverage
        n_to_draw = STEPS * BATCH_SIZE * GRAD_ACCUM
        sampler = WeightedRandomSampler(weights, num_samples=n_to_draw, replacement=True)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            sampler=sampler, num_workers=0,
                            collate_fn=collate_fn, drop_last=True, pin_memory=False)
        print(f"  Class-balanced sampling enabled (drawing {n_to_draw} samples)")
    else:
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=0, collate_fn=collate_fn,
                            drop_last=True, pin_memory=False)
    loader_iter = iter(loader)

    # ── Base SmolVLA ──────────────────────────────────────────────────────────
    print("\n[2/4] Loading base SmolVLA ...")
    cfg = make_libero_smolvla_config(device)
    cfg.load_vlm_weights = False
    base_policy = SmolVLAPolicy(cfg).to(device)
    state_dict  = torch.load(BASE_CHECKPOINT, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and "policy_state" in state_dict:
        state_dict = state_dict["policy_state"]
    base_policy.load_state_dict(state_dict)
    print("  SmolVLA loaded OK")

    # ── EEG encoder (frozen) ─────────────────────────────────────────────────
    print(f"\n[3/4] Loading EEG encoder ({ENCODER_ARCH}) ...")
    enc_ckpt = torch.load(EEG_ENCODER_PT, map_location="cpu", weights_only=False)
    if ENCODER_ARCH == "atcnet":
        from eeg_encoder_atcnet import ATCNet
        eeg_enc = ATCNet(n_channels=64, n_timepoints=320,
                         n_classes=4, embed_dim=EEG_EMBED_DIM)
    else:
        eeg_enc = EEGNet(n_channels=64, n_timepoints=320,
                         n_classes=4, embed_dim=EEG_EMBED_DIM)
    eeg_enc.load_state_dict(enc_ckpt["backbone_state"], strict=False)
    eeg_enc = eeg_enc.to(device)
    for p in eeg_enc.parameters():
        p.requires_grad = False
    print(f"  EEG encoder val_acc={enc_ckpt.get('best_val_acc','?'):.1f}%")

    # ── Combined model ────────────────────────────────────────────────────────
    print("\n[4/4] Building combined model + sampler ...")
    model = SmolVLATokenLevelEEG(base_policy, eeg_enc,
                                 vlm_hidden_dim=VLM_HIDDEN_DIM,
                                 eeg_dropout=EEG_DROPOUT).to(device)
    model.train()

    eeg_proj_params  = sum(p.numel() for p in model.eeg_proj.parameters())
    trainable_total  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  EEG projection  : {eeg_proj_params/1e3:.1f}K new params")
    print(f"  Total trainable : {trainable_total/1e6:.2f}M")

    eeg_sampler = EEGSampler(EEG_DATA_PATH)

    a_mean = stats["action"]["mean"].to(device)
    a_std  = stats["action"]["std"].to(device)
    class_count = {0: 0, 1: 0, 2: 0, 3: 0}

    # ── Optimizer ─────────────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR,
                                  betas=(0.9, 0.95), weight_decay=1e-10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"\nFine-tuning for {STEPS} steps ...")
    t_start, loss_window = time.time(), []
    optimizer.zero_grad()

    for step in range(STEPS):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        batch = batch_to_device(batch, device)

        aux_targets = None
        if SYNTHETIC_PAIRING:
            classes = batch_actions_to_classes(batch["action"], a_mean, a_std)
            for c in classes:
                class_count[c] = class_count.get(c, 0) + 1
            eeg = eeg_sampler.sample_per_class(classes, device)
            if AUX_LOSS_WEIGHT > 0:
                aux_targets = torch.tensor(classes, device=device, dtype=torch.long)
        else:
            eeg = eeg_sampler.sample(BATCH_SIZE, device)

        loss, loss_dict = model(batch, eeg=eeg,
                                aux_targets=aux_targets,
                                aux_weight=AUX_LOSS_WEIGHT)
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

        if step % 20 == 0 or step == STEPS - 1:
            elapsed = time.time() - t_start
            sps = (step + 1) / max(elapsed, 1e-6)
            extra = ""
            if SYNTHETIC_PAIRING and sum(class_count.values()) > 0:
                tot = sum(class_count.values())
                pct = {k: v/tot*100 for k,v in class_count.items()}
                extra = f" | classes(L/R/F/Fwd)={pct[0]:.0f}/{pct[1]:.0f}/{pct[2]:.0f}/{pct[3]:.0f}%"
            aux_extra = ""
            if "aux_acc" in loss_dict:
                aux_extra = f" | aux_acc={loss_dict['aux_acc']*100:.0f}%"
            print(f"  step {step:4d}/{STEPS} | loss={np.mean(loss_window):.4f} | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} | {sps:.2f} steps/s | "
                  f"ETA {(STEPS-step-1)/max(sps,1e-6)/60:.1f}min{extra}{aux_extra}",
                  flush=True)
            log_step(LOG_PATH, {
                "step": step, "loss": float(np.mean(loss_window)),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "elapsed": elapsed,
            })

        if (step + 1) % SAVE_EVERY == 0 or step == STEPS - 1:
            ckpt = os.path.join(OUTPUT_DIR, f"step_{step+1:06d}.pt")
            torch.save({
                "step":              step + 1,
                "eeg_proj_state":    model.eeg_proj.state_dict(),
                "aux_classifier_state": model.aux_classifier.state_dict(),
                "policy_state":      base_policy.state_dict(),
                "optimizer_state":   optimizer.state_dict(),
                "vlm_hidden_dim":    VLM_HIDDEN_DIM,
                "encoder_arch":      ENCODER_ARCH,
            }, ckpt)
            print(f"  Checkpoint saved: {ckpt}", flush=True)

    elapsed_total = time.time() - t_start
    print(f"\n=== Token-level VLM injection training complete ===")
    print(f"  Time       : {elapsed_total/60:.1f} min")
    print(f"  Final loss : {np.mean(loss_window):.4f}")
    print(f"  Output     : {OUTPUT_DIR}")
    print(f"\nNext: PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python "
          f"eval_token_eeg.py")


if __name__ == "__main__":
    train()
