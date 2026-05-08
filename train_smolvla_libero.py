#!/usr/bin/env python
"""
Fine-tune SmolVLA on LIBERO demonstration data (Mac, Apple MPS).

Strategy
--------
- Freeze VLM backbone (SmolVLM2-500M), only train the action expert (~50M params).
- Load real dataset stats so normalization matches the demo distribution.
- Flow-matching loss (built into SmolVLAPolicy.forward).
- Gradient accumulation to simulate batch_size=16 with physical batch_size=4.
- Save checkpoints + stats every N steps; resume from last checkpoint.

Usage
-----
    cd /Users/r/Projects/SmolVLA_cl
    PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python train_smolvla_libero.py

Environment overrides:
    SUITE         libero_spatial (default)
    STEPS         2000           total gradient steps
    BATCH_SIZE    4              physical batch per step
    GRAD_ACCUM    4              accumulation steps  (effective batch = 4*4=16)
    LR            1e-4
    SAVE_EVERY    200            checkpoint every N steps
    RESUME        1              set to 0 to ignore existing checkpoint
    MAX_DEMOS     50             demos per HDF5 file (None=all)
    LOAD_VLM      1              download SmolVLM2-500M backbone (0=random init)
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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import (
    OBS_LANGUAGE_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_STATE,
    ACTION,
)

from libero_smolvla_config import make_libero_smolvla_config, STATE_DIM, ACTION_DIM
from dataset_libero import LiberoDataset, compute_dataset_stats, collate_fn
from utils import get_device

# ── Hyperparameters ───────────────────────────────────────────────────────────
SUITE      = os.environ.get("SUITE", "libero_spatial")
STEPS      = int(os.environ.get("STEPS", "2000"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
LR         = float(os.environ.get("LR", "1e-4"))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "200"))
RESUME     = bool(int(os.environ.get("RESUME", "1")))
MAX_DEMOS  = os.environ.get("MAX_DEMOS")
MAX_DEMOS  = int(MAX_DEMOS) if MAX_DEMOS else None
LOAD_VLM   = bool(int(os.environ.get("LOAD_VLM", "1")))
RUN_NAME   = os.environ.get("RUN_NAME", SUITE)   # separate output dir without changing SUITE

OUTPUT_DIR = f"./checkpoints/{RUN_NAME}"
STATS_PATH = f"./checkpoints/{RUN_NAME}/dataset_stats.pt"
LOG_PATH   = f"./checkpoints/{RUN_NAME}/train_log.jsonl"

DATASET_DIR  = f"/Users/r/LIBERO/libero/datasets/{SUITE}"
VLM_NAME     = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
CHUNK_SIZE   = 50
MAX_GRAD_NORM = 10.0
# ──────────────────────────────────────────────────────────────────────────────


def get_or_compute_stats(hdf5_paths: list[str]) -> dict:
    """Load cached stats or compute from scratch."""
    if os.path.exists(STATS_PATH):
        print(f"Loading cached dataset stats: {STATS_PATH}")
        stats = torch.load(STATS_PATH, weights_only=False)
        return stats

    print("Computing dataset statistics (one-time) ...")
    t0 = time.time()
    stats = compute_dataset_stats(hdf5_paths, max_demos_per_file=MAX_DEMOS or 50)
    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    torch.save(stats, STATS_PATH)
    print(f"Stats computed in {time.time()-t0:.1f}s  →  {STATS_PATH}")

    print("\nDataset statistics:")
    for key in ("observation.state", "action"):
        mean = stats[key]["mean"].numpy()
        std  = stats[key]["std"].numpy()
        print(f"  {key}  mean={mean.round(3)}  std={std.round(3)}")
    return stats


def build_policy(device: str, stats: dict) -> SmolVLAPolicy:
    """Build SmolVLA with LIBERO config; freeze VLM, train expert only."""
    cfg = make_libero_smolvla_config(device)
    cfg.load_vlm_weights = LOAD_VLM

    policy = SmolVLAPolicy(cfg)
    policy = policy.to(device)
    policy.train()

    # Count trainable params
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in policy.parameters())
    print(f"  Trainable params: {trainable/1e6:.1f}M / {total/1e6:.1f}M total")
    print(f"  (VLM frozen={cfg.freeze_vision_encoder}, expert_only={cfg.train_expert_only})")

    return policy


def build_dataloader(stats: dict, tokenizer) -> DataLoader:
    dataset = LiberoDataset(
        suite=SUITE,
        chunk_size=CHUNK_SIZE,
        tokenizer=tokenizer,
        stats=stats,
        max_token_len=48,
        max_demos=MAX_DEMOS,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,     # 0 for MPS compatibility
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=False,  # MPS doesn't use pinned memory
    )


def batch_to_device(batch: dict, device: str) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def find_last_checkpoint(output_dir: str) -> str | None:
    ckpts = sorted(glob.glob(os.path.join(output_dir, "step_*.pt")))
    return ckpts[-1] if ckpts else None


def save_checkpoint(policy, optimizer, scheduler, step: int, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"step_{step:06d}.pt")
    torch.save({
        "step":            step,
        "policy_state":    policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
    }, path)
    print(f"  Checkpoint saved: {path}")


def log_step(log_path: str, entry: dict) -> None:
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def train():
    device = get_device()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"=== SmolVLA Fine-Tuning on LIBERO ===")
    print(f"Suite        : {SUITE}")
    print(f"Steps        : {STEPS}  (effective batch = {BATCH_SIZE}×{GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM})")
    print(f"LR           : {LR}")
    print(f"Device       : {device}")
    print(f"Load VLM     : {LOAD_VLM}  (set LOAD_VLM=0 for faster startup, random VLM)")
    print(f"Output       : {OUTPUT_DIR}")
    print(f"Est. time    : ~{STEPS * 1.9 / 3600:.1f}h  (~1.9s/step on M5 MPS, expert-only)")

    # ── Data ──────────────────────────────────────────────────────────────────
    hdf5_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "*.hdf5")))
    if not hdf5_paths:
        raise FileNotFoundError(f"No HDF5 files found in {DATASET_DIR}")
    print(f"\nFound {len(hdf5_paths)} task HDF5 files")

    print("\n[1/4] Computing/loading dataset statistics ...")
    stats = get_or_compute_stats(hdf5_paths)

    print("\n[2/4] Building dataloader ...")
    tokenizer = AutoTokenizer.from_pretrained(VLM_NAME)
    loader    = build_dataloader(stats, tokenizer)
    loader_iter = iter(loader)

    # ── Policy ────────────────────────────────────────────────────────────────
    print("\n[3/4] Building policy ...")
    policy = build_policy(device, stats)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
    )

    # Cosine decay with warmup
    warmup_steps = min(100, STEPS // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, STEPS - warmup_steps)
        return max(0.025, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_step = 0
    if RESUME:
        last_ckpt = find_last_checkpoint(OUTPUT_DIR)
        if last_ckpt:
            print(f"\nResuming from: {last_ckpt}")
            ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
            policy.load_state_dict(ckpt["policy_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            if ckpt.get("scheduler_state") and scheduler:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            start_step = ckpt["step"]
            print(f"  Resumed at step {start_step}")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n[4/4] Training from step {start_step} to {STEPS} ...")

    loss_window = []
    t_start = time.time()
    optimizer.zero_grad()

    for step in range(start_step, STEPS):
        # Get next batch (cycle through dataset)
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch = batch_to_device(batch, device)

        # Forward pass (flow-matching loss)
        policy.train()
        loss, loss_dict = policy.forward(batch)

        # Scale loss for gradient accumulation
        loss = loss / GRAD_ACCUM
        loss.backward()

        loss_window.append(loss_dict["loss"])
        if len(loss_window) > 50:
            loss_window.pop(0)

        # Gradient step every GRAD_ACCUM micro-batches
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Logging
        if step % 20 == 0 or step == STEPS - 1:
            elapsed = time.time() - t_start
            lr_now  = optimizer.param_groups[0]["lr"]
            avg_loss = np.mean(loss_window) if loss_window else float("nan")
            steps_per_sec = (step - start_step + 1) / max(elapsed, 1e-6)
            eta_s = (STEPS - step - 1) / max(steps_per_sec, 1e-6)

            print(
                f"  step {step:5d}/{STEPS} | loss={avg_loss:.4f} | "
                f"lr={lr_now:.2e} | {steps_per_sec:.1f} steps/s | "
                f"ETA {eta_s/60:.1f}min"
            )
            log_step(LOG_PATH, {
                "step": step, "loss": avg_loss, "lr": lr_now,
                "elapsed": elapsed,
            })

        # Checkpointing
        if (step + 1) % SAVE_EVERY == 0 or step == STEPS - 1:
            save_checkpoint(policy, optimizer, scheduler, step + 1, OUTPUT_DIR)

    # ── Save final policy (weights only, for inference) ───────────────────────
    final_path = os.path.join(OUTPUT_DIR, "policy_final.pt")
    torch.save(policy.state_dict(), final_path)

    # Save stats alongside policy for inference
    torch.save(stats, STATS_PATH)

    elapsed_total = time.time() - t_start
    print(f"\n=== Training complete ===")
    print(f"  Total time : {elapsed_total/60:.1f} min")
    print(f"  Final loss : {np.mean(loss_window):.4f}")
    print(f"  Checkpoint : {OUTPUT_DIR}")
    print(f"  Stats      : {STATS_PATH}")
    print(f"\nTo evaluate the trained policy:")
    print(f"  SUITE={SUITE} MODEL=trained PYTHONPATH=/Users/r/LIBERO \\")
    print(f"  /opt/anaconda3/envs/lerobot/bin/python eval_openloop.py")


if __name__ == "__main__":
    train()
