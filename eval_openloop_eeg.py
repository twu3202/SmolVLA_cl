#!/usr/bin/env python
"""
Open-loop evaluation of SmolVLA+EEG on LIBERO demos.

Tests 5 EEG conditions per task:
  - no_eeg        : zero EEG signal (baseline, should match SmolVLA)
  - left_fist     : imagined left fist MI
  - right_fist    : imagined right fist MI
  - both_fists    : imagined both fists (→ gripper close intent)
  - both_feet     : imagined both feet (→ forward motion intent)

Also compares against the plain SmolVLA baseline (no EEG at all).

Usage:
    cd /Users/r/Projects/SmolVLA_cl
    PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python eval_openloop_eeg.py
"""

import glob
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

LIBERO_PATH = "/Users/r/LIBERO"
if LIBERO_PATH not in sys.path:
    sys.path.insert(0, LIBERO_PATH)

import h5py
import numpy as np
import torch
from transformers import AutoTokenizer

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from libero_smolvla_config import (
    make_libero_smolvla_config,
    LIBERO_AGENTVIEW_KEY, LIBERO_WRIST_KEY, LIBERO_STATE_KEY,
)
from eeg_encoder import EEGNet
from train_smolvla_eeg import SmolVLAWithEEG, EEGSampler
from utils import get_device, obs_to_policy_batch

# ── Config ────────────────────────────────────────────────────────────────────
SUITE        = os.environ.get("SUITE", "libero_spatial")
MAX_DEMOS    = int(os.environ.get("MAX_DEMOS", "3"))
MAX_STEPS    = int(os.environ.get("MAX_STEPS", "20"))
EEG_CKPT     = os.environ.get("EEG_CKPT",
    f"./checkpoints/{SUITE}_eeg/step_001000.pt")
DATASET_DIR  = f"/Users/r/LIBERO/libero/datasets/{SUITE}"
EEG_DATA_PATH = "./data/eeg_physionet/epochs.npz"
EEG_ENC_PT    = "./checkpoints/eeg_encoder/encoder_only.pt"
STATS_PATH    = f"./checkpoints/{SUITE}/dataset_stats.pt"
OUTPUT_DIR    = "./eval_output"

EEG_CONDITIONS = {
    "no_eeg":     None,    # zero EEG — nulled out
    "left_fist":  0,
    "right_fist": 1,
    "both_fists": 2,
    "both_feet":  3,
}

ACTION_LABELS = ["delta_x", "delta_y", "delta_z",
                 "delta_roll", "delta_pitch", "delta_yaw", "gripper"]
# ──────────────────────────────────────────────────────────────────────────────


def build_state_from_obs(obs_grp, t: int) -> np.ndarray:
    ee_pos   = obs_grp["ee_pos"][t]
    ee_ori   = obs_grp["ee_ori"][t]
    joint    = obs_grp["joint_states"][t]
    quat     = np.array([ee_ori[0]/2, ee_ori[1]/2, ee_ori[2]/2, 1.0], dtype=np.float32)
    quat    /= np.linalg.norm(quat)
    return np.concatenate([ee_pos, quat, joint]).astype(np.float32)


def hdf5_to_batch(obs_grp, t: int, task: str, tokenizer, device: str,
                  stats: dict) -> dict:
    img_a = obs_grp["agentview_rgb"][t]
    img_w = obs_grp["eye_in_hand_rgb"][t]
    state = build_state_from_obs(obs_grp, t)

    obs = {
        "pixels": {"image": img_a, "image2": img_w},
        "robot_state": {
            "eef":    {"pos": state[:3], "quat": state[3:7]},
            "joints": {"pos": state[7:14]},
            "gripper": {"qpos": np.zeros(2)},
        },
    }
    from lerobot.utils.constants import OBS_STATE
    batch = obs_to_policy_batch(
        obs, task, tokenizer, device,
        state_key=LIBERO_STATE_KEY,
        image_key=LIBERO_AGENTVIEW_KEY,
        image2_key=LIBERO_WRIST_KEY,
    )
    # Apply normalization (match training pipeline)
    s_mean = stats["observation.state"]["mean"].to(device)
    s_std  = stats["observation.state"]["std"].to(device)
    batch[OBS_STATE] = (batch[OBS_STATE] - s_mean) / (s_std + 1e-8)
    return batch


def evaluate_condition(hdf5_files, model, tokenizer, eeg_sampler,
                       device, stats, condition_name, cls, a_mean, a_std):
    """Run open-loop eval for one EEG condition. Returns {task: metrics}."""
    all_mae, all_l2, all_grip = [], [], []

    for hdf5_path in hdf5_files:
        task_name = os.path.basename(hdf5_path).replace("_demo.hdf5","").replace("_"," ")

        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(f["data"].keys())[:MAX_DEMOS]
            for dk in demo_keys:
                demo      = f["data"][dk]
                gt_raw    = demo["actions"][:MAX_STEPS].astype(np.float32)
                T         = len(gt_raw)

                model.reset()
                preds = []
                for t in range(T):
                    batch = hdf5_to_batch(demo["obs"], t, task_name,
                                         tokenizer, device, stats)
                    if cls is not None:
                        eeg = eeg_sampler.sample(1, device, cls=cls)
                    else:
                        # Zero EEG
                        eeg = torch.zeros(1, 1, 64, 320, device=device)

                    with torch.no_grad():
                        action = model.select_action(batch, eeg=eeg)
                        action = action.squeeze(0).cpu().numpy()

                    # Denormalize
                    action = action * a_std + a_mean
                    preds.append(action)

                pred = np.array(preds)
                gt   = gt_raw[:T]
                all_mae.append(np.abs(pred - gt).mean())
                all_l2.append(np.linalg.norm(pred - gt, axis=1).mean())
                all_grip.append((np.sign(pred[:, 6]) == np.sign(gt[:, 6])).mean())

    return {
        "mae":  np.mean(all_mae),
        "l2":   np.mean(all_l2),
        "grip": np.mean(all_grip) * 100,
    }


def main():
    device = get_device()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"=== SmolVLA+EEG Open-Loop Evaluation ===")
    print(f"Suite : {SUITE}  |  Demos: {MAX_DEMOS}/task  |  Steps: {MAX_STEPS}")

    # ── Check prerequisite files ───────────────────────────────────────────────
    for path, name in [
        (EEG_CKPT,      "EEG fine-tuned checkpoint"),
        (EEG_ENC_PT,    "EEG encoder"),
        (EEG_DATA_PATH, "EEG epoch data"),
        (STATS_PATH,    "Dataset stats"),
    ]:
        if not os.path.exists(path):
            print(f"\nMissing {name}: {path}")
            raise SystemExit(1)

    # ── Load stats ────────────────────────────────────────────────────────────
    stats  = torch.load(STATS_PATH, map_location="cpu", weights_only=False)
    a_mean = stats["action"]["mean"].numpy()
    a_std  = stats["action"]["std"].numpy()

    # ── Build model ───────────────────────────────────────────────────────────
    cfg = make_libero_smolvla_config(device)
    cfg.load_vlm_weights = False
    base_policy = SmolVLAPolicy(cfg).to(device)

    eeg_enc = EEGNet(n_channels=64, n_timepoints=320, n_classes=4, embed_dim=64)
    enc_ckpt = torch.load(EEG_ENC_PT, map_location="cpu", weights_only=False)
    eeg_enc.load_state_dict(enc_ckpt["backbone_state"], strict=False)
    eeg_enc = eeg_enc.to(device)
    for p in eeg_enc.parameters():
        p.requires_grad = False

    model = SmolVLAWithEEG(base_policy, eeg_enc, eeg_dropout=0.0)  # no dropout at eval
    model = model.to(device)

    ckpt = torch.load(EEG_CKPT, map_location=device, weights_only=False)
    model.base_policy.load_state_dict(ckpt["policy_state"])
    model.eeg_proj.load_state_dict(ckpt["eeg_proj_state"])
    model.eval()
    print(f"Loaded EEG checkpoint: {EEG_CKPT} (step {ckpt.get('step','?')})")

    tokenizer   = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    eeg_sampler = EEGSampler(EEG_DATA_PATH)

    hdf5_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.hdf5")))
    print(f"Found {len(hdf5_files)} task files\n")

    # ── Evaluate all conditions ────────────────────────────────────────────────
    t_start  = time.time()
    cond_results = {}

    for cond_name, cls in EEG_CONDITIONS.items():
        print(f"Evaluating condition: {cond_name} (class={cls}) ...")
        res = evaluate_condition(
            hdf5_files, model, tokenizer, eeg_sampler,
            device, stats, cond_name, cls, a_mean, a_std,
        )
        cond_results[cond_name] = res
        print(f"  MAE={res['mae']:.3f}  L2={res['l2']:.3f}  Grip={res['grip']:.1f}%")

    elapsed = time.time() - t_start

    # ── Results table ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SmolVLA+EEG Evaluation — {SUITE}  ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"{'Condition':<16} {'MAE↓':>7} {'L2↓':>7} {'Grip%↑':>8}")
    print("-" * 44)
    for cond, res in cond_results.items():
        print(f"{cond:<16} {res['mae']:>7.3f} {res['l2']:>7.3f} {res['grip']:>7.1f}%")
    print("=" * 60)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, f"eeg_eval_{SUITE}.npz")
    np.savez(
        out_path,
        conditions=list(cond_results.keys()),
        mae=[r["mae"]  for r in cond_results.values()],
        l2= [r["l2"]   for r in cond_results.values()],
        grip=[r["grip"] for r in cond_results.values()],
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle(f"SmolVLA+EEG — {SUITE} — EEG Condition Comparison", fontsize=13)

        conds  = list(cond_results.keys())
        maes   = [cond_results[c]["mae"]  for c in conds]
        l2s    = [cond_results[c]["l2"]   for c in conds]
        grips  = [cond_results[c]["grip"] for c in conds]
        colors = ["#aaaaaa", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        x      = np.arange(len(conds))

        for ax, vals, ylabel, title in [
            (axes[0], maes,  "MAE ↓",         "Action MAE by EEG Condition"),
            (axes[1], l2s,   "L2 Distance ↓", "L2 Distance by EEG Condition"),
            (axes[2], grips, "Gripper Acc %↑","Gripper Accuracy by EEG Condition"),
        ]:
            bars = ax.bar(x, vals, color=colors[:len(conds)], alpha=0.85, edgecolor="white")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                        f"{v:.3f}" if "Gripper" not in title else f"{v:.1f}%",
                        ha="center", va="bottom", fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(conds, rotation=25, ha="right", fontsize=9)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=10)
            ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f"eeg_eval_{SUITE}.png")
        plt.savefig(plot_path, dpi=130)
        print(f"\nPlot saved: {plot_path}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
