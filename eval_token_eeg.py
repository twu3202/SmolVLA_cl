#!/usr/bin/env python
"""
Stage 5: Controllability eval for the token-level VLM injection model.

Same protocol as `eval_synthetic_eeg.py` (Stage 3), but loads the token-level
checkpoint. Side-by-side comparison answers: does deeper VLM-level integration
fix the architectural failure of additive injection?

Usage:
    SUITE=libero_spatial PYTHONPATH=/Users/r/LIBERO \
        /opt/anaconda3/envs/lerobot/bin/python eval_token_eeg.py
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
from train_smolvla_eeg import EEGSampler
from train_smolvla_eeg_token import SmolVLATokenLevelEEG, VLM_HIDDEN_DIM
from utils import get_device, obs_to_policy_batch

# ── Config ────────────────────────────────────────────────────────────────────
SUITE         = os.environ.get("SUITE", "libero_spatial")
MAX_DEMOS     = int(os.environ.get("MAX_DEMOS", "5"))
MAX_STEPS     = int(os.environ.get("MAX_STEPS", "30"))
EEG_CKPT      = os.environ.get("EEG_CKPT",
    f"./checkpoints/{SUITE}_eeg_token/step_001000.pt")
DATASET_DIR   = f"/Users/r/LIBERO/libero/datasets/{SUITE}"
EEG_DATA_PATH = "./data/eeg_physionet/epochs.npz"
EEG_ENC_PT    = "./checkpoints/eeg_encoder/encoder_only.pt"
STATS_PATH    = f"./checkpoints/{SUITE}/dataset_stats.pt"
OUTPUT_DIR    = "./eval_output"

EEG_CONDITIONS = [
    ("no_eeg",     None, "—"),
    ("left_fist",  0,    "Δx < 0"),
    ("right_fist", 1,    "Δx > 0"),
    ("both_fists", 2,    "gripper close"),
    ("both_feet",  3,    "Δy > 0"),
]
# ──────────────────────────────────────────────────────────────────────────────


def build_state(obs_grp, t):
    ee_pos = obs_grp["ee_pos"][t]
    ee_ori = obs_grp["ee_ori"][t]
    joint  = obs_grp["joint_states"][t]
    quat = np.array([ee_ori[0]/2, ee_ori[1]/2, ee_ori[2]/2, 1.0], dtype=np.float32)
    quat /= np.linalg.norm(quat)
    return np.concatenate([ee_pos, quat, joint]).astype(np.float32)


def hdf5_to_batch(obs_grp, t, task, tokenizer, device, stats):
    img_a = obs_grp["agentview_rgb"][t]
    img_w = obs_grp["eye_in_hand_rgb"][t]
    state = build_state(obs_grp, t)
    obs = {
        "pixels": {"image": img_a, "image2": img_w},
        "robot_state": {
            "eef":    {"pos": state[:3], "quat": state[3:7]},
            "joints": {"pos": state[7:14]},
            "gripper": {"qpos": np.zeros(2)},
        },
    }
    from lerobot.utils.constants import OBS_STATE
    batch = obs_to_policy_batch(obs, task, tokenizer, device,
                                state_key=LIBERO_STATE_KEY,
                                image_key=LIBERO_AGENTVIEW_KEY,
                                image2_key=LIBERO_WRIST_KEY)
    s_mean = stats["observation.state"]["mean"].to(device)
    s_std  = stats["observation.state"]["std"].to(device)
    batch[OBS_STATE] = (batch[OBS_STATE] - s_mean) / (s_std + 1e-8)
    return batch


def collect_actions(model, hdf5_files, tokenizer, eeg_sampler,
                    device, stats, cls, a_mean_np, a_std_np):
    actions = []
    for hdf5_path in hdf5_files:
        task = os.path.basename(hdf5_path).replace("_demo.hdf5","").replace("_"," ")
        with h5py.File(hdf5_path, "r") as f:
            for dk in sorted(f["data"].keys())[:MAX_DEMOS]:
                demo = f["data"][dk]
                T = min(MAX_STEPS, len(demo["actions"]))
                model.reset()
                for t in range(T):
                    batch = hdf5_to_batch(demo["obs"], t, task, tokenizer, device, stats)
                    if cls is None:
                        eeg = torch.zeros(1, 1, 64, 320, device=device)
                    else:
                        eeg = eeg_sampler.sample(1, device, cls=cls)
                    with torch.no_grad():
                        a_norm = model.select_action(batch, eeg=eeg).squeeze(0).cpu().numpy()
                    actions.append(a_norm * a_std_np + a_mean_np)
    return np.array(actions)


def bootstrap_ci(values, n_boot=1000, ci=0.95):
    rng = np.random.default_rng(0)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(n_boot)]
    a = (1 - ci) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def main():
    device = get_device()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"=== Token-Level VLM Injection — Controllability Test ===")
    print(f"Suite : {SUITE}  |  Demos: {MAX_DEMOS}/task  |  Steps/demo: {MAX_STEPS}")
    print(f"Checkpoint: {EEG_CKPT}\n")

    for path in [EEG_CKPT, EEG_ENC_PT, EEG_DATA_PATH, STATS_PATH]:
        if not os.path.exists(path):
            print(f"Missing: {path}")
            raise SystemExit(1)

    stats     = torch.load(STATS_PATH, map_location="cpu", weights_only=False)
    a_mean_np = stats["action"]["mean"].numpy()
    a_std_np  = stats["action"]["std"].numpy()

    cfg = make_libero_smolvla_config(device)
    cfg.load_vlm_weights = False
    base_policy = SmolVLAPolicy(cfg).to(device)

    eeg_enc = EEGNet(n_channels=64, n_timepoints=320, n_classes=4, embed_dim=64)
    eeg_enc.load_state_dict(
        torch.load(EEG_ENC_PT, map_location="cpu",
                   weights_only=False)["backbone_state"], strict=False)
    eeg_enc = eeg_enc.to(device)
    for p in eeg_enc.parameters():
        p.requires_grad = False

    model = SmolVLATokenLevelEEG(base_policy, eeg_enc,
                                 vlm_hidden_dim=VLM_HIDDEN_DIM,
                                 eeg_dropout=0.0).to(device)
    ckpt = torch.load(EEG_CKPT, map_location=device, weights_only=False)
    model.base_policy.load_state_dict(ckpt["policy_state"])
    model.eeg_proj.load_state_dict(ckpt["eeg_proj_state"])
    model.eval()
    print(f"Loaded token-level checkpoint (step {ckpt.get('step','?')})")

    tokenizer   = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    eeg_sampler = EEGSampler(EEG_DATA_PATH)
    hdf5_files  = sorted(glob.glob(os.path.join(DATASET_DIR, "*.hdf5")))

    t_start = time.time()
    all_actions = {}
    for cond, cls, expectation in EEG_CONDITIONS:
        print(f"\nCollecting actions: {cond}  (expects: {expectation}) ...")
        all_actions[cond] = collect_actions(
            model, hdf5_files, tokenizer, eeg_sampler,
            device, stats, cls, a_mean_np, a_std_np,
        )
        a = all_actions[cond]
        print(f"  N={len(a)}  mean Δx={a[:,0].mean():+.4f}  "
              f"Δy={a[:,1].mean():+.4f}  grip={a[:,6].mean():+.4f}")
    print(f"\nTotal eval time: {time.time()-t_start:.0f}s")

    base_dx = all_actions["no_eeg"][:, 0].mean()
    base_dy = all_actions["no_eeg"][:, 1].mean()
    base_gr = all_actions["no_eeg"][:, 6].mean()

    summary = []
    for cond, cls, exp in EEG_CONDITIONS:
        arr = all_actions[cond]
        dx, dy, gr = arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 6].mean()
        summary.append({
            "condition": cond, "expectation": exp,
            "dx_mean": dx, "dy_mean": dy, "gripper_mean": gr,
            "dx_ci":  bootstrap_ci(arr[:, 0]),
            "dy_ci":  bootstrap_ci(arr[:, 1]),
            "grip_ci": bootstrap_ci(arr[:, 6]),
            "n":      len(arr),
        })

    print(f"\n{'='*78}")
    print("Per-condition verdict (expected directional shift?)")
    print(f"{'='*78}")
    verdicts = {}
    for s in summary:
        cond = s["condition"]
        if cond == "no_eeg":
            continue
        passed = False
        if cond == "left_fist":
            shift = s["dx_mean"] - base_dx
            passed = shift < 0 and s["dx_ci"][1] < base_dx
            msg = f"shift in Δx = {shift:+.4f} (want < 0)"
        elif cond == "right_fist":
            shift = s["dx_mean"] - base_dx
            passed = shift > 0 and s["dx_ci"][0] > base_dx
            msg = f"shift in Δx = {shift:+.4f} (want > 0)"
        elif cond == "both_fists":
            shift = s["gripper_mean"] - base_gr
            passed = shift < 0 and s["grip_ci"][1] < base_gr
            msg = f"shift in gripper = {shift:+.4f} (want < 0)"
        elif cond == "both_feet":
            shift = s["dy_mean"] - base_dy
            passed = shift > 0 and s["dy_ci"][0] > base_dy
            msg = f"shift in Δy = {shift:+.4f} (want > 0)"
        verdicts[cond] = passed
        print(f"  {cond:<14} {'✓ PASS' if passed else '✗ fail':<8}  {msg}")

    n_pass = sum(verdicts.values())
    print(f"\nOverall: {n_pass}/4 directional tests passed.")
    if n_pass >= 3:
        print("  → ARCHITECTURE FIXED: deep VLM injection enables EEG controllability.")
    elif n_pass >= 2:
        print("  → Substantial improvement vs additive injection (1/4).")
    else:
        print("  → Still no controllability — deeper changes needed.")

    np.savez(
        os.path.join(OUTPUT_DIR, f"controllability_token_{SUITE}.npz"),
        conditions=[s["condition"] for s in summary],
        dx_mean=[s["dx_mean"] for s in summary],
        dy_mean=[s["dy_mean"] for s in summary],
        gripper_mean=[s["gripper_mean"] for s in summary],
        n_pass=n_pass,
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        conds  = [s["condition"] for s in summary]
        x      = np.arange(len(conds))
        colors = ["#aaaaaa","#1f77b4","#ff7f0e","#2ca02c","#d62728"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"SmolVLA+EEG TOKEN-LEVEL VLM Injection — Controllability "
            f"({n_pass}/4 directional tests passed)",
            fontsize=13, fontweight="bold",
        )
        ci_keys = {"dx_mean":"dx_ci","dy_mean":"dy_ci","gripper_mean":"grip_ci"}
        for ax, key, title, ylabel in [
            (axes[0], "dx_mean",      "Δx (left/right intent)",  "mean Δx"),
            (axes[1], "dy_mean",      "Δy (forward intent)",     "mean Δy"),
            (axes[2], "gripper_mean", "Gripper (close intent)",  "mean gripper"),
        ]:
            vals = [s[key] for s in summary]
            ckey = ci_keys[key]
            errs_lo = [s[key] - s[ckey][0] for s in summary]
            errs_hi = [s[ckey][1] - s[key] for s in summary]
            ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white",
                   yerr=np.array([errs_lo, errs_hi]), capsize=5)
            ax.axhline(vals[0], color="black", ls="--", lw=1, alpha=0.5,
                       label=f"no_eeg baseline ({vals[0]:+.3f})")
            ax.set_xticks(x); ax.set_xticklabels(conds, rotation=25, ha="right", fontsize=9)
            ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
            for i, v in enumerate(vals):
                ax.text(i, v + (max(vals)-min(vals))*0.04 if v >= 0 else
                           v - (max(vals)-min(vals))*0.07,
                        f"{v:+.3f}", ha="center", fontsize=8.5)
        plt.tight_layout()
        out = os.path.join(OUTPUT_DIR, f"controllability_token_{SUITE}.png")
        plt.savefig(out, dpi=130)
        print(f"\nPlot saved: {out}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
