#!/usr/bin/env python
"""
Controllability evaluation: does the synthetic-paired SmolVLA+EEG model
actually USE the EEG signal to steer its action output?

Test design:
  Hold the camera/state/language input fixed for each LIBERO observation.
  Inject 5 different EEG conditions in turn (no_eeg + 4 motor-imagery classes).
  Measure how the OUTPUT action distribution shifts.

Expected behavior if the model has learned the EEG→action mapping:
  EEG class 0 (left_fist)  → mean delta_x significantly NEGATIVE
  EEG class 1 (right_fist) → mean delta_x significantly POSITIVE
  EEG class 2 (both_fists) → mean gripper output more NEGATIVE (closing)
  EEG class 3 (both_feet)  → mean delta_y significantly POSITIVE

Usage:
    SUITE=libero_spatial PYTHONPATH=/Users/r/LIBERO \
        /opt/anaconda3/envs/lerobot/bin/python eval_synthetic_eeg.py
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
SUITE         = os.environ.get("SUITE", "libero_spatial")
MAX_DEMOS     = int(os.environ.get("MAX_DEMOS", "5"))
MAX_STEPS     = int(os.environ.get("MAX_STEPS", "30"))
EEG_CKPT      = os.environ.get("EEG_CKPT",
    f"./checkpoints/{SUITE}_eeg_synth/step_001000.pt")
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
    quat   = np.array([ee_ori[0]/2, ee_ori[1]/2, ee_ori[2]/2, 1.0], dtype=np.float32)
    quat  /= np.linalg.norm(quat)
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
    """Run model with one EEG condition; return (N,7) array of generated actions."""
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
                    a_raw = a_norm * a_std_np + a_mean_np
                    actions.append(a_raw)
    return np.array(actions)


def bootstrap_ci(values, n_boot=1000, ci=0.95):
    """Returns (low, high) bootstrap CI for the mean."""
    rng = np.random.default_rng(0)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(n_boot)]
    a = (1 - ci) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def main():
    device = get_device()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"=== SmolVLA+EEG Controllability Test ===")
    print(f"Suite : {SUITE}  |  Demos: {MAX_DEMOS}/task  |  Steps: {MAX_STEPS}")
    print(f"Checkpoint: {EEG_CKPT}\n")

    # Validate inputs
    for path, name in [(EEG_CKPT, "synthetic-trained checkpoint"),
                       (EEG_ENC_PT, "EEG encoder"),
                       (EEG_DATA_PATH, "EEG epochs"),
                       (STATS_PATH, "Dataset stats")]:
        if not os.path.exists(path):
            print(f"Missing {name}: {path}")
            if path == EEG_CKPT:
                print("Run: SYNTHETIC_PAIRING=1 PYTHONPATH=/Users/r/LIBERO "
                      "/opt/anaconda3/envs/lerobot/bin/python train_smolvla_eeg.py")
            raise SystemExit(1)

    stats = torch.load(STATS_PATH, map_location="cpu", weights_only=False)
    a_mean_np = stats["action"]["mean"].numpy()
    a_std_np  = stats["action"]["std"].numpy()

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

    model = SmolVLAWithEEG(base_policy, eeg_enc, eeg_dropout=0.0).to(device)
    ckpt = torch.load(EEG_CKPT, map_location=device, weights_only=False)
    model.base_policy.load_state_dict(ckpt["policy_state"])
    model.eeg_proj.load_state_dict(ckpt["eeg_proj_state"])
    model.eval()
    print(f"Loaded synthetic-paired checkpoint (step {ckpt.get('step','?')})")

    tokenizer   = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    eeg_sampler = EEGSampler(EEG_DATA_PATH)
    hdf5_files  = sorted(glob.glob(os.path.join(DATASET_DIR, "*.hdf5")))

    # ── Run all conditions ────────────────────────────────────────────────────
    t_start = time.time()
    all_actions = {}
    for cond, cls, expectation in EEG_CONDITIONS:
        print(f"\nCollecting actions for condition: {cond}  (expects: {expectation}) ...")
        all_actions[cond] = collect_actions(
            model, hdf5_files, tokenizer, eeg_sampler,
            device, stats, cls, a_mean_np, a_std_np,
        )
        print(f"  N={len(all_actions[cond])}  "
              f"mean Δx={all_actions[cond][:,0].mean():+.4f}  "
              f"mean Δy={all_actions[cond][:,1].mean():+.4f}  "
              f"mean grip={all_actions[cond][:,6].mean():+.4f}")

    elapsed = time.time() - t_start
    print(f"\nTotal eval time: {elapsed:.0f}s")

    # ── Stats: shift relative to no_eeg baseline + bootstrap CI ───────────────
    print(f"\n{'='*78}")
    print(f"Controllability Results — shift vs no_eeg baseline")
    print(f"{'='*78}")
    print(f"{'Condition':<14} {'Δx mean':>14} {'Δy mean':>14} {'gripper mean':>14}")
    print("-" * 78)

    base_dx = all_actions["no_eeg"][:, 0].mean()
    base_dy = all_actions["no_eeg"][:, 1].mean()
    base_gr = all_actions["no_eeg"][:, 6].mean()

    summary = []
    for cond, cls, exp in EEG_CONDITIONS:
        arr = all_actions[cond]
        dx, dy, gr = arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 6].mean()
        dx_lo, dx_hi = bootstrap_ci(arr[:, 0])
        dy_lo, dy_hi = bootstrap_ci(arr[:, 1])
        gr_lo, gr_hi = bootstrap_ci(arr[:, 6])

        if cond == "no_eeg":
            print(f"{cond:<14} "
                  f"{dx:>+8.4f} [{dx_lo:+.3f},{dx_hi:+.3f}]"[:14]+" "
                  f"{dy:>+8.4f}".rjust(14)+" "
                  f"{gr:>+8.4f}".rjust(14))
        else:
            ddx = dx - base_dx
            ddy = dy - base_dy
            dgr = gr - base_gr
            print(f"{cond:<14} "
                  f"{ddx:>+8.4f} [{dx_lo-base_dx:+.3f},{dx_hi-base_dx:+.3f}]"[:14]+" "
                  f"{ddy:>+8.4f}".rjust(14)+" "
                  f"{dgr:>+8.4f}".rjust(14))

        summary.append({
            "condition": cond, "expectation": exp,
            "dx_mean": dx, "dy_mean": dy, "gripper_mean": gr,
            "dx_ci":  [dx_lo, dx_hi],
            "dy_ci":  [dy_lo, dy_hi],
            "grip_ci":[gr_lo, gr_hi],
            "n":      len(arr),
        })

    # ── Verdict per condition ─────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("Per-condition verdict (expected directional shift achieved?)")
    print(f"{'='*78}")
    verdicts = {}
    for s in summary:
        cond = s["condition"]
        if cond == "no_eeg":
            continue
        passed = False
        msg = ""
        if cond == "left_fist":
            shift = s["dx_mean"] - base_dx
            passed = shift < 0 and s["dx_ci"][1] < base_dx  # CI doesn't cross baseline
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
        verdict = "✓ PASS" if passed else "✗ fail"
        verdicts[cond] = passed
        print(f"  {cond:<14} {verdict}    {msg}")

    n_pass = sum(verdicts.values())
    print(f"\nOverall: {n_pass}/4 directional tests passed.")
    if n_pass >= 3:
        print("  → Architecture is controllable. EEG modality is feasible.")
    elif n_pass >= 1:
        print("  → Partial controllability. Mapping signal exists but not robust.")
    else:
        print("  → No controllability. Architecture needs redesign (likely deeper VLM injection).")

    # ── Save results + plot ───────────────────────────────────────────────────
    out_npz = os.path.join(OUTPUT_DIR, f"controllability_{SUITE}.npz")
    np.savez(
        out_npz,
        conditions=[s["condition"] for s in summary],
        dx_mean=[s["dx_mean"] for s in summary],
        dy_mean=[s["dy_mean"] for s in summary],
        gripper_mean=[s["gripper_mean"] for s in summary],
        n_pass=n_pass,
    )
    print(f"\nSaved: {out_npz}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        conds  = [s["condition"] for s in summary]
        x      = np.arange(len(conds))
        colors = ["#aaaaaa", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"SmolVLA+EEG Controllability — {SUITE}  (synthetic pairing, {n_pass}/4 tests passed)",
            fontsize=13, fontweight="bold",
        )

        ci_key_map = {"dx_mean": "dx_ci", "dy_mean": "dy_ci", "gripper_mean": "grip_ci"}
        for ax, key, title, ylabel, expected in [
            (axes[0], "dx_mean",      "Δx (left/right intent)",
             "mean Δx output",    {"left_fist": "−", "right_fist": "+"}),
            (axes[1], "dy_mean",      "Δy (forward intent)",
             "mean Δy output",    {"both_feet": "+"}),
            (axes[2], "gripper_mean", "Gripper (close intent)",
             "mean gripper output",{"both_fists":"−"}),
        ]:
            ci_key = ci_key_map[key]
            vals  = [s[key] for s in summary]
            errs_lo = [s[key] - s[ci_key][0] for s in summary]
            errs_hi = [s[ci_key][1] - s[key] for s in summary]
            yerr = np.array([errs_lo, errs_hi])

            bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white",
                          yerr=yerr, capsize=5)
            ax.axhline(vals[0], color="black", ls="--", lw=1, alpha=0.5,
                       label=f"no_eeg baseline ({vals[0]:+.3f})")
            ax.set_xticks(x)
            ax.set_xticklabels(conds, rotation=25, ha="right", fontsize=9)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=10)
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(fontsize=8)

            # Annotate expected direction
            for i, c in enumerate(conds):
                if c in expected:
                    sign = expected[c]
                    ax.text(i, vals[i] + (max(vals)-min(vals))*0.05,
                            f"want {sign}", ha="center", fontsize=8,
                            color="red", fontweight="bold")

        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f"controllability_{SUITE}.png")
        plt.savefig(plot_path, dpi=130)
        print(f"Plot saved: {plot_path}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
