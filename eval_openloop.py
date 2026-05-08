#!/usr/bin/env python
"""
Open-loop action prediction accuracy on LIBERO demo data.

For each timestep in a demo, we feed the (image, state, task) observation
to SmolVLA and compare the predicted action to the ground-truth action.

Metrics reported per task and overall:
  - MAE  (mean absolute error, per action dim and mean across dims)
  - L2   (Euclidean distance between predicted and GT action vectors)
  - Gripper accuracy  (% of timesteps where predicted gripper sign matches GT)

This is the FAST evaluation path — no simulation rollout needed.
Useful for checking training progress after fine-tuning.

Usage:
    cd /Users/r/Projects/SmolVLA_cl
    PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python eval_openloop.py

Options (env vars):
    SUITE       libero_spatial (default) | libero_object | libero_goal
    MAX_DEMOS   5  — demos per task HDF5 file to evaluate
    MAX_STEPS   30 — max steps per demo (None = full demo)
    MODEL       random | smolvla_base | trained  (default: random)
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
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE

from libero_smolvla_config import (
    make_libero_smolvla_config,
    LIBERO_AGENTVIEW_KEY,
    LIBERO_WRIST_KEY,
    LIBERO_STATE_KEY,
    STATE_DIM,
    ACTION_DIM,
)
from utils import get_device, obs_to_policy_batch, dummy_dataset_stats

# ── Config ────────────────────────────────────────────────────────────────────
SUITE      = os.environ.get("SUITE", "libero_spatial")
MAX_DEMOS  = int(os.environ.get("MAX_DEMOS", "5"))
MAX_STEPS  = os.environ.get("MAX_STEPS")
MAX_STEPS  = int(MAX_STEPS) if MAX_STEPS else None
MODEL       = os.environ.get("MODEL", "random")   # "random" | "smolvla_base" | "trained"
CKPT_DIR    = os.environ.get("CHECKPOINT_DIR", None)  # override for trained model
MODEL_TAG   = os.environ.get("MODEL_TAG", MODEL)      # label for output file
OUTPUT_DIR  = "./eval_output"
DATASET_DIR = f"/Users/r/LIBERO/libero/datasets/{SUITE}"
ACTION_LABELS = ["delta_x", "delta_y", "delta_z", "delta_roll", "delta_pitch", "delta_yaw", "gripper"]
# ──────────────────────────────────────────────────────────────────────────────


def build_state_from_hdf5_obs(obs_grp, t: int) -> np.ndarray:
    """Build 14-dim state from HDF5 demo obs at timestep t."""
    ee_pos   = obs_grp["ee_pos"][t]      # (3,)
    ee_ori   = obs_grp["ee_ori"][t]      # (3,) axis-angle
    joint    = obs_grp["joint_states"][t] # (7,)
    # Convert 3-dim axis-angle to 4-dim quaternion approximation (small angle)
    # For demo purposes use zeros for quat w=1, xyz=axis-angle/2
    quat = np.array([ee_ori[0]/2, ee_ori[1]/2, ee_ori[2]/2, 1.0], dtype=np.float32)
    quat /= np.linalg.norm(quat)
    return np.concatenate([ee_pos, quat, joint]).astype(np.float32)


def hdf5_obs_to_batch(obs_grp, t: int, task: str, tokenizer, device: str,
                      stats: dict | None = None) -> dict:
    """Convert one HDF5 demo timestep to a SmolVLA-compatible batch.
    If stats provided, normalize state (matches training pipeline)."""
    img_agent = obs_grp["agentview_rgb"][t]
    img_wrist = obs_grp["eye_in_hand_rgb"][t]

    state = build_state_from_hdf5_obs(obs_grp, t)

    obs = {
        "pixels": {"image": img_agent, "image2": img_wrist},
        "robot_state": {
            "eef":    {"pos": state[:3], "quat": state[3:7]},
            "joints": {"pos": state[7:14]},
            "gripper": {"qpos": np.zeros(2)},
        },
    }
    batch = obs_to_policy_batch(
        obs, task, tokenizer, device,
        state_key=LIBERO_STATE_KEY,
        image_key=LIBERO_AGENTVIEW_KEY,
        image2_key=LIBERO_WRIST_KEY,
    )

    # Apply dataset normalization to state (matches how training data was prepared)
    if stats is not None:
        s_mean = stats["observation.state"]["mean"].to(device)
        s_std  = stats["observation.state"]["std"].to(device)
        batch[OBS_STATE] = (batch[OBS_STATE] - s_mean) / (s_std + 1e-8)

    return batch


def evaluate_hdf5(
    hdf5_path: str,
    task_name: str,
    policy: SmolVLAPolicy,
    tokenizer,
    device: str,
    max_demos: int,
    max_steps: int | None,
    stats: dict | None = None,
) -> dict:
    """Run open-loop evaluation on one HDF5 file. Returns per-demo metrics."""
    results = []

    # Pre-compute action denorm constants if stats available
    if stats is not None:
        a_mean = stats["action"]["mean"].numpy()   # (7,)
        a_std  = stats["action"]["std"].numpy()    # (7,)
    else:
        a_mean = np.zeros(ACTION_DIM)
        a_std  = np.ones(ACTION_DIM)

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(f["data"].keys())[:max_demos]

        for demo_key in demo_keys:
            demo = f["data"][demo_key]
            gt_actions = demo["actions"][:]   # (T, 7) — raw unnormalized
            T = len(gt_actions)
            if max_steps:
                T = min(T, max_steps)

            pred_actions = []
            policy.reset()

            for t in range(T):
                batch = hdf5_obs_to_batch(
                    demo["obs"], t, task_name, tokenizer, device, stats=stats
                )
                with torch.no_grad():
                    action = policy.select_action(batch).squeeze(0).cpu().numpy()
                # Denormalize prediction back to raw action space
                action = action * a_std + a_mean
                pred_actions.append(action)

            pred = np.array(pred_actions)        # (T, 7)
            gt   = gt_actions[:T].astype(np.float32)

            mae_per_dim = np.abs(pred - gt).mean(axis=0)   # (7,)
            mae_mean    = mae_per_dim.mean()
            l2_mean     = np.linalg.norm(pred - gt, axis=1).mean()
            grip_acc    = (np.sign(pred[:, 6]) == np.sign(gt[:, 6])).mean()

            results.append({
                "demo":        demo_key,
                "steps":       T,
                "mae_mean":    mae_mean,
                "mae_per_dim": mae_per_dim,
                "l2_mean":     l2_mean,
                "grip_acc":    grip_acc,
            })

    return results


def print_results_table(all_results: dict[str, list]) -> None:
    print(f"\n{'Task':<55} {'Demos':>5} {'MAE↓':>7} {'L2↓':>7} {'Grip%↑':>7}")
    print("-" * 85)

    global_mae, global_l2, global_grip = [], [], []

    for task, results in sorted(all_results.items()):
        mae  = np.mean([r["mae_mean"] for r in results])
        l2   = np.mean([r["l2_mean"] for r in results])
        grip = np.mean([r["grip_acc"] for r in results]) * 100
        n    = sum(r["steps"] for r in results)
        short_task = task[:54]
        print(f"{short_task:<55} {len(results):>5} {mae:>7.3f} {l2:>7.3f} {grip:>6.1f}%")
        global_mae.append(mae)
        global_l2.append(l2)
        global_grip.append(grip)

    print("-" * 85)
    print(
        f"{'OVERALL':<55} {len(all_results):>5} "
        f"{np.mean(global_mae):>7.3f} {np.mean(global_l2):>7.3f} "
        f"{np.mean(global_grip):>6.1f}%"
    )

    # Per-dim breakdown (average across tasks)
    all_mae_dims = np.array([
        r["mae_per_dim"] for results in all_results.values() for r in results
    ])
    print(f"\nPer-dimension MAE (averaged across all tasks):")
    for i, (label, val) in enumerate(zip(ACTION_LABELS, all_mae_dims.mean(axis=0))):
        bar = "█" * int(val * 20)
        print(f"  [{i}] {label:<12} {val:.4f}  {bar}")


def main():
    device = get_device()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"=== SmolVLA Open-Loop Evaluation ===")
    print(f"Suite   : {SUITE}")
    print(f"Model   : {MODEL}")
    print(f"Demos   : {MAX_DEMOS} per task  |  Max steps: {MAX_STEPS or 'full'}")
    print(f"Device  : {device}")

    eval_stats = None   # dataset stats for normalization (trained model only)

    # Build policy
    if MODEL == "trained":
        from load_trained import load_trained_policy
        ckpt_label = CKPT_DIR or f"checkpoints/{SUITE}"
        print(f"\nLoading trained checkpoint from {ckpt_label} ...")
        policy, tokenizer, eval_stats = load_trained_policy(
            suite=SUITE, checkpoint_dir=CKPT_DIR, device=device
        )
        policy_type = MODEL_TAG
        policy.eval()
    elif MODEL == "smolvla_base":
        print("\nLoading lerobot/smolvla_base ...")
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        policy_type = "smolvla_base"
        policy = policy.to(device)
        policy.eval()
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    else:
        print("\nBuilding random-weight SmolVLA ...")
        cfg = make_libero_smolvla_config(device)
        cfg.load_vlm_weights = False
        cfg.num_steps = 2
        policy = SmolVLAPolicy(cfg)
        policy_type = "random"
        policy = policy.to(device)
        policy.eval()
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

    # Find demo HDF5 files
    hdf5_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.hdf5")))
    if not hdf5_files:
        print(f"No HDF5 files found in {DATASET_DIR}")
        return

    print(f"\nFound {len(hdf5_files)} task files in {SUITE}")

    # Run evaluation
    all_results: dict[str, list] = {}
    t_start = time.time()

    for hdf5_path in hdf5_files:
        task_name = os.path.basename(hdf5_path).replace("_demo.hdf5", "").replace("_", " ")
        print(f"\n  Evaluating: {task_name[:60]} ...")

        results = evaluate_hdf5(
            hdf5_path, task_name, policy, tokenizer, device,
            max_demos=MAX_DEMOS,
            max_steps=MAX_STEPS,
            stats=eval_stats,
        )
        all_results[task_name] = results

        mae = np.mean([r["mae_mean"] for r in results])
        grip = np.mean([r["grip_acc"] for r in results]) * 100
        print(f"    MAE={mae:.3f}  Gripper={grip:.1f}%  ({len(results)} demos)")

    elapsed = time.time() - t_start
    print(f"\nTotal evaluation time: {elapsed:.1f}s")

    print_results_table(all_results)

    # Save results
    out_path = os.path.join(OUTPUT_DIR, f"openloop_{SUITE}_{MODEL_TAG}.npz")
    np.savez(
        out_path,
        tasks=list(all_results.keys()),
        mae=[np.mean([r["mae_mean"] for r in v]) for v in all_results.values()],
        l2=[np.mean([r["l2_mean"] for r in v]) for v in all_results.values()],
        grip=[np.mean([r["grip_acc"] for r in v]) for v in all_results.values()],
    )
    print(f"\nResults saved: {out_path}")

    print(f"\nNote: with '{MODEL}' weights, MAE ~1.0+ is expected (untrained).")
    print("After fine-tuning on LIBERO, expect MAE < 0.1 for good policies.")


if __name__ == "__main__":
    main()
