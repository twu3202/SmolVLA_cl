#!/usr/bin/env python
"""
Closed-loop task success rate evaluation on LIBERO.

Runs the policy in simulation and measures what fraction of episodes it
successfully completes the task. This is the standard VLA benchmark metric.

Metrics:
  - Success rate (%) per task
  - Mean success rate across tasks in the suite
  - Episode length distribution for successful episodes

Usage:
    cd /Users/r/Projects/SmolVLA_cl
    PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python eval_closedloop.py

Options (env vars):
    SUITE       libero_spatial (default) | libero_object | libero_goal
    N_EPISODES  3   — rollout episodes per task
    TASK_IDS    0,1,2  — comma-separated task IDs (default: all in suite)
    MODEL       random | smolvla_base  (default: random)
    MAX_STEPS   280 — max steps per episode (uses LIBERO defaults if unset)
"""

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
from transformers import AutoTokenizer

from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv, TASK_SUITE_MAX_STEPS
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

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
N_EPISODES = int(os.environ.get("N_EPISODES", "3"))
MODEL      = os.environ.get("MODEL", "random")
OUTPUT_DIR = "./eval_output"

_task_ids_env = os.environ.get("TASK_IDS")
TASK_IDS = [int(x) for x in _task_ids_env.split(",")] if _task_ids_env else None

_max_steps_env = os.environ.get("MAX_STEPS")
MAX_STEPS = int(_max_steps_env) if _max_steps_env else None
# ──────────────────────────────────────────────────────────────────────────────

ACTION_LABELS = ["delta_x", "delta_y", "delta_z", "delta_roll", "delta_pitch", "delta_yaw", "gripper"]


def build_policy(device: str):
    if MODEL == "smolvla_base":
        print("Loading lerobot/smolvla_base ...")
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
    else:
        print("Building random-weight SmolVLA ...")
        cfg = make_libero_smolvla_config(device)
        cfg.load_vlm_weights = False
        cfg.num_steps = 2
        policy = SmolVLAPolicy(cfg)
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

    policy = policy.to(device)
    policy.eval()
    return policy, tokenizer


def run_episode(
    env: LiberoEnv,
    task_description: str,
    policy: SmolVLAPolicy,
    tokenizer,
    device: str,
    max_steps: int,
    stats: dict,
) -> dict:
    """Run one episode. Returns success, steps taken, and action trace."""
    obs, info = env.reset()
    policy.reset()

    actions = []
    for step in range(max_steps):
        batch = obs_to_policy_batch(
            obs, task_description, tokenizer, device,
            state_key=LIBERO_STATE_KEY,
            image_key=LIBERO_AGENTVIEW_KEY,
            image2_key=LIBERO_WRIST_KEY,
        )
        # Normalize state
        mean = stats["observation.state"]["mean"].to(device)
        std  = stats["observation.state"]["std"].to(device)
        batch[LIBERO_STATE_KEY] = (batch[LIBERO_STATE_KEY] - mean) / (std + 1e-8)

        with torch.no_grad():
            action = policy.select_action(batch).squeeze(0).cpu().numpy()

        # smolvla_base outputs 6-DOF; pad to 7-DOF for LIBERO
        if len(action) == 6:
            action = np.append(action, 0.0)

        action = np.clip(action, -1.0, 1.0)
        actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)

        if info.get("is_success"):
            return {"success": True, "steps": step + 1, "actions": np.array(actions)}
        if terminated or truncated:
            break

    return {"success": False, "steps": step + 1, "actions": np.array(actions)}


def evaluate_task(
    suite,
    suite_name: str,
    task_id: int,
    policy: SmolVLAPolicy,
    tokenizer,
    device: str,
    n_episodes: int,
    max_steps: int,
    stats: dict,
) -> dict:
    task = suite.get_task(task_id)
    task_desc = task.language

    env = LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=suite_name,
        camera_name="agentview_image,robot0_eye_in_hand_image",
        obs_type="pixels_agent_pos",
        observation_width=256,
        observation_height=256,
        init_states=True,
        num_steps_wait=10,
        control_mode="relative",
    )

    episodes = []
    for ep in range(n_episodes):
        result = run_episode(env, task_desc, policy, tokenizer, device, max_steps, stats)
        episodes.append(result)
        status = "✓" if result["success"] else "✗"
        print(f"    ep {ep+1}/{n_episodes}: {status}  steps={result['steps']}")

    env.close()

    successes = sum(1 for e in episodes if e["success"])
    success_rate = successes / n_episodes
    avg_steps = np.mean([e["steps"] for e in episodes])
    success_steps = [e["steps"] for e in episodes if e["success"]]

    return {
        "task_id":      task_id,
        "task_name":    task.name,
        "task_desc":    task_desc,
        "n_episodes":   n_episodes,
        "n_success":    successes,
        "success_rate": success_rate,
        "avg_steps":    avg_steps,
        "success_steps": success_steps,
    }


def print_summary(results: list[dict]) -> None:
    print(f"\n{'='*80}")
    print(f"{'Task':<55} {'SR':>6} {'Succ/N':>8} {'Avg steps':>10}")
    print("-" * 80)
    for r in results:
        short = r["task_name"][:54]
        sr_pct = r["success_rate"] * 100
        print(
            f"{short:<55} {sr_pct:>5.1f}% "
            f"{r['n_success']:>3}/{r['n_episodes']:<4} "
            f"{r['avg_steps']:>9.1f}"
        )
    print("-" * 80)
    overall_sr = np.mean([r["success_rate"] for r in results]) * 100
    print(f"{'MEAN SUCCESS RATE':<55} {overall_sr:>5.1f}%")
    print(f"\nNote: random-weight model → ~0% success (expected).")
    print("Fine-tune on LIBERO demos to achieve meaningful success rates.")


def main():
    device = get_device()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"=== SmolVLA Closed-Loop Evaluation ===")
    print(f"Suite      : {SUITE}")
    print(f"Model      : {MODEL}")
    print(f"Episodes   : {N_EPISODES} per task")
    print(f"Task IDs   : {TASK_IDS or 'all'}")
    print(f"Device     : {device}")

    bench = benchmark.get_benchmark_dict()
    suite = bench[SUITE]()
    total_tasks = len(suite.tasks)

    task_ids = TASK_IDS if TASK_IDS else list(range(total_tasks))
    print(f"\nTasks: {len(task_ids)}/{total_tasks} in {SUITE}")

    max_steps = MAX_STEPS or TASK_SUITE_MAX_STEPS.get(SUITE, 300)
    print(f"Max steps/episode: {max_steps}")

    policy, tokenizer = build_policy(device)
    stats = dummy_dataset_stats(STATE_DIM, ACTION_DIM, device=device)

    results = []
    t_start = time.time()

    for i, task_id in enumerate(task_ids):
        task = suite.get_task(task_id)
        print(f"\n[{i+1}/{len(task_ids)}] Task {task_id}: {task.language[:60]}")

        result = evaluate_task(
            suite, SUITE, task_id,
            policy, tokenizer, device,
            n_episodes=N_EPISODES,
            max_steps=max_steps,
            stats=stats,
        )
        results.append(result)

        sr = result["success_rate"] * 100
        print(f"  → success rate: {sr:.1f}% ({result['n_success']}/{N_EPISODES})")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s  ({elapsed/len(results)/N_EPISODES:.1f}s per episode)")

    print_summary(results)

    out_path = os.path.join(OUTPUT_DIR, f"closedloop_{SUITE}_{MODEL}.npz")
    np.savez(
        out_path,
        task_ids=[r["task_id"] for r in results],
        task_names=[r["task_name"] for r in results],
        success_rates=[r["success_rate"] for r in results],
        n_episodes=N_EPISODES,
        model=MODEL,
        suite=SUITE,
    )
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
