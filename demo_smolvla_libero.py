#!/usr/bin/env python
"""
SmolVLA + LIBERO Demo — Early-stage VLA on robot arm simulation (Mac, no GPU required).

This script demonstrates the full pipeline:
  1. Create a LIBERO simulation environment via lerobot's LiberoEnv wrapper
  2. Build a SmolVLA policy configured for LIBERO's 7-DOF action space
  3. Load the SmolVLM2-500M vision-language backbone (runs on Apple MPS)
  4. Process LIBERO observations through the SmolVLA preprocessing pipeline
  5. Execute a rollout with SmolVLA-generated actions and save frames

Note: The action expert is randomly initialized (not fine-tuned on LIBERO).
      This demonstrates the inference pipeline; task success requires fine-tuning.

Usage:
    cd /Users/r/Projects/SmolVLA_cl
    PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python demo_smolvla_libero.py

    # Options (edit constants below or pass as env vars):
    SUITE=libero_object TASK_ID=1 STEPS=30 python demo_smolvla_libero.py
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

LIBERO_PATH = "/Users/r/LIBERO"
if LIBERO_PATH not in sys.path:
    sys.path.insert(0, LIBERO_PATH)

# ── Config ────────────────────────────────────────────────────────────────────
SUITE_NAME = os.environ.get("SUITE", "libero_spatial")
TASK_ID    = int(os.environ.get("TASK_ID", "0"))
N_STEPS    = int(os.environ.get("STEPS", "20"))
OUTPUT_DIR = "./output"
SAVE_EVERY = 5   # save a frame every N steps
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch

from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

from libero_smolvla_config import (
    make_libero_smolvla_config,
    LIBERO_AGENTVIEW_KEY,
    LIBERO_WRIST_KEY,
    LIBERO_STATE_KEY,
    STATE_DIM,
    ACTION_DIM,
)
from utils import get_device, obs_to_policy_batch, save_episode_frames, dummy_dataset_stats


# ── helpers ───────────────────────────────────────────────────────────────────

def build_env(suite_name: str, task_id: int) -> tuple[LiberoEnv, str]:
    bench = benchmark.get_benchmark_dict()
    suite = bench[suite_name]()
    task = suite.get_task(task_id)
    print(f"Task: {task.language}")

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
    return env, task.language


def build_policy(device: str) -> tuple[SmolVLAPolicy, object]:
    """Build SmolVLA with LIBERO feature spec and load VLM backbone."""
    from transformers import AutoTokenizer

    config = make_libero_smolvla_config(device=device)

    print("\nBuilding SmolVLAPolicy for LIBERO ...")
    print(f"  VLM backbone  : {config.vlm_model_name}")
    print(f"  State dim     : {STATE_DIM}")
    print(f"  Action dim    : {ACTION_DIM}")
    print(f"  Device        : {device}")
    print(f"  Chunk size    : {config.chunk_size}")
    print(f"  Denoising steps: {config.num_steps}")

    t0 = time.time()
    policy = SmolVLAPolicy(config)
    policy = policy.to(device)
    policy.eval()
    print(f"  Policy built in {time.time() - t0:.1f}s")

    # Load tokenizer for language processing
    tokenizer = AutoTokenizer.from_pretrained(config.vlm_model_name)
    return policy, tokenizer


def apply_normalization(
    batch: dict[str, torch.Tensor],
    stats: dict,
    device: str,
) -> dict[str, torch.Tensor]:
    """Apply mean-std normalization to state (images use IDENTITY norm)."""
    if LIBERO_STATE_KEY in batch and "observation.state" in stats:
        mean = stats["observation.state"]["mean"].to(device)
        std = stats["observation.state"]["std"].to(device)
        batch[LIBERO_STATE_KEY] = (batch[LIBERO_STATE_KEY] - mean) / (std + 1e-8)
    return batch


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"=== SmolVLA + LIBERO Demo ===")
    print(f"Device: {device}  |  Suite: {SUITE_NAME}  |  Task: {TASK_ID}  |  Steps: {N_STEPS}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Environment
    print("\n[1/4] Building LIBERO environment ...")
    env, task_description = build_env(SUITE_NAME, TASK_ID)
    obs, info = env.reset()
    print(f"  agentview obs shape: {obs['pixels']['image'].shape}")
    print(f"  wrist obs shape    : {obs['pixels']['image2'].shape}")
    print(f"  state (eef pos)    : {obs['robot_state']['eef']['pos'].round(3)}")

    # 2. Policy + tokenizer
    print("\n[2/4] Loading SmolVLA policy (VLM backbone) ...")
    policy, tokenizer = build_policy(device)
    policy.reset()

    # 3. Dataset statistics (identity for demo — no fine-tuned checkpoint)
    stats = dummy_dataset_stats(STATE_DIM, ACTION_DIM, device=device)

    # 4. Rollout
    print(f"\n[3/4] Running {N_STEPS}-step rollout ...")
    print(f"  Task: '{task_description}'")

    rollout_frames = []
    action_log = []
    inference_times = []

    for step in range(N_STEPS):
        # Prepare batch from current observation
        batch = obs_to_policy_batch(
            obs,
            task_description=task_description,
            tokenizer=tokenizer,
            device=device,
            state_key=LIBERO_STATE_KEY,
            image_key=LIBERO_AGENTVIEW_KEY,
            image2_key=LIBERO_WRIST_KEY,
            max_token_length=48,
        )
        batch = apply_normalization(batch, stats, device)

        # SmolVLA inference
        t_inf = time.time()
        with torch.no_grad():
            action_tensor = policy.select_action(batch)  # (batch, action_dim)
        inference_times.append(time.time() - t_inf)

        # select_action returns (batch_size, action_dim); squeeze for single-env step
        action_np = action_tensor.squeeze(0).cpu().numpy()
        action_log.append(action_np)

        # Clip to valid action range
        action_np = np.clip(action_np, -1.0, 1.0)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action_np)

        if step % SAVE_EVERY == 0 or step == N_STEPS - 1:
            frame = env.render()  # (H, W, 3) RGB
            rollout_frames.append(frame)
            print(
                f"  step {step:3d} | action[:3]={action_np[:3].round(3)} "
                f"| grip={action_np[-1]:.2f} | success={info.get('is_success', False)} "
                f"| inf={inference_times[-1]*1000:.0f}ms"
            )

        if terminated or truncated:
            print(f"  Episode ended at step {step} (success={info.get('is_success')})")
            obs, info = env.reset()
            policy.reset()

    env.close()

    # 5. Save results
    print(f"\n[4/4] Saving results to {OUTPUT_DIR} ...")
    save_episode_frames(
        rollout_frames,
        path=os.path.join(OUTPUT_DIR, "rollout_frames.png"),
    )

    # Save action trajectory
    actions_arr = np.array(action_log)
    np.save(os.path.join(OUTPUT_DIR, "actions.npy"), actions_arr)

    avg_inf = np.mean(inference_times) * 1000
    print(f"\n=== Summary ===")
    print(f"  Steps completed  : {N_STEPS}")
    print(f"  Avg inference    : {avg_inf:.0f} ms/step  ({1000/avg_inf:.1f} Hz)")
    print(f"  Actions saved    : {os.path.join(OUTPUT_DIR, 'actions.npy')}")
    print(f"  Frames saved     : {os.path.join(OUTPUT_DIR, 'rollout_frames.png')}")
    print(f"\nNote: actions are from a randomly-initialized expert head.")
    print(f"Fine-tune on LIBERO demonstrations to achieve task success.")


if __name__ == "__main__":
    main()
