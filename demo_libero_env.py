#!/usr/bin/env python
"""
Quick sanity check: LIBERO environment running through lerobot's LiberoEnv wrapper.

Usage:
    PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python demo_libero_env.py

This verifies:
- LIBERO benchmark suites load correctly
- LiberoEnv wrapper initializes and resets
- Random-action rollout produces observations of the right shape
- Frames are saved to ./frames/ for inspection
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Ensure lerobot source is on the path when run via PYTHONPATH=/Users/r/LIBERO
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

LIBERO_PATH = "/Users/r/LIBERO"
if LIBERO_PATH not in sys.path:
    sys.path.insert(0, LIBERO_PATH)

from lerobot.envs.libero import LiberoEnv

from libero.libero import benchmark


def main():
    suite_name = "libero_spatial"
    task_id = 0
    n_steps = 20
    save_frames = True
    frame_dir = "./frames"

    print(f"=== LIBERO Environment Demo ===")
    print(f"Suite: {suite_name}  |  Task ID: {task_id}  |  Steps: {n_steps}")

    # Load benchmark suite
    bench = benchmark.get_benchmark_dict()
    suite = bench[suite_name]()
    task = suite.get_task(task_id)
    print(f"\nTask name   : {task.name}")
    print(f"Description : {task.language}")

    # Create environment via lerobot's wrapper
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

    obs, info = env.reset()
    print(f"\nReset OK.")
    print(f"  pixels.image shape    : {obs['pixels']['image'].shape}")
    print(f"  pixels.image2 shape   : {obs['pixels']['image2'].shape}")
    print(f"  eef pos               : {obs['robot_state']['eef']['pos'].round(3)}")
    print(f"  eef quat              : {obs['robot_state']['eef']['quat'].round(3)}")
    print(f"  gripper qpos          : {obs['robot_state']['gripper']['qpos'].round(3)}")
    print(f"  joint pos (first 3)   : {obs['robot_state']['joints']['pos'][:3].round(3)}")

    if save_frames:
        os.makedirs(frame_dir, exist_ok=True)

    total_reward = 0.0
    successes = 0

    for step in range(n_steps):
        # Random action in [-1, 1]^7  (6 DOF + gripper)
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if info.get("is_success"):
            successes += 1

        if save_frames and step % 5 == 0:
            frame = env.render()  # (H, W, 3) RGB
            # Save via matplotlib to avoid needing cv2
            try:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].imshow(obs["pixels"]["image"])
                axes[0].set_title(f"agentview (step {step})")
                axes[0].axis("off")
                axes[1].imshow(obs["pixels"]["image2"])
                axes[1].set_title("wrist cam")
                axes[1].axis("off")
                fig.suptitle(f"Task: {task.language[:50]}", fontsize=8)
                plt.tight_layout()
                path = os.path.join(frame_dir, f"step_{step:04d}.png")
                plt.savefig(path, dpi=100)
                plt.close(fig)
                print(f"  Saved frame: {path}")
            except Exception as e:
                print(f"  Frame save failed: {e}")

        if terminated or truncated:
            print(f"  Episode ended at step {step} (success={info.get('is_success')})")
            obs, info = env.reset()

    env.close()
    print(f"\nDone. Steps: {n_steps} | Total reward: {total_reward:.3f} | Successes: {successes}")
    if save_frames:
        print(f"Frames saved to: {os.path.abspath(frame_dir)}")


if __name__ == "__main__":
    main()
