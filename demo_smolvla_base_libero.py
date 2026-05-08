#!/usr/bin/env python
"""
SmolVLA-base + LIBERO Demo — loads the pretrained lerobot/smolvla_base checkpoint
and runs it on LIBERO observations via the official preprocessor pipeline.

smolvla_base is trained for SO100 (6-DOF SO-ARM100), not LIBERO (7-DOF Panda).
Actions will not produce task-relevant behavior, but this script validates that
the full pretrained pipeline (tokenizer + VLM + flow-matching action expert) runs
end-to-end on a Mac M-series chip via MPS.

Usage:
    cd /Users/r/Projects/SmolVLA_cl
    PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python demo_smolvla_base_libero.py
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
from lerobot.envs.libero import LiberoEnv
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE

from utils import get_device, save_episode_frames

# ── Config ────────────────────────────────────────────────────────────────────
SUITE_NAME  = "libero_spatial"
TASK_ID     = 0
N_STEPS     = 15
OUTPUT_DIR  = "./output_base"
MODEL_ID    = "lerobot/smolvla_base"
# ──────────────────────────────────────────────────────────────────────────────

# smolvla_base image keys (SO100 naming)
IMG_KEY1  = "observation.images.camera1"
IMG_KEY2  = "observation.images.camera2"
IMG_KEY3  = "observation.images.camera3"


def libero_obs_to_smolvla_base_batch(
    obs: dict,
    task: str,
    tokenizer,
    device: str,
    tokenizer_max_length: int = 48,
) -> dict[str, torch.Tensor]:
    """
    Adapt a LIBERO observation to the feature layout of smolvla_base.

    smolvla_base expects:
      - observation.state          (6,)   — SO100 joint positions
      - observation.images.camera1 (3, 256, 256)
      - observation.images.camera2 (3, 256, 256)
      - observation.images.camera3 (3, 256, 256)

    We remap:
      - LIBERO agentview → camera1
      - LIBERO wrist cam → camera2
      - Zero-padded dummy → camera3   (marked absent via padding mask)
      - LIBERO joint pos (7) truncated to 6 → observation.state
    """
    batch: dict[str, torch.Tensor] = {}

    def img_to_tensor(arr: np.ndarray) -> torch.Tensor:
        # (H, W, 3) uint8 → (1, 3, H, W) float [0,1]
        t = torch.from_numpy(arr).float() / 255.0
        return t.permute(2, 0, 1).unsqueeze(0).to(device)

    batch[IMG_KEY1] = img_to_tensor(obs["pixels"]["image"])
    batch[IMG_KEY2] = img_to_tensor(obs["pixels"]["image2"])

    # Dummy third camera — zero image, mask = False (absent)
    H, W = obs["pixels"]["image"].shape[:2]
    batch[IMG_KEY3] = torch.zeros(1, 3, H, W, device=device)
    batch[f"{IMG_KEY3}_padding_mask"] = torch.zeros(1, dtype=torch.bool, device=device)

    # State: use first 6 joint positions (truncate from 7-DOF LIBERO)
    joint_pos = obs["robot_state"]["joints"]["pos"][:6].astype(np.float32)
    batch[OBS_STATE] = torch.from_numpy(joint_pos).unsqueeze(0).to(device)

    # Language tokens
    task_nl = task if task.endswith("\n") else task + "\n"
    tokens = tokenizer(
        task_nl,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation=True,
    )
    batch[OBS_LANGUAGE_TOKENS]         = tokens["input_ids"].to(device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = tokens["attention_mask"].bool().to(device)

    return batch


def main():
    device = get_device()
    print(f"=== SmolVLA-base + LIBERO Demo ===")
    print(f"Model  : {MODEL_ID}")
    print(f"Device : {device}")
    print(f"Suite  : {SUITE_NAME}  |  Task: {TASK_ID}  |  Steps: {N_STEPS}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. LIBERO env
    print("\n[1/4] Building LIBERO environment ...")
    bench = benchmark.get_benchmark_dict()
    suite = bench[SUITE_NAME]()
    task = suite.get_task(TASK_ID)
    print(f"  Task: {task.language}")

    env = LiberoEnv(
        task_suite=suite,
        task_id=TASK_ID,
        task_suite_name=SUITE_NAME,
        camera_name="agentview_image,robot0_eye_in_hand_image",
        obs_type="pixels_agent_pos",
        observation_width=256,
        observation_height=256,
        init_states=True,
        num_steps_wait=10,
        control_mode="relative",
    )
    obs, _ = env.reset()

    # 2. Load pretrained smolvla_base
    print(f"\n[2/4] Loading {MODEL_ID} ...")
    t0 = time.time()
    policy: SmolVLAPolicy = SmolVLAPolicy.from_pretrained(MODEL_ID)
    policy = policy.to(device)
    policy.eval()
    policy.reset()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Config device  : {policy.config.device}")
    print(f"  Input features : {list(policy.config.input_features.keys())}")
    print(f"  Action shape   : {policy.config.output_features}")

    tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)

    # 3. Rollout
    print(f"\n[3/4] Running {N_STEPS}-step rollout ...")
    rollout_frames: list[np.ndarray] = []
    inference_times: list[float] = []

    for step in range(N_STEPS):
        batch = libero_obs_to_smolvla_base_batch(
            obs, task.language, tokenizer, device,
            tokenizer_max_length=policy.config.tokenizer_max_length,
        )

        t_inf = time.time()
        with torch.no_grad():
            # select_action returns (batch_size, action_dim); squeeze for single env
            action_6dof = policy.select_action(batch).squeeze(0).cpu().numpy()
        inference_times.append(time.time() - t_inf)

        # Pad SO100 6-DOF action → LIBERO 7-DOF (add gripper = 0)
        action_7dof = np.append(action_6dof, 0.0)
        action_7dof = np.clip(action_7dof, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action_7dof)

        frame = env.render()
        rollout_frames.append(frame)

        print(
            f"  step {step:3d} | act[:3]={action_6dof[:3].round(3)} "
            f"| inf={inference_times[-1]*1000:.0f}ms"
            f"| success={info.get('is_success', False)}"
        )

        if terminated or truncated:
            print(f"  Episode ended at step {step}")
            obs, _ = env.reset()
            policy.reset()

    env.close()

    # 4. Save
    print(f"\n[4/4] Saving results ...")
    save_episode_frames(rollout_frames, os.path.join(OUTPUT_DIR, "rollout_base.png"))

    avg_ms = np.mean(inference_times) * 1000
    print(f"\n=== Summary ===")
    print(f"  Steps: {N_STEPS} | Avg inference: {avg_ms:.0f} ms | Hz: {1000/avg_ms:.1f}")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}")
    print(f"\nThis demo validates that smolvla_base runs on LIBERO observations.")
    print(f"For task-relevant actions, fine-tune on LIBERO demonstration data.")


if __name__ == "__main__":
    main()
