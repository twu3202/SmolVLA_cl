"""Shared utilities for SmolVLA + LIBERO demo scripts."""

import os
import sys
import numpy as np
import torch

LIBERO_PATH = "/Users/r/LIBERO"
LEROBOT_PATH = "/Users/r/lerobot/src"

# Ensure LIBERO is importable
for p in [LIBERO_PATH, LEROBOT_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)


def get_device() -> str:
    """Return best available device (mps > cpu)."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def obs_to_policy_batch(
    obs: dict,
    task_description: str,
    tokenizer,
    device: str,
    state_key: str = "observation.state",
    image_key: str = "observation.images.image",
    image2_key: str = "observation.images.image2",
    max_token_length: int = 48,
) -> dict[str, torch.Tensor]:
    """
    Convert a LiberoEnv observation dict to a SmolVLA-compatible batch.

    LiberoEnv observation structure (pixels_agent_pos mode):
        obs["pixels"]["image"]                 - (H, W, 3) uint8 agentview
        obs["pixels"]["image2"]                - (H, W, 3) uint8 wrist cam
        obs["robot_state"]["eef"]["pos"]       - (3,) float64
        obs["robot_state"]["eef"]["quat"]      - (4,) float64
        obs["robot_state"]["joints"]["pos"]    - (7,) float64

    Returns a dict with:
        observation.images.image               - (1, 3, H, W) float [0,1]
        observation.images.image2              - (1, 3, H, W) float [0,1]
        observation.state                      - (1, STATE_DIM) float
        observation.language.tokens            - (1, T) long
        observation.language.attention_mask    - (1, T) long
    """
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

    batch = {}

    # --- Images (H, W, 3) uint8 → (1, 3, H, W) float32 [0,1] ---
    for src_key, dst_key in [
        ("image", image_key),
        ("image2", image2_key),
    ]:
        img = obs["pixels"][src_key]  # (H, W, 3)
        tensor = torch.from_numpy(img).float() / 255.0  # [0,1]
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        batch[dst_key] = tensor.to(device)

    # --- State: eef_pos(3) + eef_quat(4) + joint_pos(7) = 14 ---
    rs = obs["robot_state"]
    state = np.concatenate([
        rs["eef"]["pos"],       # (3,)
        rs["eef"]["quat"],      # (4,)
        rs["joints"]["pos"],    # (7,)
    ])
    batch[state_key] = torch.from_numpy(state).float().unsqueeze(0).to(device)

    # --- Language tokens ---
    task_with_newline = task_description if task_description.endswith("\n") else task_description + "\n"
    tokens = tokenizer(
        task_with_newline,
        return_tensors="pt",
        padding="max_length",
        max_length=max_token_length,
        truncation=True,
    )
    batch[OBS_LANGUAGE_TOKENS] = tokens["input_ids"].to(device)
    # SmolVLA's embed_prefix appends lang_masks to pad_masks, which is then used
    # in make_att_2d_masks. torch.where expects a bool condition, so cast here.
    batch[OBS_LANGUAGE_ATTENTION_MASK] = tokens["attention_mask"].bool().to(device)

    return batch


def save_episode_frames(frames: list[np.ndarray], path: str, fps: int = 10) -> None:
    """Save a list of (H, W, 3) frames as a video or PNG grid."""
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = len(frames)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        axes[r][c].imshow(frame)
        axes[r][c].set_title(f"step {i}", fontsize=7)
        axes[r][c].axis("off")

    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)
    print(f"Saved frame grid: {path}")


def dummy_dataset_stats(state_dim: int = 14, action_dim: int = 7, device: str = "cpu") -> dict:
    """
    Identity normalization stats (zero mean, unit std) for demo inference
    when no real dataset statistics are available.
    """
    return {
        "observation.state": {
            "mean": torch.zeros(state_dim, device=device),
            "std": torch.ones(state_dim, device=device),
        },
        "action": {
            "mean": torch.zeros(action_dim, device=device),
            "std": torch.ones(action_dim, device=device),
        },
    }
