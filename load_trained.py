"""
Helper: load a trained LIBERO SmolVLA checkpoint for evaluation.

Usage in eval scripts:
    from load_trained import load_trained_policy
    policy, tokenizer, stats = load_trained_policy(suite="libero_spatial", device="mps")
"""

import os
import glob
import torch
from transformers import AutoTokenizer

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from libero_smolvla_config import make_libero_smolvla_config


def load_trained_policy(
    suite: str = "libero_spatial",
    checkpoint_dir: str | None = None,
    step: int | None = None,
    device: str = "mps",
) -> tuple[SmolVLAPolicy, object, dict]:
    """
    Load a SmolVLA policy trained by train_smolvla_libero.py.

    Args:
        suite:           LIBERO suite name (used to find default checkpoint dir).
        checkpoint_dir:  Override checkpoint directory. Defaults to ./checkpoints/<suite>.
        step:            Load specific step checkpoint. Defaults to latest.
        device:          Torch device.

    Returns:
        (policy, tokenizer, stats)
    """
    if checkpoint_dir is None:
        checkpoint_dir = f"./checkpoints/{suite}"

    stats_path = os.path.join(checkpoint_dir, "dataset_stats.pt")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"No dataset_stats.pt in {checkpoint_dir}. Run train_smolvla_libero.py first."
        )

    stats = torch.load(stats_path, map_location="cpu", weights_only=False)

    # Find checkpoint file
    if step is not None:
        ckpt_path = os.path.join(checkpoint_dir, f"step_{step:06d}.pt")
    else:
        ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "step_*.pt")))
        final = os.path.join(checkpoint_dir, "policy_final.pt")
        if os.path.exists(final):
            ckpt_path = final
            is_weights_only = True
        elif ckpts:
            ckpt_path = ckpts[-1]
            is_weights_only = False
        else:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    print(f"Loading checkpoint: {ckpt_path}")

    cfg = make_libero_smolvla_config(device)
    cfg.load_vlm_weights = False   # weights come from checkpoint, not HuggingFace
    policy = SmolVLAPolicy(cfg)

    if "is_weights_only" in dir() and is_weights_only:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    else:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["policy_state"]
        print(f"  Loaded from step {ckpt.get('step', '?')}")

    policy.load_state_dict(state_dict)
    policy = policy.to(device)
    policy.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.vlm_model_name)
    return policy, tokenizer, stats
