#!/usr/bin/env python
"""
Quick import + shape test — runs in ~10 seconds, no model download needed.

Usage:
    PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python quick_test.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

LIBERO_PATH = "/Users/r/LIBERO"
if LIBERO_PATH not in sys.path:
    sys.path.insert(0, LIBERO_PATH)

import numpy as np
import torch

PASS = "✓"
FAIL = "✗"

results = []


def check(label, fn):
    try:
        fn()
        results.append((PASS, label))
        print(f"  {PASS}  {label}")
    except Exception as e:
        results.append((FAIL, label))
        print(f"  {FAIL}  {label}: {e}")


print("=== SmolVLA + LIBERO Quick Test ===\n")

# ── Imports ───────────────────────────────────────────────────────────────────
print("Imports:")

def _check_torch():
    assert torch.__version__
    assert torch.backends.mps.is_available(), "MPS not available (M-series Mac expected)"

def _check_lerobot():
    import lerobot
    assert lerobot.__version__

def _check_libero():
    from libero.libero import benchmark
    bench = benchmark.get_benchmark_dict()
    assert len(bench) >= 4

def _check_libero_env_cls():
    from lerobot.envs.libero import LiberoEnv
    assert LiberoEnv

def _check_smolvla():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    assert SmolVLAPolicy

def _check_smolvla_config():
    from libero_smolvla_config import make_libero_smolvla_config
    cfg = make_libero_smolvla_config("cpu")
    assert cfg.vlm_model_name

def _check_utils():
    from utils import get_device, dummy_dataset_stats
    d = get_device()
    assert d in ("mps", "cpu")
    stats = dummy_dataset_stats(14, 7)
    assert "observation.state" in stats

check("torch + MPS available", _check_torch)
check("lerobot package", _check_lerobot)
check("libero benchmarks", _check_libero)
check("lerobot LiberoEnv class", _check_libero_env_cls)
check("SmolVLAPolicy class", _check_smolvla)
check("libero_smolvla_config", _check_smolvla_config)
check("utils module", _check_utils)

# ── LIBERO env (no rendering, just init + reset) ──────────────────────────────
print("\nLiberoEnv:")

def _check_env_reset():
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv
    bench = benchmark.get_benchmark_dict()
    suite = bench["libero_spatial"]()
    env = LiberoEnv(
        task_suite=suite,
        task_id=0,
        task_suite_name="libero_spatial",
        camera_name="agentview_image,robot0_eye_in_hand_image",
        obs_type="pixels_agent_pos",
        observation_width=128,  # small for speed
        observation_height=128,
        init_states=False,       # skip loading init states for speed
        num_steps_wait=0,
        control_mode="relative",
    )
    obs, info = env.reset()

    assert obs["pixels"]["image"].shape == (128, 128, 3)
    assert obs["pixels"]["image2"].shape == (128, 128, 3)
    assert obs["robot_state"]["eef"]["pos"].shape == (3,)
    assert obs["robot_state"]["joints"]["pos"].shape == (7,)

    action = env.action_space.sample()
    obs2, rew, term, trunc, info2 = env.step(action)
    assert obs2["pixels"]["image"].shape == (128, 128, 3)
    env.close()

check("reset + step + close", _check_env_reset)

# ── Batch construction ────────────────────────────────────────────────────────
print("\nBatch construction (no model download):")

def _check_batch():
    from transformers import AutoTokenizer
    from utils import obs_to_policy_batch
    from libero_smolvla_config import LIBERO_AGENTVIEW_KEY, LIBERO_WRIST_KEY, LIBERO_STATE_KEY
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

    # Fake LIBERO observation
    obs = {
        "pixels": {
            "image":  np.zeros((256, 256, 3), dtype=np.uint8),
            "image2": np.zeros((256, 256, 3), dtype=np.uint8),
        },
        "robot_state": {
            "eef":     {"pos": np.zeros(3), "quat": np.array([0, 0, 0, 1.])},
            "joints":  {"pos": np.zeros(7)},
            "gripper": {"qpos": np.zeros(2)},
        },
    }

    batch = obs_to_policy_batch(
        obs, "pick up the bowl\n", tokenizer, "cpu",
        state_key=LIBERO_STATE_KEY,
        image_key=LIBERO_AGENTVIEW_KEY,
        image2_key=LIBERO_WRIST_KEY,
    )

    assert batch[LIBERO_AGENTVIEW_KEY].shape == (1, 3, 256, 256)
    assert batch[LIBERO_WRIST_KEY].shape     == (1, 3, 256, 256)
    assert batch[LIBERO_STATE_KEY].shape     == (1, 14)
    assert "observation.language.tokens" in batch
    # SmolVLA's attention mask must be bool (not long) for torch.where
    assert batch[OBS_LANGUAGE_ATTENTION_MASK].dtype == torch.bool

check("obs_to_policy_batch shapes", _check_batch)

# ── SmolVLA config instantiation ──────────────────────────────────────────────
print("\nSmolVLA config (no weight download):")

def _check_policy_init():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from libero_smolvla_config import make_libero_smolvla_config

    cfg = make_libero_smolvla_config("cpu")
    cfg.load_vlm_weights = False  # skip downloading weights
    policy = SmolVLAPolicy(cfg)
    policy.eval()
    assert next(policy.parameters()).device.type == "cpu"

check("SmolVLA instantiation (random weights)", _check_policy_init)

# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(1 for r, _ in results if r == PASS)
total  = len(results)
print(f"\n{'='*40}")
print(f"Results: {passed}/{total} passed")

if passed == total:
    print("\nAll checks passed! Ready to run full demos:")
    print("  PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python demo_libero_env.py")
    print("  PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python demo_smolvla_libero.py")
    print("  PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python demo_smolvla_base_libero.py")
else:
    failed = [label for r, label in results if r == FAIL]
    print(f"\nFailed: {failed}")
