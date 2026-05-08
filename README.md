# SmolVLA + LIBERO Demo

Early-stage demonstration of [SmolVLA](https://huggingface.co/lerobot/smolvla_base) (Vision-Language-Action model) running on LIBERO robot arm simulation on a Mac (no NVIDIA GPU — uses Apple MPS).

## What this shows

| File | What it does |
|------|-------------|
| `quick_test.py` | Sanity check — verifies all imports and tensor shapes (~15s) |
| `demo_libero_env.py` | LIBERO env via lerobot's `LiberoEnv` wrapper, random policy, saves frames |
| `demo_smolvla_libero.py` | **Main demo** — LIBERO-configured SmolVLA (7-DOF, 2 cameras), random-init expert |
| `demo_smolvla_base_libero.py` | Loads pretrained `lerobot/smolvla_base` (SO100) adapted to LIBERO observations |
| `libero_smolvla_config.py` | LIBERO-specific `SmolVLAConfig` (14-dim state, 7-dim action, 2 cameras) |
| `utils.py` | Shared helpers: `obs_to_policy_batch`, `dummy_dataset_stats`, `save_episode_frames` |

## Environment setup

Python 3.12 with both `lerobot` (from source) and `libero` (via PYTHONPATH):

```bash
# lerobot installed from /Users/r/lerobot in the `lerobot` conda env
conda env list   # should show: lerobot, libero

# Run any script with:
PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python <script.py>
```

## Quickstart

```bash
cd /Users/r/Projects/SmolVLA_cl

# 1. Verify setup (no model download needed, ~15s)
PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python quick_test.py

# 2. LIBERO env sanity check with random actions
PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python demo_libero_env.py

# 3. Main demo: SmolVLA pipeline on LIBERO (randomly-init action expert)
#    Downloads SmolVLM2-500M backbone (~1GB) on first run
PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python demo_smolvla_libero.py

# 4. Pretrained smolvla_base on LIBERO observations (cross-embodiment test)
#    Downloads lerobot/smolvla_base (~1GB) on first run
PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python demo_smolvla_base_libero.py

# Run a different LIBERO suite/task
SUITE=libero_object TASK_ID=2 STEPS=30 PYTHONPATH=/Users/r/LIBERO \
    /opt/anaconda3/envs/lerobot/bin/python demo_smolvla_libero.py
```

## Performance on M5 (MPS)

SmolVLA uses **action chunking**: one VLM call generates 50 actions, which are executed sequentially.

| Phase | Latency (random weights) |
|-------|--------------------------|
| First inference (chunk of 50) | ~5–6 s |
| Steps 1–49 (pop from queue) | ~1 ms |
| Effective throughput | ~9 actions/s |

With the full VLM backbone loaded (SmolVLM2-500M), expect ~8–15s per chunk on MPS.

## LIBERO feature layout

```
observation.images.image    (1, 3, 256, 256)  ← agentview camera
observation.images.image2   (1, 3, 256, 256)  ← wrist camera
observation.state           (1, 14)           ← eef_pos(3) + eef_quat(4) + joint_pos(7)
observation.language.tokens (1, 48)           ← tokenized task description
action                      (7,)              ← delta_xyz(3) + delta_rpy(3) + gripper(1)
```

## Next steps: fine-tuning on LIBERO data

```bash
# Download LIBERO demonstration dataset (from HuggingFace)
# Then train SmolVLA with LIBERO config:
PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python -m lerobot.scripts.train \
    --policy.type=smolvla \
    --dataset.repo_id=<user>/libero_spatial_lerobot \
    --policy.device=mps \
    --batch_size=4 \
    --steps=10000
```

## Architecture

```
LIBERO sim (robosuite/MuJoCo)
      │ OffScreenRenderEnv
      ▼
lerobot LiberoEnv (gymnasium wrapper)
      │ obs = {pixels: {image, image2}, robot_state: {eef, joints, gripper}}
      ▼
obs_to_policy_batch()
      │ normalize images [0,1], stack state, tokenize task
      ▼
SmolVLAPolicy.select_action(batch)
      │ SmolVLM2-500M VLM backbone (vision + language encoding)
      │ Flow-matching action expert (generates 50-action chunk)
      │ Pops one action from queue per step
      ▼
env.step(action)  ← 7-DOF delta control
```
