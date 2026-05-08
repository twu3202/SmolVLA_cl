"""
PyTorch Dataset for LIBERO HDF5 demonstration files.

Each sample is one (observation, action_chunk) pair extracted from a demo.
The action chunk has length `chunk_size`; chunks near the end of a demo are
zero-padded and marked with `action_is_pad`.

HDF5 structure per demo:
    obs/agentview_rgb     (T, 128, 128, 3)  uint8
    obs/eye_in_hand_rgb   (T, 128, 128, 3)  uint8
    obs/ee_pos            (T, 3)            float64  — end-effector XYZ
    obs/ee_ori            (T, 3)            float64  — axis-angle orientation
    obs/joint_states      (T, 7)            float64  — joint positions
    actions               (T, 7)            float64  — delta control + gripper
"""

import glob
import os
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

LIBERO_DATASETS = "/Users/r/LIBERO/libero/datasets"


# ── State construction ────────────────────────────────────────────────────────

def build_state(ee_pos: np.ndarray, ee_ori: np.ndarray, joint_pos: np.ndarray) -> np.ndarray:
    """
    Concatenate ee_pos(3) + quat_from_axisangle(4) + joint_pos(7) = 14-dim state.

    Uses small-angle approximation to convert axis-angle → quaternion:
        quat ≈ [axis*sin(θ/2), cos(θ/2)] ≈ [axis*θ/2, 1] for small θ
    This is fine for normalizing; the policy sees normalized values anyway.
    """
    angle = np.linalg.norm(ee_ori) + 1e-12
    axis  = ee_ori / angle
    quat  = np.array([*(axis * np.sin(angle / 2)), np.cos(angle / 2)], dtype=np.float32)
    return np.concatenate([ee_pos.astype(np.float32), quat, joint_pos.astype(np.float32)])


# ── Dataset statistics ────────────────────────────────────────────────────────

def compute_dataset_stats(hdf5_paths: list[str], max_demos_per_file: int = 50) -> dict:
    """
    Compute mean/std of state and action across all demos.
    Returns dict with keys 'observation.state' and 'action', each with 'mean'/'std'.
    """
    all_states  = []
    all_actions = []

    for path in hdf5_paths:
        with h5py.File(path, "r") as f:
            demo_keys = list(f["data"].keys())[:max_demos_per_file]
            for dk in demo_keys:
                d = f["data"][dk]
                T = len(d["actions"])
                for t in range(T):
                    state = build_state(
                        d["obs/ee_pos"][t],
                        d["obs/ee_ori"][t],
                        d["obs/joint_states"][t],
                    )
                    all_states.append(state)
                all_actions.append(d["actions"][:].astype(np.float32))

    states  = np.array(all_states,  dtype=np.float32)   # (N, 14)
    actions = np.vstack(all_actions).astype(np.float32)  # (N_total, 7)

    eps = 1e-8
    return {
        "observation.state": {
            "mean": torch.from_numpy(states.mean(axis=0)),
            "std":  torch.from_numpy(states.std(axis=0).clip(eps)),
        },
        "action": {
            "mean": torch.from_numpy(actions.mean(axis=0)),
            "std":  torch.from_numpy(actions.std(axis=0).clip(eps)),
        },
    }


# ── Dataset ───────────────────────────────────────────────────────────────────

class LiberoDataset(Dataset):
    """
    One sample = (obs_t, action_chunk_t) for a random (demo, timestep).

    Args:
        suite:          LIBERO suite name, e.g. "libero_spatial"
        chunk_size:     number of future actions per sample (matches SmolVLA chunk_size)
        tokenizer:      HuggingFace tokenizer for task language
        stats:          normalization stats from compute_dataset_stats()
        max_token_len:  max token length for language
        max_demos:      cap demos per HDF5 file (None = all)
        img_size:       resize images to this square size (None = keep original 128)
    """

    def __init__(
        self,
        suite: str = "libero_spatial",
        chunk_size: int = 50,
        tokenizer=None,
        stats: dict | None = None,
        max_token_len: int = 48,
        max_demos: int | None = None,
        img_size: int | None = None,
    ):
        self.chunk_size    = chunk_size
        self.tokenizer     = tokenizer
        self.stats         = stats
        self.max_token_len = max_token_len
        self.img_size      = img_size

        dataset_dir = os.path.join(LIBERO_DATASETS, suite)
        hdf5_paths  = sorted(glob.glob(os.path.join(dataset_dir, "*.hdf5")))
        if not hdf5_paths:
            raise FileNotFoundError(f"No HDF5 files in {dataset_dir}")

        # Index: list of (hdf5_path, demo_key, task_language, T)
        self._index: list[tuple[str, str, str, int]] = []

        for path in hdf5_paths:
            # Derive task language from filename
            task_lang = (
                Path(path).stem
                .replace("_demo", "")
                .replace("_", " ")
            )
            with h5py.File(path, "r") as f:
                demo_keys = sorted(f["data"].keys())
                if max_demos:
                    demo_keys = demo_keys[:max_demos]
                for dk in demo_keys:
                    T = len(f["data"][dk]["actions"])
                    self._index.append((path, dk, task_lang, T))

        print(
            f"LiberoDataset: {suite} | "
            f"{len(hdf5_paths)} tasks | "
            f"{len(self._index)} demos | "
            f"chunk_size={chunk_size}"
        )

        # Pre-tokenize all unique task descriptions
        if tokenizer is not None:
            unique_tasks = list({lang for _, _, lang, _ in self._index})
            self._token_cache: dict[str, dict] = {}
            for task in unique_tasks:
                task_nl = task if task.endswith("\n") else task + "\n"
                enc = tokenizer(
                    task_nl,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=max_token_len,
                    truncation=True,
                )
                self._token_cache[task] = {
                    "input_ids":      enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0).bool(),
                }

        # Flat index: one entry per (demo, timestep)
        self._flat: list[tuple[int, int]] = []  # (demo_idx, t)
        for demo_idx, (_, _, _, T) in enumerate(self._index):
            for t in range(T):
                self._flat.append((demo_idx, t))

        print(f"  Total samples: {len(self._flat)}")

    def __len__(self) -> int:
        return len(self._flat)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        demo_idx, t = self._flat[idx]
        hdf5_path, demo_key, task_lang, T = self._index[demo_idx]

        with h5py.File(hdf5_path, "r") as f:
            d = f["data"][demo_key]

            # Images
            img_agent = d["obs/agentview_rgb"][t]     # (H, W, 3) uint8
            img_wrist = d["obs/eye_in_hand_rgb"][t]   # (H, W, 3) uint8

            # State
            state = build_state(
                d["obs/ee_pos"][t],
                d["obs/ee_ori"][t],
                d["obs/joint_states"][t],
            )  # (14,)

            # Action chunk [t, t+chunk_size) with end-padding
            chunk_end   = min(t + self.chunk_size, T)
            raw_actions = d["actions"][t:chunk_end].astype(np.float32)  # (<=chunk_size, 7)

        # ── Images → (3, H, W) float [0,1] ──────────────────────────────────
        def to_tensor(img: np.ndarray) -> torch.Tensor:
            t_ = torch.from_numpy(img).float().div(255.0).permute(2, 0, 1)
            if self.img_size and img.shape[0] != self.img_size:
                import torch.nn.functional as F
                t_ = F.interpolate(
                    t_.unsqueeze(0), size=(self.img_size, self.img_size), mode="bilinear"
                ).squeeze(0)
            return t_

        img_agent_t = to_tensor(img_agent)
        img_wrist_t = to_tensor(img_wrist)

        # ── State normalization ───────────────────────────────────────────────
        state_t = torch.from_numpy(state)
        if self.stats is not None:
            mean = self.stats["observation.state"]["mean"]
            std  = self.stats["observation.state"]["std"]
            state_t = (state_t - mean) / (std + 1e-8)

        # ── Action chunk + padding mask ───────────────────────────────────────
        n_real   = len(raw_actions)
        n_pad    = self.chunk_size - n_real
        actions  = torch.from_numpy(raw_actions)

        if self.stats is not None:
            mean = self.stats["action"]["mean"]
            std  = self.stats["action"]["std"]
            actions = (actions - mean) / (std + 1e-8)

        if n_pad > 0:
            actions = torch.cat([actions, torch.zeros(n_pad, 7)], dim=0)

        action_is_pad = torch.zeros(self.chunk_size, dtype=torch.bool)
        if n_pad > 0:
            action_is_pad[n_real:] = True

        # ── Language tokens ───────────────────────────────────────────────────
        tokens = self._token_cache[task_lang]

        sample = {
            "observation.images.image":          img_agent_t,   # (3, H, W)
            "observation.images.image2":          img_wrist_t,   # (3, H, W)
            "observation.state":                  state_t,        # (14,)
            "action":                             actions,        # (chunk_size, 7)
            "action_is_pad":                      action_is_pad,  # (chunk_size,)
            "observation.language.tokens":        tokens["input_ids"],       # (T_lang,)
            "observation.language.attention_mask": tokens["attention_mask"],  # (T_lang,) bool
        }
        return sample


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Stack list of samples into a batched dict."""
    keys = batch[0].keys()
    return {k: torch.stack([s[k] for s in batch]) for k in keys}
