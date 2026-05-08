#!/usr/bin/env python
"""
Download and preprocess PhysioNet EEGMMIDB (EEG Motor Movement/Imagery Dataset).

Downloads motor-imagery runs for all 109 subjects and extracts 2-second epochs.

4 classes:
  0 — left fist MI    (runs 4, 8, 12 · event T1)
  1 — right fist MI   (runs 4, 8, 12 · event T2)
  2 — both fists MI   (runs 5, 9, 13  · event T1)
  3 — both feet MI    (runs 5, 9, 13  · event T2)

Output:
  ./data/eeg_physionet/epochs.npz   — X:(N,64,320)  y:(N,)  float32 / int64
  ./data/eeg_physionet/meta.json    — channel names, sfreq, class map

Usage:
    /opt/anaconda3/envs/lerobot/bin/python download_eeg_data.py
    /opt/anaconda3/envs/lerobot/bin/python download_eeg_data.py --subjects 20
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import mne
mne.set_log_level("ERROR")
from mne.datasets import eegbci

# ── Config ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--subjects",  type=int,   default=109,  help="# subjects (1-109)")
parser.add_argument("--out",       default="./data/eeg_physionet")
parser.add_argument("--data_path", default="/tmp/eegmmidb")
parser.add_argument("--t_epoch",   type=float, default=2.0,  help="epoch length in seconds")
parser.add_argument("--tmin",      type=float, default=0.5,  help="onset offset (s)")
args = parser.parse_args()

SFREQ    = 160          # PhysioNet recording rate
T_POINTS = int(SFREQ * args.t_epoch)   # 320 at 160 Hz, 2 s

# Run mapping: (runs, event→class)
RUN_SETS = [
    ([4, 8, 12], {"T1": 0, "T2": 1}),   # left fist (0), right fist (1)
    ([5, 9, 13], {"T1": 2, "T2": 3}),   # both fists (2), both feet (3)
]

os.makedirs(args.out, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def bandpass(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """4-40 Hz bandpass + notch at 60 Hz."""
    raw.filter(4.0, 40.0, fir_design="firwin", verbose=False)
    raw.notch_filter(60.0, verbose=False)
    return raw


def extract_epochs(raw: mne.io.BaseRaw, event_map: dict[str, int],
                   tmin: float, t_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) arrays for epochs matching event_map."""
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    Xs, ys = [], []
    for label, cls in event_map.items():
        if label not in event_id:
            continue
        eid = event_id[label]
        mask = events[:, 2] == eid
        onsets = events[mask, 0]     # sample indices
        offset = int(tmin * raw.info["sfreq"])
        for onset in onsets:
            start = onset + offset
            end   = start + t_points
            if end > len(raw.times):
                continue
            data = raw.get_data(start=start, stop=end)   # (64, T)
            # z-score per channel
            mu  = data.mean(axis=1, keepdims=True)
            std = data.std(axis=1,  keepdims=True) + 1e-8
            data = (data - mu) / std
            Xs.append(data.astype(np.float32))
            ys.append(cls)
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)


# ── Main ──────────────────────────────────────────────────────────────────────

all_X, all_y = [], []
n_subjects = min(args.subjects, 109)
ch_names = None

print(f"Downloading PhysioNet EEGMMIDB — {n_subjects} subjects")
print(f"Epoch: {args.t_epoch}s ({T_POINTS} points) · onset offset: {args.tmin}s")

skipped = 0
for subj_idx, subj in enumerate(range(1, n_subjects + 1), 1):
    Xs_subj, ys_subj = [], []

    for runs, event_map in RUN_SETS:
        try:
            paths = eegbci.load_data(subjects=subj, runs=runs,
                                     path=args.data_path, update_path=True,
                                     verbose=False)
        except Exception as e:
            skipped += 1
            continue

        for path in paths:
            try:
                raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
                eegbci.standardize(raw)           # fix channel names to 10-10
                raw.set_montage("standard_1005", verbose=False)
                bandpass(raw)
                X, y = extract_epochs(raw, event_map, args.tmin, T_POINTS)
                if len(X):
                    Xs_subj.append(X)
                    ys_subj.append(y)
                if ch_names is None:
                    ch_names = raw.ch_names
            except Exception:
                continue

    if Xs_subj:
        all_X.append(np.concatenate(Xs_subj))
        all_y.append(np.concatenate(ys_subj))

    n_ep = sum(len(x) for x in Xs_subj) if Xs_subj else 0
    print(f"  [{subj_idx:3d}/{n_subjects}] subj {subj:03d} — {n_ep} epochs", flush=True)

# ── Concatenate and save ──────────────────────────────────────────────────────
X = np.concatenate(all_X)   # (N, 64, 320)
y = np.concatenate(all_y)   # (N,)

out_path = os.path.join(args.out, "epochs.npz")
np.savez_compressed(out_path, X=X, y=y)

meta = {
    "n_epochs":   int(len(X)),
    "n_channels": int(X.shape[1]),
    "t_points":   int(X.shape[2]),
    "sfreq":      SFREQ,
    "t_epoch_s":  args.t_epoch,
    "tmin":       args.tmin,
    "n_subjects": n_subjects,
    "skipped_files": skipped,
    "classes": {
        "0": "left_fist_MI",
        "1": "right_fist_MI",
        "2": "both_fists_MI",
        "3": "both_feet_MI",
    },
    "ch_names": ch_names or [],
}
with open(os.path.join(args.out, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

cls_counts = {i: int((y == i).sum()) for i in range(4)}
print(f"\nSaved {len(X)} epochs → {out_path}")
print(f"Shape: X={X.shape}  y={y.shape}")
print(f"Class counts: {cls_counts}")
print(f"  0=left_fist {cls_counts[0]}  1=right_fist {cls_counts[1]}"
      f"  2=both_fists {cls_counts[2]}  3=both_feet {cls_counts[3]}")
