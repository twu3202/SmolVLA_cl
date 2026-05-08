#!/bin/bash
# Auto-pipeline: wait for EEG download → retrain encoder → train token-level → eval
# Designed to run unattended overnight.
#
# Logs to /tmp/full_pipeline.log
# Status file at /tmp/pipeline_status.txt updated at each stage

set -e
cd /Users/r/Projects/SmolVLA_cl

PY=/opt/anaconda3/envs/lerobot/bin/python
LOG=/tmp/full_pipeline.log
STATUS=/tmp/pipeline_status.txt
DOWNLOAD_LOG=/tmp/eeg_download.log
DOWNLOAD_PID=7102

log() {
    echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG"
    echo "$1" > "$STATUS"
}

log "=== Full Pipeline Started ==="

# ── Stage A: wait for download to finish ─────────────────────────────────────
log "Stage A: waiting for EEG download to complete (PID $DOWNLOAD_PID)"
while ps -p $DOWNLOAD_PID > /dev/null 2>&1; do
    n_files=$(find /tmp/eegmmidb -name '*.edf' 2>/dev/null | wc -l | tr -d ' ')
    n_subj=$(grep -c "intermediate save" "$DOWNLOAD_LOG" 2>/dev/null || echo 0)
    log "  download progress: $n_files / 654 EDF files (~${n_subj}0 subjects done)"
    sleep 600   # check every 10 min
done

if ! grep -q "^Saved " "$DOWNLOAD_LOG"; then
    log "Download did not complete cleanly. Aborting."
    exit 1
fi

log "Stage A done: download complete"
$PY -c "
import numpy as np
d = np.load('./data/eeg_physionet/epochs.npz')
print(f'Final epochs: {d[\"X\"].shape}, classes: {np.bincount(d[\"y\"])}')
" >> "$LOG"

# ── Stage B: retrain EEGNet on full data ─────────────────────────────────────
log "Stage B: retraining EEGNet on full dataset"
# Backup the old (40.7% val_acc) encoder before overwriting
cp ./checkpoints/eeg_encoder/encoder_only.pt \
   ./checkpoints/eeg_encoder/encoder_only_v1_10subj.pt 2>/dev/null || true

$PY train_eeg_encoder.py >> "$LOG" 2>&1
log "Stage B done: EEGNet retrained"

# ── Stage C: train SmolVLA + EEG token-level w/ class balancing, 3000 steps ──
log "Stage C: token-level training (3000 steps, class-balanced)"
SUITE=libero_spatial \
SYNTHETIC_PAIRING=1 \
CLASS_BALANCED=1 \
STEPS=3000 \
RUN_TAG=_balanced_v2 \
PYTHONPATH=/Users/r/LIBERO \
PYTHONUNBUFFERED=1 \
$PY -u train_smolvla_eeg_token.py >> "$LOG" 2>&1
log "Stage C done: token-level training complete"

# ── Stage D: controllability eval ────────────────────────────────────────────
log "Stage D: controllability evaluation"
EEG_CKPT=./checkpoints/libero_spatial_eeg_token_balanced_v2/step_003000.pt \
SUITE=libero_spatial \
PYTHONPATH=/Users/r/LIBERO \
$PY eval_token_eeg.py >> "$LOG" 2>&1

# Save eval artifacts under unique name
cp ./eval_output/controllability_token_libero_spatial.png \
   ./eval_output/controllability_token_balanced_v2.png 2>/dev/null || true

log "Stage D done: evaluation complete"
log "=== Full Pipeline Complete ==="

# Print final verdict from log
echo "" >> "$LOG"
echo "=== FINAL VERDICT ===" >> "$LOG"
grep -A 6 "Per-condition verdict" "$LOG" | tail -8 >> "$LOG"
