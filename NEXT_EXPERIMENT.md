# Pending Experiment: ATCNet + expert-from-scratch

## Hypothesis
ATCNet's clean embedding mismatched the EEGNet-trained action expert
(v5/v6 results). The expert needs to be re-trained from scratch on
ATCNet's representation rather than continuing from the v4 checkpoint.

## Plan
1. Skip BASE_CHECKPOINT loading — initialize action expert randomly
2. Use ATCNet (val_acc 57.7%) as frozen encoder
3. Token-level VLM injection
4. Class-balanced sampling, 3000-5000 steps
5. Compare to v4 (EEGNet, 3/4) and v5/v6 (ATCNet on top of v4 expert)

## Trigger
User will remind in the morning to start.

## Implementation note
Need to add NO_BASE_CKPT flag to train_smolvla_eeg_token.py to skip
loading the policy state — initialize SmolVLA with random expert
weights but pretrained VLM backbone (the existing 0.5B SmolVLM2).
