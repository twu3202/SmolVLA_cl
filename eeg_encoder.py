"""
EEGNet: compact depthwise-separable CNN for EEG classification.

Reference: Lawhern et al. (2018) EEGNet: a compact CNN for EEG-based BCIs.
Adapted for PyTorch + Apple MPS.

Input shape:  (batch, 1, n_channels, n_timepoints)   e.g. (B, 1, 64, 320)
Output:
  embedding:  (batch, embed_dim)                      e.g. (B, 64)
  logits:     (batch, n_classes)                      for pretraining only

Usage:
    from eeg_encoder import EEGNet
    model = EEGNet(n_channels=64, n_timepoints=320, n_classes=4, embed_dim=64)
    emb, logits = model(x)   # x: (B, 1, 64, 320)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DepthwiseConv2d(nn.Module):
    """Spatial filter: one conv per input channel, no cross-channel mixing."""
    def __init__(self, in_ch: int, depth_multiplier: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch * depth_multiplier,
            kernel_size=(in_ch, 1),    # full spatial (all channels at once)
            groups=in_ch,
            bias=False,
        )
    def forward(self, x):
        return self.conv(x)


class _SeparableConv2d(nn.Module):
    """Depthwise + pointwise conv for efficient temporal feature mixing."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        super().__init__()
        self.depthwise  = nn.Conv2d(in_ch, in_ch, (1, kernel_size),
                                    padding=(0, kernel_size // 2),
                                    groups=in_ch, bias=False)
        self.pointwise  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class EEGNet(nn.Module):
    """
    EEGNet encoder.

    Args:
        n_channels:    EEG channels (e.g. 64)
        n_timepoints:  samples per epoch (e.g. 320 for 2 s @ 160 Hz)
        n_classes:     output classes for pretraining head
        embed_dim:     size of the latent embedding (fed to SmolVLA)
        F1:            number of temporal filters (default 8)
        D:             depth multiplier for spatial filters (default 2)
        F2:            number of separable filters (default F1*D = 16)
        dropout:       dropout rate
    """

    def __init__(
        self,
        n_channels:   int = 64,
        n_timepoints: int = 320,
        n_classes:    int = 4,
        embed_dim:    int = 64,
        F1:           int = 8,
        D:            int = 2,
        dropout:      float = 0.5,
    ):
        super().__init__()
        F2 = F1 * D

        # ── Block 1: temporal conv + spatial depthwise ────────────────────────
        self.block1 = nn.Sequential(
            # Temporal convolution (1, 64) → captures 0.5s patterns
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
            # Spatial depthwise: learns a spatial filter per temporal filter
            _DepthwiseConv2d(F1, depth_multiplier=D),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),   # reduce time axis by 4
            nn.Dropout(dropout),
        )

        # ── Block 2: separable (depthwise + pointwise) ────────────────────────
        self.block2 = nn.Sequential(
            _SeparableConv2d(F2, F2, kernel_size=16),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),   # reduce time axis by 8
            nn.Dropout(dropout),
        )

        # ── Compute flattened feature size dynamically ────────────────────────
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            # block1: spatial depthwise collapses the channel dim to 1
            out1  = self.block1(dummy)         # (1, F2, 1, T/4)
            out2  = self.block2(out1)          # (1, F2, 1, T/32)
            flat  = out2.flatten(1).shape[1]

        # ── Embedding projection ───────────────────────────────────────────────
        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # ── Classification head (used only during pretraining) ─────────────────
        self.head = nn.Linear(embed_dim, n_classes)

        self.embed_dim   = embed_dim
        self.n_classes   = n_classes
        self.n_channels  = n_channels
        self.n_timepoints = n_timepoints

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return embedding only.  x: (B, 1, C, T) or (B, C, T)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)           # (B, C, T) → (B, 1, C, T)
        return self.embed(self.block2(self.block1(x)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (embedding, logits).  x: (B, 1, C, T) or (B, C, T)."""
        emb    = self.encode(x)
        logits = self.head(emb)
        return emb, logits
