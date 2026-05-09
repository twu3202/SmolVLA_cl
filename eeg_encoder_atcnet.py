"""
ATCNet — Attention-based Temporal Convolutional Network for EEG decoding.

Reference: Altaheri et al. (2023) "Physics-informed attention temporal
convolutional network for EEG-based motor imagery classification"
IEEE Trans. Industrial Informatics.

Architecture:
  1. Convolutional Block (EEGNet-style)  — temporal + spatial filters
  2. Sliding-Window split                — K parallel branches
  3. Multi-head Self-Attention per window
  4. Temporal Convolutional Network (TCN) per window — dilated causal convs
  5. Average of branches → embedding + classification head

Same interface as EEGNet:
    encoder.encode(x) → (B, embed_dim)
    encoder(x)        → (embedding, logits)

Trained as drop-in replacement on PhysioNet EEGMMIDB (4-class MI).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ──────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    """EEGNet-style temporal + spatial convolutional feature extractor."""
    def __init__(self, n_channels: int, F1: int = 16, D: int = 2,
                 kernel_size: int = 64, pool1: int = 8, pool2: int = 7,
                 dropout: float = 0.3):
        super().__init__()
        F2 = F1 * D
        self.block = nn.Sequential(
            # Temporal conv (along time axis)
            nn.Conv2d(1, F1, kernel_size=(1, kernel_size),
                      padding=(0, kernel_size // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise spatial conv (per-temporal-filter spatial pattern)
            nn.Conv2d(F1, F2, kernel_size=(n_channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool1)),
            nn.Dropout(dropout),
            # Separable conv (depthwise + pointwise)
            nn.Conv2d(F2, F2, kernel_size=(1, 16),
                      padding=(0, 8), groups=F2, bias=False),
            nn.Conv2d(F2, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool2)),
            nn.Dropout(dropout),
        )
        self.out_channels = F2

    def forward(self, x):                      # x: (B, 1, C, T)
        return self.block(x).squeeze(2)        # → (B, F2, T')


class _AttentionBlock(nn.Module):
    """Multi-head self-attention along the temporal axis."""
    def __init__(self, dim: int, heads: int = 2, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads,
                                          dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                      # x: (B, T, D)
        h = self.norm(x)
        out, _ = self.attn(h, h, h, need_weights=False)
        return x + self.drop(out)


class _TemporalConv(nn.Module):
    """Causal dilated 1-D convolution stack (TCN block)."""
    def __init__(self, dim: int, n_layers: int = 2,
                 kernel_size: int = 4, dropout: float = 0.3):
        super().__init__()
        layers = []
        for i in range(n_layers):
            dilation = 2 ** i
            pad = (kernel_size - 1) * dilation
            layers += [
                nn.Conv1d(dim, dim, kernel_size,
                          padding=pad, dilation=dilation),
                nn.ELU(),
                nn.Dropout(dropout),
            ]
        self.net = nn.Sequential(*layers)
        self.kernel_size = kernel_size

    def forward(self, x):                      # x: (B, D, T)
        out = self.net(x)
        # Causal: trim future-information leak
        return out[..., :x.size(-1)]


# ── ATCNet ───────────────────────────────────────────────────────────────────

class ATCNet(nn.Module):
    """
    Attention-based Temporal Convolutional Network for EEG.

    Args:
        n_channels:   EEG channels (64 for PhysioNet)
        n_timepoints: samples per epoch (320 for 2 s @ 160 Hz)
        n_classes:    output classes (4)
        embed_dim:    latent embedding size (64, matches EEGNet API)
        n_windows:    number of parallel sliding-window branches
        F1, D:        EEGNet conv block hyperparameters
    """

    def __init__(
        self,
        n_channels:   int = 64,
        n_timepoints: int = 320,
        n_classes:    int = 4,
        embed_dim:    int = 64,
        n_windows:    int = 5,
        F1:           int = 16,
        D:            int = 2,
        kernel_size:  int = 64,
        attn_heads:   int = 2,
        tcn_layers:   int = 2,
        dropout:      float = 0.3,
    ):
        super().__init__()
        self.n_windows    = n_windows
        self.embed_dim    = embed_dim
        self.n_classes    = n_classes
        self.n_channels   = n_channels
        self.n_timepoints = n_timepoints

        self.conv_block = _ConvBlock(n_channels, F1=F1, D=D,
                                     kernel_size=kernel_size, dropout=dropout)
        F2 = self.conv_block.out_channels

        # Compute T' (length after conv block) dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            T_prime = self.conv_block(dummy).shape[-1]
        if T_prime < n_windows:
            n_windows = T_prime
            self.n_windows = n_windows
        self.window_len = T_prime - n_windows + 1   # sliding-window length

        # K parallel branches: attention + TCN per window
        self.attn_blocks = nn.ModuleList(
            _AttentionBlock(F2, heads=attn_heads, dropout=dropout)
            for _ in range(n_windows)
        )
        self.tcn_blocks = nn.ModuleList(
            _TemporalConv(F2, n_layers=tcn_layers, dropout=dropout)
            for _ in range(n_windows)
        )

        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.head = nn.Linear(embed_dim, n_classes)

    def _process_window(self, h_win: torch.Tensor, k: int) -> torch.Tensor:
        # h_win: (B, F2, window_len)
        x = h_win.transpose(1, 2)              # (B, T_w, F2)
        x = self.attn_blocks[k](x)             # self-attention
        x = x.transpose(1, 2)                  # (B, F2, T_w)
        x = self.tcn_blocks[k](x)              # temporal conv
        # Last-step pooling (causal: takes the most recent context)
        return x[..., -1]                      # (B, F2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return embedding only."""
        if x.dim() == 3:
            x = x.unsqueeze(1)                 # (B, C, T) → (B, 1, C, T)
        h = self.conv_block(x)                 # (B, F2, T')

        # K sliding windows averaged
        outs = []
        for k in range(self.n_windows):
            start = k
            end   = k + self.window_len
            outs.append(self._process_window(h[..., start:end], k))
        feats = torch.stack(outs, dim=0).mean(dim=0)   # (B, F2)
        return self.embed(feats)                # (B, embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb    = self.encode(x)
        logits = self.head(emb)
        return emb, logits
