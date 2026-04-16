"""
TCN-AE (Temporal Convolutional Network Autoencoder) for vibration anomaly detection.
Based on: Meng et al., Anomaly Detection in Construction Site Vibration Monitoring.
"""

import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    """1-D causal convolution with dilation (no future leakage)."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
        )

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNResidualBlock(nn.Module):
    """TCN residual block: two causal dilated convolutions + skip connection."""

    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        return self.relu(out + residual)


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    TCN encoder that compresses (batch, 1, L) -> (batch, bottleneck_dim).
    Uses dilated causal convolutions with exponentially growing dilation rates.
    """

    def __init__(self, window_size=128, n_channels=32, kernel_size=3,
                 n_blocks=4, bottleneck_dim=16):
        super().__init__()
        self.input_conv = nn.Conv1d(1, n_channels, 1)

        dilations = [2 ** i for i in range(n_blocks)]  # [1, 2, 4, 8, ...]
        self.tcn_blocks = nn.ModuleList(
            [TCNResidualBlock(n_channels, kernel_size, d) for d in dilations]
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_channels, bottleneck_dim)

    def forward(self, x):
        # x: (batch, 1, L)
        out = self.input_conv(x)
        for block in self.tcn_blocks:
            out = block(out)
        out = self.pool(out).squeeze(-1)  # (batch, n_channels)
        out = self.fc(out)                # (batch, bottleneck_dim)
        return out


class Decoder(nn.Module):
    """
    Decoder that reconstructs (batch, bottleneck_dim) -> (batch, 1, L)
    using transposed convolutions.
    """

    def __init__(self, window_size=128, n_channels=32, kernel_size=3,
                 n_blocks=4, bottleneck_dim=16):
        super().__init__()
        self.window_size = window_size
        self.n_channels = n_channels

        self.fc = nn.Linear(bottleneck_dim, n_channels * (window_size // (2 ** n_blocks)))
        self.init_length = window_size // (2 ** n_blocks)

        layers = []
        for i in range(n_blocks):
            layers.append(
                nn.ConvTranspose1d(n_channels, n_channels, kernel_size=4,
                                   stride=2, padding=1)
            )
            layers.append(nn.BatchNorm1d(n_channels))
            layers.append(nn.ReLU())
        self.deconv = nn.Sequential(*layers)

        self.output_conv = nn.Conv1d(n_channels, 1, 1)

    def forward(self, z):
        # z: (batch, bottleneck_dim)
        out = self.fc(z)
        out = out.view(-1, self.n_channels, self.init_length)
        out = self.deconv(out)
        # Trim or pad to exact window_size
        out = out[:, :, :self.window_size]
        out = self.output_conv(out)
        return out  # (batch, 1, window_size)


# ---------------------------------------------------------------------------
# Full Autoencoder
# ---------------------------------------------------------------------------

class TCNAE(nn.Module):
    """TCN-AE: Temporal Convolutional Network Autoencoder."""

    def __init__(self, window_size=128, n_channels=32, kernel_size=3,
                 n_blocks=4, bottleneck_dim=16):
        super().__init__()
        self.encoder = Encoder(window_size, n_channels, kernel_size,
                               n_blocks, bottleneck_dim)
        self.decoder = Decoder(window_size, n_channels, kernel_size,
                               n_blocks, bottleneck_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


# ---------------------------------------------------------------------------
# Utility helpers (sliding window, threshold)
# ---------------------------------------------------------------------------

def sliding_window(data: np.ndarray, window_size: int, step: int = 1) -> np.ndarray:
    """
    Slice a 1-D array into overlapping windows.

    Parameters
    ----------
    data : np.ndarray, shape (N,)
    window_size : int
    step : int

    Returns
    -------
    windows : np.ndarray, shape (n_windows, window_size)
    """
    windows = np.lib.stride_tricks.sliding_window_view(data, window_size)[::step]
    return windows.copy()


def compute_adaptive_threshold(errors: np.ndarray, alpha: float = 3.2) -> float:
    """
    Adaptive threshold: mu + alpha * sigma.

    Parameters
    ----------
    errors : np.ndarray  – per-window reconstruction MSE on **training** data
    alpha  : float       – multiplier (paper recommends 3.2)

    Returns
    -------
    threshold : float
    """
    mu = errors.mean()
    sigma = errors.std()
    return float(mu + alpha * sigma)
