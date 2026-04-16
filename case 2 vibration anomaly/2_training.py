"""
2_training.py
=============
Train the TCN-AE model on normal vibration data.

Reads
-----
data/train_normal.csv

Outputs
-------
model/tcn_ae.pth              – trained model weights
model/train_stats.npz         – mu / sigma of training reconstruction errors
output/training_loss.png      – loss curve
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))
from tcn_ae_model import TCNAE, sliding_window, compute_adaptive_threshold

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WINDOW_SIZE = 128       # sliding window length (samples)
STEP = 32               # sliding window step
BATCH_SIZE = 64
EPOCHS = 80
LR = 1e-3
N_CHANNELS = 32
KERNEL_SIZE = 3
N_BLOCKS = 4            # dilation rates: 1, 2, 4, 8
BOTTLENECK_DIM = 16
ALPHA = 3.2             # threshold multiplier

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
VIS_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess(csv_path: str):
    """Load CSV, Z-score normalise, return (normalised_signal, mu, sigma)."""
    df = pd.read_csv(csv_path)
    raw = df["acceleration_g"].values.astype(np.float32)
    mu = raw.mean()
    sigma = raw.std()
    normed = (raw - mu) / (sigma + 1e-8)
    return normed, mu, sigma


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")

    # --- Load training data ---
    train_path = os.path.join(DATA_DIR, "train_normal.csv")
    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} not found. Run 1_generate_data.py first.")
        sys.exit(1)

    signal, data_mu, data_sigma = load_and_preprocess(train_path)
    print(f"Training signal length: {len(signal)} samples")

    # --- Sliding window ---
    windows = sliding_window(signal, WINDOW_SIZE, STEP)
    print(f"Number of training windows: {windows.shape[0]}")

    # Reshape to (N, 1, L) for Conv1d
    X = torch.tensor(windows, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X, X)  # target = input (autoencoder)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=False)

    # --- Build model ---
    model = TCNAE(
        window_size=WINDOW_SIZE,
        n_channels=N_CHANNELS,
        kernel_size=KERNEL_SIZE,
        n_blocks=N_BLOCKS,
        bottleneck_dim=BOTTLENECK_DIM,
    ).to(DEVICE)
    print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --- Training loop ---
    loss_history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            x_hat = model(xb)
            loss = criterion(x_hat, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(dataset)
        loss_history.append(epoch_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  Loss: {epoch_loss:.6f}")

    # --- Save model ---
    model_path = os.path.join(MODEL_DIR, "tcn_ae.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved -> {model_path}")

    # --- Compute training reconstruction errors for threshold ---
    model.eval()
    all_errors = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            x_hat = model(xb)
            mse = ((xb - x_hat) ** 2).mean(dim=(1, 2))  # per-window MSE
            all_errors.append(mse.cpu().numpy())
    all_errors = np.concatenate(all_errors)

    err_mu = all_errors.mean()
    err_sigma = all_errors.std()
    threshold = compute_adaptive_threshold(all_errors, alpha=ALPHA)
    print(f"Training reconstruction error  mu={err_mu:.6f}  sigma={err_sigma:.6f}")
    print(f"Adaptive threshold (alpha={ALPHA}): {threshold:.6f}")

    # Save stats for the detection script
    stats_path = os.path.join(MODEL_DIR, "train_stats.npz")
    np.savez(stats_path,
             err_mu=err_mu, err_sigma=err_sigma, threshold=threshold,
             data_mu=data_mu, data_sigma=data_sigma,
             window_size=WINDOW_SIZE, step=STEP, alpha=ALPHA)
    print(f"Stats saved -> {stats_path}")

    # --- Visualise loss ---
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, EPOCHS + 1), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("TCN-AE Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_fig_path = os.path.join(VIS_DIR, "training_loss.png")
    plt.savefig(loss_fig_path, dpi=150)
    print(f"Loss curve saved -> {loss_fig_path}")
    plt.show()

    # --- Visualise training error distribution ---
    plt.figure(figsize=(8, 4))
    plt.hist(all_errors, bins=60, density=True, alpha=0.7, label="Train MSE")
    plt.axvline(threshold, color="red", linestyle="--",
                label=f"Threshold ({ALPHA}σ) = {threshold:.4f}")
    plt.xlabel("Reconstruction MSE")
    plt.ylabel("Density")
    plt.title("Training Reconstruction Error Distribution")
    plt.legend()
    plt.tight_layout()
    err_fig_path = os.path.join(VIS_DIR, "train_error_dist.png")
    plt.savefig(err_fig_path, dpi=150)
    print(f"Error distribution saved -> {err_fig_path}")
    plt.show()

    print("Training complete.")


if __name__ == "__main__":
    main()
