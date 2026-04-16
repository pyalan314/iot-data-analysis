"""
3_detect.py
===========
Run anomaly detection on test data using the trained TCN-AE model.

Reads
-----
data/test_mixed.csv
data/test_labels.csv
model/tcn_ae.pth
model/train_stats.npz

Outputs
-------
output/detection_result.png   – overlay of signal, errors, threshold, labels
output/detection_detail.png   – zoomed views of detected anomaly regions
data/detection_output.csv     – per-sample detection results
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(__file__))
from tcn_ae_model import TCNAE, sliding_window

# ---------------------------------------------------------------------------
# Config (model hyper-params must match training)
# ---------------------------------------------------------------------------
N_CHANNELS = 32
KERNEL_SIZE = 3
N_BLOCKS = 4
BOTTLENECK_DIM = 16

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
VIS_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(VIS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def windows_to_sample_errors(window_errors: np.ndarray, window_size: int,
                             step: int, n_samples: int) -> np.ndarray:
    """
    Map per-window reconstruction errors back to per-sample errors by
    averaging over all windows that cover each sample position.
    """
    error_sum = np.zeros(n_samples, dtype=np.float64)
    count = np.zeros(n_samples, dtype=np.float64)
    for i, e in enumerate(window_errors):
        start = i * step
        end = start + window_size
        if end > n_samples:
            break
        error_sum[start:end] += e
        count[start:end] += 1
    count[count == 0] = 1  # avoid division by zero for uncovered tails
    return (error_sum / count).astype(np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute precision, recall, F1 from binary arrays."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")

    # --- Load stats from training ---
    stats_path = os.path.join(MODEL_DIR, "train_stats.npz")
    if not os.path.exists(stats_path):
        print(f"ERROR: {stats_path} not found. Run 2_training.py first.")
        sys.exit(1)
    stats = np.load(stats_path)
    threshold = float(stats["threshold"])
    data_mu = float(stats["data_mu"])
    data_sigma = float(stats["data_sigma"])
    window_size = int(stats["window_size"])
    step = int(stats["step"])
    alpha = float(stats["alpha"])
    print(f"Loaded threshold={threshold:.6f}  (alpha={alpha})")

    # --- Load test data ---
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_mixed.csv"))
    label_df = pd.read_csv(os.path.join(DATA_DIR, "test_labels.csv"))
    raw_signal = test_df["acceleration_g"].values.astype(np.float32)
    true_labels = label_df["label"].values.astype(np.int32)
    time_s = test_df["time_s"].values

    # Z-score normalise using training statistics
    normed = (raw_signal - data_mu) / (data_sigma + 1e-8)

    # --- Sliding window ---
    windows = sliding_window(normed, window_size, step)
    print(f"Test windows: {windows.shape[0]}")

    # --- Load model ---
    model = TCNAE(
        window_size=window_size,
        n_channels=N_CHANNELS,
        kernel_size=KERNEL_SIZE,
        n_blocks=N_BLOCKS,
        bottleneck_dim=BOTTLENECK_DIM,
    ).to(DEVICE)
    model_path = os.path.join(MODEL_DIR, "tcn_ae.pth")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    print("Model loaded.")

    # --- Compute per-window reconstruction errors ---
    X = torch.tensor(windows, dtype=torch.float32).unsqueeze(1)  # (N,1,L)
    batch_size = 256
    window_errors = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size].to(DEVICE)
            x_hat = model(xb)
            mse = ((xb - x_hat) ** 2).mean(dim=(1, 2))
            window_errors.append(mse.cpu().numpy())
    window_errors = np.concatenate(window_errors)

    # --- Map to per-sample errors ---
    sample_errors = windows_to_sample_errors(
        window_errors, window_size, step, len(raw_signal))

    # --- Anomaly detection with confidence levels ---
    predicted = (sample_errors > threshold).astype(np.int32)
    
    # Compute confidence score: normalized error relative to threshold
    # confidence = 0 when error = 0, confidence = 1 when error = threshold, >1 when anomalous
    confidence_scores = sample_errors / (threshold + 1e-8)

    # --- Metrics ---
    precision, recall, f1 = compute_metrics(true_labels, predicted)
    print(f"\n{'='*40}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"{'='*40}\n")

    # --- Visualise metrics ---
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics = ['Precision', 'Recall', 'F1 Score']
    values = [precision, recall, f1]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Anomaly Detection Performance Metrics', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    metrics_path = os.path.join(VIS_DIR, "metrics.png")
    plt.savefig(metrics_path, dpi=150)
    print(f"Metrics visualisation saved -> {metrics_path}")
    plt.show()

    # --- Save detection output ---
    out_df = pd.DataFrame({
        "time_s": time_s,
        "acceleration_g": raw_signal,
        "reconstruction_error": sample_errors,
        "threshold": threshold,
        "confidence_score": confidence_scores,
        "predicted_anomaly": predicted,
        "true_label": true_labels,
    })
    out_csv = os.path.join(DATA_DIR, "detection_output.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Detection output saved -> {out_csv}")

    # --- Visualisation: full overview ---
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    # Signal
    axes[0].plot(time_s, raw_signal, linewidth=0.3, color="steelblue")
    axes[0].set_ylabel("Acceleration (g)")
    axes[0].set_title("Test Signal")

    # Reconstruction error + threshold
    axes[1].plot(time_s, sample_errors, linewidth=0.4, color="darkorange",
                 label="Recon. Error")
    axes[1].axhline(threshold, color="red", linestyle="--", linewidth=1.0,
                    label=f"Threshold (μ+{alpha}σ)={threshold:.4f}")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Per-Sample Reconstruction Error")
    axes[1].legend(loc="upper right")

    # Ground truth vs prediction
    axes[2].fill_between(time_s, true_labels, alpha=0.35, color="red",
                         label="Ground Truth")
    axes[2].fill_between(time_s, predicted * 0.8, alpha=0.35, color="blue",
                         label="Predicted")
    axes[2].set_ylabel("Anomaly")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Anomaly Detection Result")
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    result_path = os.path.join(VIS_DIR, "detection_result.png")
    plt.savefig(result_path, dpi=150)
    print(f"Result visualisation saved -> {result_path}")
    plt.show()

    # --- Visualisation: Confidence levels ---
    fig, ax = plt.subplots(figsize=(16, 5))
    
    # Color-coded confidence levels
    colors = np.zeros((len(confidence_scores), 4))
    for i, conf in enumerate(confidence_scores):
        if conf < 0.5:
            colors[i] = [0.2, 0.8, 0.2, 0.6]  # Green - Normal
        elif conf < 1.0:
            colors[i] = [1.0, 0.8, 0.0, 0.6]  # Yellow - Elevated
        elif conf < 2.0:
            colors[i] = [1.0, 0.5, 0.0, 0.7]  # Orange - Warning
        else:
            colors[i] = [1.0, 0.0, 0.0, 0.8]  # Red - Critical
    
    ax.scatter(time_s, confidence_scores, c=colors, s=1, marker='.')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='Threshold (Confidence=1.0)')
    ax.axhline(0.5, color='yellow', linestyle=':', linewidth=1.0, alpha=0.7, label='Elevated (0.5)')
    ax.axhline(2.0, color='darkred', linestyle=':', linewidth=1.0, alpha=0.7, label='Critical (2.0)')
    
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Confidence Score (Error / Threshold)", fontsize=12)
    ax.set_title("Anomaly Confidence Levels Over Time", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=[0.2, 0.8, 0.2, 0.6], label='Normal (< 0.5)'),
        Patch(facecolor=[1.0, 0.8, 0.0, 0.6], label='Elevated (0.5-1.0)'),
        Patch(facecolor=[1.0, 0.5, 0.0, 0.7], label='Warning (1.0-2.0)'),
        Patch(facecolor=[1.0, 0.0, 0.0, 0.8], label='Critical (> 2.0)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    confidence_path = os.path.join(VIS_DIR, "confidence_levels.png")
    plt.savefig(confidence_path, dpi=150)
    print(f"Confidence levels saved -> {confidence_path}")
    plt.show()

    # --- Visualisation: zoom into ALL ground-truth anomaly regions ---
    gt_starts = np.where(np.diff(true_labels.astype(int)) == 1)[0]
    n_zoom = len(gt_starts)
    if n_zoom > 0:
        fig, axes = plt.subplots(n_zoom, 1, figsize=(14, 3.5 * n_zoom))
        if n_zoom == 1:
            axes = [axes]
        for idx, start in enumerate(gt_starts):
            # find the end of this ground-truth region
            gt_end_candidates = np.where(
                (np.diff(true_labels.astype(int)) == -1)
                & (np.arange(len(true_labels) - 1) > start)
            )[0]
            gt_end = gt_end_candidates[0] if len(gt_end_candidates) > 0 else len(true_labels) - 1
            margin = window_size * 2
            lo = max(0, start - margin)
            hi = min(len(raw_signal), gt_end + margin)
            ax = axes[idx]
            ax.plot(time_s[lo:hi], raw_signal[lo:hi], linewidth=0.5,
                    color="steelblue", label="Signal")
            ax2 = ax.twinx()
            ax2.plot(time_s[lo:hi], sample_errors[lo:hi], linewidth=0.8,
                     color="darkorange", label="Error")
            ax2.axhline(threshold, color="red", linestyle="--", linewidth=0.8)
            # Shade ground truth
            ax.fill_between(time_s[lo:hi],
                            raw_signal[lo:hi].min(),
                            raw_signal[lo:hi].max(),
                            where=true_labels[lo:hi] == 1,
                            alpha=0.15, color="red", label="True Anomaly")
            # Shade predicted
            ax.fill_between(time_s[lo:hi],
                            raw_signal[lo:hi].min(),
                            raw_signal[lo:hi].max(),
                            where=predicted[lo:hi] == 1,
                            alpha=0.10, color="blue", label="Predicted")
            ax.set_title(f"GT Region #{idx+1} (t={time_s[start]:.2f}s – {time_s[gt_end]:.2f}s)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Acceleration (g)")
            ax2.set_ylabel("Recon. Error")
            ax.legend(loc="upper left", fontsize=8)
            ax2.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        detail_path = os.path.join(VIS_DIR, "detection_detail.png")
        plt.savefig(detail_path, dpi=150)
        print(f"Detail visualisation saved -> {detail_path}")
        plt.show()

    print("Detection complete.")


if __name__ == "__main__":
    main()
