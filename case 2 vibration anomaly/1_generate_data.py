"""
1_generate_data.py
==================
Generate synthetic construction-site vibration data (accelerometer, unit: g).

Outputs
-------
data/train_normal.csv   – normal vibration for training (no anomalies)
data/test_mixed.csv     – mixed signal containing both normal segments and
                          injected anomalies (spike, step, drift)
data/test_labels.csv    – ground-truth binary labels (0=normal, 1=anomaly)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
FS = 256                # sampling frequency (Hz)
TRAIN_DURATION = 120    # seconds of normal data for training
TEST_DURATION = 60      # seconds of test data

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
VIS_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Signal generation helpers
# ---------------------------------------------------------------------------

def generate_normal_vibration(n_samples: int, fs: int) -> np.ndarray:
    """
    Simulate normal construction-site vibration:
      - background noise
      - periodic machine hum (low freq)
      - intermittent work bursts
    """
    t = np.arange(n_samples) / fs

    # Background Gaussian noise (low amplitude)
    noise = 0.02 * np.random.randn(n_samples)

    # Machine hum – combination of low-freq sinusoids
    hum = 0.05 * np.sin(2 * np.pi * 12 * t) + 0.03 * np.sin(2 * np.pi * 25 * t)

    # Intermittent work bursts (amplitude-modulated noise segments)
    burst_env = np.zeros(n_samples)
    n_bursts = n_samples // (fs * 5)  # roughly one burst every 5s
    for _ in range(n_bursts):
        start = np.random.randint(0, max(n_samples - fs * 2, 1))
        length = np.random.randint(fs // 2, fs * 2)
        end = min(start + length, n_samples)
        burst_env[start:end] = np.random.uniform(0.05, 0.15)
    bursts = burst_env * np.random.randn(n_samples)

    signal = noise + hum + bursts
    return signal.astype(np.float32)


def inject_anomalies(signal: np.ndarray, fs: int):
    """
    Inject three anomaly types into the signal and return labels.

    Anomaly types (from the paper):
      - Spike:  sudden large-amplitude impulse
      - Step:   abrupt baseline shift
      - Drift:  gradual linear drift over time
    """
    n = len(signal)
    labels = np.zeros(n, dtype=np.int32)
    anomalous = signal.copy()

    # ---- Spike anomalies ----
    n_spikes = 5
    for _ in range(n_spikes):
        pos = np.random.randint(fs, n - fs)
        width = np.random.randint(3, 15)
        amp = np.random.choice([-1, 1]) * np.random.uniform(0.8, 1.5)
        end = min(pos + width, n)
        anomalous[pos:end] += amp
        labels[pos:end] = 1

    # ---- Step anomaly ----
    step_start = np.random.randint(n // 4, n // 3)
    step_len = int(1.5 * fs)
    step_end = min(step_start + step_len, n)
    anomalous[step_start:step_end] += 0.5
    labels[step_start:step_end] = 1

    # ---- Drift anomaly ----
    drift_start = np.random.randint(n // 2, 2 * n // 3)
    drift_len = int(3.0 * fs)
    drift_end = min(drift_start + drift_len, n)
    drift_ramp = np.linspace(0, 0.6, drift_end - drift_start)
    anomalous[drift_start:drift_end] += drift_ramp
    labels[drift_start:drift_end] = 1

    return anomalous, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Generating synthetic vibration data …")

    # --- Training data (normal only) ---
    n_train = TRAIN_DURATION * FS
    train_signal = generate_normal_vibration(n_train, FS)

    train_df = pd.DataFrame({
        "time_s": np.arange(n_train) / FS,
        "acceleration_g": train_signal,
    })
    train_path = os.path.join(OUTPUT_DIR, "train_normal.csv")
    train_df.to_csv(train_path, index=False)
    print(f"  Saved training data  -> {train_path}  ({n_train} samples, {TRAIN_DURATION}s)")

    # --- Test data (normal + injected anomalies) ---
    n_test = TEST_DURATION * FS
    test_normal = generate_normal_vibration(n_test, FS)
    test_signal, test_labels = inject_anomalies(test_normal, FS)

    test_df = pd.DataFrame({
        "time_s": np.arange(n_test) / FS,
        "acceleration_g": test_signal,
    })
    test_path = os.path.join(OUTPUT_DIR, "test_mixed.csv")
    test_df.to_csv(test_path, index=False)
    print(f"  Saved test data      -> {test_path}  ({n_test} samples, {TEST_DURATION}s)")

    label_df = pd.DataFrame({
        "time_s": np.arange(n_test) / FS,
        "label": test_labels,
    })
    label_path = os.path.join(OUTPUT_DIR, "test_labels.csv")
    label_df.to_csv(label_path, index=False)
    print(f"  Saved test labels    -> {label_path}")

    # --- Visualise ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=False)

    axes[0].plot(train_df["time_s"], train_df["acceleration_g"], linewidth=0.3)
    axes[0].set_title("Training Signal (Normal Only)")
    axes[0].set_ylabel("Acceleration (g)")

    axes[1].plot(test_df["time_s"], test_df["acceleration_g"], linewidth=0.3)
    axes[1].set_title("Test Signal (With Injected Anomalies)")
    axes[1].set_ylabel("Acceleration (g)")

    axes[2].fill_between(label_df["time_s"], label_df["label"],
                         color="red", alpha=0.4, label="Anomaly Region")
    axes[2].set_title("Ground-Truth Labels")
    axes[2].set_ylabel("Label")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()

    plt.tight_layout()
    vis_path = os.path.join(VIS_DIR, "data_overview.png")
    plt.savefig(vis_path, dpi=150)
    print(f"  Saved visualisation  -> {vis_path}")
    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
