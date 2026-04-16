# Vibration Anomaly Detection for Highway Infrastructure

AI-powered vibration monitoring system using TCN-AE (Temporal Convolutional Network Autoencoder) for autonomous structural health assessment.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline
python 1_generate_data.py
python 2_training.py
python 3_detect.py
```

## What It Does

- **Detects anomalies** in vibration signals (spikes, steps, drift)
- **Confidence-based alerts** (not just binary yes/no)
- **Weather-aware decisions** for maintenance prioritization
- **Edge-deployable** (no cloud required)

## Output

- `data/` — CSV datasets
- `model/` — Trained TCN-AE weights
- `output/` — Visualizations (metrics, confidence levels, detection results)

## Key Features

✅ Unsupervised learning (normal data only)  
✅ Real-time capable (~10ms per window)  
✅ Adaptive thresholding (μ + 3.2σ)  
✅ Multi-factor decision matrix (vibration + weather)

## Documentation

See [REPORT.md](REPORT.md) for full methodology, results, and decision framework.

## Architecture

```
Input (128 samples) → TCN Encoder → Bottleneck (16D) → TCN Decoder → Reconstruction
                                                              ↓
                                                    Reconstruction Error
                                                              ↓
                                                    Confidence Score = Error / Threshold
```

## Applications

- Highway bridges
- Building structures  
- Railway tracks
- Industrial machinery
- Offshore platforms

---

**Based on:** Meng et al., "Anomaly Detection for Construction Vibration Signals using Unsupervised Deep Learning and Cloud Computing" (PolyU, 2024)
