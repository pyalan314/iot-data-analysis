# IoT Anomaly Detection Using Machine Learning

Anomaly detection system for IoT sensor devices using Isolation Forest algorithm.

## Overview

This project demonstrates machine learning-based anomaly detection for IoT temperature sensors, specifically designed for server room monitoring scenarios.

## Features

- **Isolation Forest** algorithm for unsupervised anomaly detection
- **Three-level severity classification**: Mild, Moderate, Severe
- **Temporal feature engineering**: Lag features, rolling statistics, change detection
- **Automated visualization** of detection results

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Data

```bash
python 1_generate_data.py
```

### 3. Train Model

```bash
python 2_training.py
```

### 4. Detect Anomalies

```bash
python 3_detect.py
```

## Project Structure

```
├── 1_generate_data.py    # Generate synthetic sensor data
├── 2_training.py          # Train Isolation Forest model
├── 3_detect.py            # Detect anomalies with severity levels
├── requirements.txt       # Python dependencies
├── REPORT.MD              # Detailed project report
├── data/                  # Generated data and results
├── model/                 # Trained models
└── output/                # Visualization outputs
```

## Output

- **CSV files**: Sensor data and detection results in `/data`
- **Trained models**: Isolation Forest model and scaler in `/model`
- **Visualizations**: Charts and graphs in `/output`

## Use Cases

- Server room temperature monitoring
- Industrial equipment monitoring
- Smart building management
- Environmental monitoring
- Healthcare IoT devices

## Documentation

See [REPORT.MD](REPORT.MD) for detailed documentation including methodology, results evaluation, and IoT applications.
