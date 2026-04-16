# IoT Anomaly Detection Using TimesFM

Anomaly detection for IoT sensor data using Google's TimesFM foundation model with forecast-based residual analysis.

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure HuggingFace Token

Create a `.env` file:
```bash
HF_TOKEN=your_huggingface_token_here
```

Get your token from: https://huggingface.co/settings/tokens

### 3. Run Scripts

```bash
# Step 1: Generate synthetic data
python 1_generate_data.py

# Step 2: Validate model
python 2_training.py

# Step 3: Forecast and detect anomalies
python 3_forecast_detect.py
```

## Output

- **Data**: `data/server_room_temperature.csv`
- **Visualizations**: `output/*.png`
- **Results**: `output/forecast_results.csv`
- **Config**: `output/config.pkl`

## Features

- **Three Severity Levels**: Detects MILD (+3°C), MODERATE (+5°C), and SEVERE (+8°C) anomalies
- **Two Forecasting Methods**: Fixed window and moving window approaches
- **Zero-Shot Learning**: No training required using pre-trained TimesFM model
- **Quantile Forecasting**: 80% prediction intervals for robust anomaly detection

## System Requirements

- **RAM**: 8 GB minimum, 16 GB recommended
- **Virtual Memory**: 16 GB recommended (see troubleshooting)
- **Disk Space**: 2 GB free

## Troubleshooting

If you encounter **"paging file is too small"** error:

1. Increase Windows virtual memory to 16 GB
2. See `FIX_MEMORY_ISSUE.md` for detailed instructions
3. Restart computer after changing settings

For other issues, see `TROUBLESHOOTING.md`

## Documentation

- **REPORT.md**: Full technical report with methodology and results
- **FIX_MEMORY_ISSUE.md**: Memory configuration guide
- **TROUBLESHOOTING.md**: Common issues and solutions

## License

MIT
