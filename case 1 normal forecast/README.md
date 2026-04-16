# IoT Time Series Forecasting Demo

Comparative study of **DLinear** and **Google TimesFM** for office tower power consumption forecasting.

## Quick Start

### 1. Setup Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Hugging Face Token (Optional)

TimesFM requires a Hugging Face token for model download:

```powershell
# Copy example file
copy .env.example .env

# Edit .env and add your token:
# HF_TOKEN=your_huggingface_token_here
```

Get your token from: https://huggingface.co/settings/tokens

### 3. Run Scripts

```powershell
# Generate synthetic data (1 year, 15-min intervals)
python 1_generate_data.py

# Train DLinear model
python 2_training.py

# Generate forecasts with both models
python 3_forecast.py
```

## Output

- **Data**: `data/office_power_data.csv`
- **Model**: `model/` (DLinear checkpoint)
- **Visualizations**: `output/*.png`
- **Forecast Results**: `output/forecast_results.csv`
- **Report**: `REPORT.md` / `REPORT.pdf`

## Models

- **DLinear**: Fast, interpretable, requires training
- **TimesFM**: Zero-shot, pre-trained foundation model

## Requirements

- Python 3.11+
- PyTorch
- NeuralForecast
- TimesFM (from GitHub)

See `REPORT.md` for detailed analysis and methodology.
