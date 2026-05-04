# Energy Management POC

AI-powered energy optimization solution for a 3-storey office building.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up API key
cp example.env .env
# Edit .env and add your OPENAI_API_KEY or XAI_API_KEY

# Run the pipeline
python 1_generate_data.py
python 2_forecast_anomaly_detection.py
python 4_generate_narrative.py
```

## What It Does

1. **Generate Data** - Creates 90 days of realistic energy consumption data with anomalies
2. **Detect & Forecast** - Identifies anomalies using Isolation Forest and forecasts with TimesFM/DLinear
3. **Generate Narrative** - Uses LLM (Grok) to create actionable recommendations

## Output

- `data/` - Generated datasets and analysis results
- `model/` - Trained models
- `output/` - Visualizations and reports

## Configuration

Edit `config.py` to customize:
- Data generation parameters
- Anomaly detection thresholds
- Forecast horizons
- LLM settings

## Requirements

- Python 3.11+
- See `requirements.txt` for dependencies
- API key for Grok or OpenAI-compatible LLM

## Documentation

- `proposal.md` - Original business proposal
- `TECHNICAL_PROPOSAL.md` - Technical implementation details
- `context.json` - Building usage context for LLM
