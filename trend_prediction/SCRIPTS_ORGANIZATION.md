# IoT Data Analysis Scripts Organization

## Overview
This folder contains **6 main prediction scripts** organized by:
- **3 Algorithms**: SARIMAX, Prophet, LightGBM
- **2 Prediction Types**: Weekly (7 days), Intraday (24 hours)

---

## Main Prediction Scripts (3 × 2 = 6)

### 1. SARIMAX Algorithm

#### `1_sarimax_weekly_prediction.py`
- **Purpose**: Predict power usage for the next 7 days (Apr 1-7, 2026)
- **Model**: SARIMAX(1,1,1)x(1,1,1,168) with exogenous variables
- **Features**: 
  - Weekly seasonality (168 hours)
  - Exogenous variables: is_weekday, hour, is_office_hour
  - Explicitly models weekday/weekend differences
- **Output**: 
  - `1_predictions_sarimax_weekly.csv` (168 hourly predictions)
  - `1_sarimax_weekly_results.png` (visualizations)
- **Run**: `python 1_sarimax_weekly_prediction.py`

#### `2_sarimax_intraday_prediction.py`
- **Purpose**: Predict power usage for the next 24 hours
- **Model**: SARIMAX(1,1,1)x(1,1,1,96) with exogenous variables
- **Features**:
  - Daily seasonality (96 intervals = 24h × 4 per hour)
  - 15-minute interval predictions
  - Exogenous variables: is_weekday, hour, minute, is_office_hour
- **Output**:
  - `2_predictions_sarimax_intraday.csv` (96 15-minute predictions)
  - `2_sarimax_intraday_results.png` (visualizations)
- **Run**: `python 2_sarimax_intraday_prediction.py`

---

### 2. Prophet Algorithm

#### `3_prophet_weekly_prediction.py`
- **Purpose**: Predict power usage for the next 7 days (Apr 1-7, 2026)
- **Model**: Facebook Prophet with HK holidays
- **Features**:
  - 30-minute interval predictions
  - Hong Kong public holidays integration
  - Automatic seasonality detection
- **Output**:
  - `3_predictions_prophet_weekly.csv` (336 30-minute predictions)
  - `3_prophet_weekly_results.png` (visualizations)
- **Run**: `python 3_prophet_weekly_prediction.py`

#### `4_prophet_intraday_prediction.py`
- **Purpose**: Predict power usage for the next 12 hours (intraday)
- **Model**: Facebook Prophet with HK holidays
- **Features**:
  - Hourly interval predictions
  - Hong Kong public holidays integration
  - Captures daily patterns
- **Output**:
  - `4_predictions_prophet_intraday.csv` (12 hourly predictions)
  - `4_prophet_intraday_results.png` (visualizations)
- **Run**: `python 4_prophet_intraday_prediction.py`

---

### 3. LightGBM Algorithm

#### `5_lightgbm_weekly_prediction.py`
- **Purpose**: Predict power usage for the next 7 days (Apr 1-7, 2026)
- **Model**: LightGBM Gradient Boosting
- **Features**:
  - Rich feature engineering (cyclical encoding, office hours, holidays)
  - 30-minute interval predictions
  - Lag features and rolling statistics
- **Output**:
  - `5_predictions_lightgbm_weekly.csv` (336 30-minute predictions)
  - `5_lightgbm_weekly_results.png` (visualizations)
- **Run**: `python 5_lightgbm_weekly_prediction.py`

#### `6_lightgbm_intraday_prediction.py`
- **Purpose**: Predict power usage for the next 24 hours
- **Model**: LightGBM Gradient Boosting
- **Features**:
  - Rich feature engineering (cyclical encoding, office hours, holidays)
  - 15-minute interval predictions
  - Lag features and rolling statistics
- **Output**:
  - `6_predictions_lightgbm_intraday.csv` (96 15-minute predictions)
  - `6_lightgbm_intraday_results.png` (visualizations)
- **Run**: `python 6_lightgbm_intraday_prediction.py`

---

## Quick Reference Table

| Algorithm | Weekly Script | Intraday Script |
|-----------|--------------|-----------------|
| **SARIMAX** | `1_sarimax_weekly_prediction.py` | `2_sarimax_intraday_prediction.py` |
| **Prophet** | `3_prophet_weekly_prediction.py` | `4_prophet_intraday_prediction.py` |
| **LightGBM** | `5_lightgbm_weekly_prediction.py` | `6_lightgbm_intraday_prediction.py` |

---

## Prediction Outputs

| Algorithm | Weekly Output | Intraday Output |
|-----------|--------------|-----------------|
| **SARIMAX** | `1_predictions_sarimax_weekly.csv` | `2_predictions_sarimax_intraday.csv` |
| **Prophet** | `3_predictions_prophet_weekly.csv` | `4_predictions_prophet_intraday.csv` |
| **LightGBM** | `5_predictions_lightgbm_weekly.csv` | `6_predictions_lightgbm_intraday.csv` |

---

## Supporting Files

### Data
- `power_usage_data.csv` - Historical power usage data (minute-level)
- `0_generate_data.py` - Script to generate synthetic power usage data (run this first)

### Analysis Scripts
- `plot_march_vs_prediction.py` - Compare March historical data with April predictions
- `analyze_weekday_weekend_issue.py` - Analyze weekday/weekend pattern differences

### Legacy/Experimental Scripts
- `trend_prediction_sarima.py` - Original SARIMA implementation
- `trend_prediction_sarimax_improved.py` - Improved SARIMAX with exogenous variables
- `trend_prediction_prophet.py` - Original Prophet trend prediction
- `trend_prediction_ml.py` - ML-based trend prediction
- `prophet_intraday_prediction_minute.py` - Minute-level Prophet prediction (experimental)

---

## Algorithm Comparison

### SARIMAX
- **Strengths**: Statistical rigor, captures seasonality explicitly, confidence intervals
- **Best for**: Time series with clear seasonal patterns, interpretable results
- **Limitations**: Computationally expensive, requires stationarity assumptions

### Prophet
- **Strengths**: Easy to use, handles holidays well, robust to missing data
- **Best for**: Quick predictions, business-friendly, automatic seasonality detection
- **Limitations**: Less control over model parameters, black-box approach

### LightGBM
- **Strengths**: High accuracy, handles complex patterns, feature importance
- **Best for**: Complex non-linear patterns, feature-rich datasets
- **Limitations**: Requires careful feature engineering, prone to overfitting

---

## Running All Scripts

To run all 6 main prediction scripts:

```bash
# SARIMAX
python 1_sarimax_weekly_prediction.py
python 2_sarimax_intraday_prediction.py

# Prophet
python 3_prophet_weekly_prediction.py
python 4_prophet_intraday_prediction.py

# LightGBM
python 5_lightgbm_weekly_prediction.py
python 6_lightgbm_intraday_prediction.py
```

---

## Dependencies

```
pandas
numpy
matplotlib
scikit-learn
statsmodels
prophet
lightgbm
```

Install with:
```bash
pip install pandas numpy matplotlib scikit-learn statsmodels prophet lightgbm
```

---

## Notes

1. All scripts use the same input data: `power_usage_data.csv`
2. Weekly predictions target Apr 1-7, 2026 (7 days)
3. Intraday predictions target the next 12-24 hours
4. Each script includes model evaluation on validation set
5. Visualizations are automatically saved as PNG files
