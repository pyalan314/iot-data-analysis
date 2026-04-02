# Power Usage Trend Prediction - Machine Learning Case Study

## Overview
This case study demonstrates the application of machine learning (LSTM neural networks) for predicting power usage trends in IoT devices.

## Objective
Learn from historical power usage data and predict future trends for a single device (DEVICE_001) with one telemetry metric: `power_usage`.

## Data Characteristics

### Dataset
- **Device**: DEVICE_001
- **Telemetry**: power_usage (kW)
- **Time Period**: January 1, 2026 - March 31, 2026 (3 months)
- **Interval**: 1 minute
- **Total Records**: ~129,600 data points

### Patterns Embedded in Data
1. **Time-of-Day Pattern**: Higher power usage during office hours (9 AM - 6 PM)
2. **Weekday Pattern**: Higher usage on weekdays, lower on weekends
3. **Holiday Pattern**: Significantly lower usage on Hong Kong public holidays
   - New Year's Day (Jan 1)
   - Lunar New Year (Jan 29-31, Feb 2)

## Machine Learning Approach

### Model: SARIMA (Seasonal AutoRegressive Integrated Moving Average)
- **Model Order**: SARIMA(1,1,1)x(1,1,1,168)
  - AR order (p): 1
  - Differencing (d): 1
  - MA order (q): 1
  - Seasonal AR (P): 1
  - Seasonal differencing (D): 1
  - Seasonal MA (Q): 1
  - Seasonal period (s): 168 hours (7 days)
- **Data Frequency**: Hourly (resampled from minute data)
- **Features**: Univariate time series (power_usage only)

### Training Strategy
- **Train/Validation Split**: 80/20
- **Seasonal Decomposition**: Additive model with 168-hour (weekly) period
- **Optimization**: Maximum Likelihood Estimation
- **Confidence Intervals**: 95% forecast intervals
- **Captures**: Daily cycles + weekly patterns (weekday vs weekend)

## Prediction Task
- **Historical Data**: Jan 1 - Mar 31, 2026
- **Prediction Period**: Apr 1 - Apr 7, 2026 (7 days)
- **Output**: Minute-level power usage predictions

## Files

### 1. `generate_data.py`
Generates synthetic power usage data with realistic patterns:
- Time-based variations
- Weekday/weekend differences
- Holiday effects
- Random noise and gradual trend

**Usage**:
```bash
python generate_data.py
```

**Output**: `power_usage_data.csv`

### 2. `trend_prediction_sarima.py`
Main prediction script that:
- Loads historical data
- Resamples to hourly frequency
- Performs seasonal decomposition
- Builds and trains SARIMA model
- Evaluates model performance
- Generates predictions for Apr 1-7 with confidence intervals
- Creates comprehensive visualizations

**Usage**:
```bash
python trend_prediction_sarima.py
```

**Outputs**:
- `predictions_sarima.csv` - Predicted power usage for Apr 1-7
- `prediction_results_sarima.png` - Main visualization dashboard
- `daily_patterns_sarima.png` - Daily pattern analysis

## Results

### Visualizations
The script generates multiple plots:

**Main Dashboard** (`prediction_results_sarima.png`):
1. **Original Time Series**: Training data
2. **Trend Component**: Long-term trend extracted
3. **Seasonal Component**: 168-hour weekly pattern (captures weekday/weekend differences)
4. **Residual Component**: Random variations
5. **Validation Performance**: Actual vs predicted on validation set
6. **Historical Data**: Last 7 days of actual data
7. **Future Predictions**: 7-day forecast with 95% confidence intervals
8. **Combined View**: Transition from historical to predicted

**Daily Patterns** (`daily_patterns_sarima.png`):
1. **Hourly Pattern by Day**: Shows predicted usage pattern for each day of the week
2. **Daily Averages**: Average predicted usage per day

### Evaluation Metrics
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

## Requirements
```
pandas
numpy
matplotlib
scikit-learn
statsmodels
```

## Key Insights
- SARIMA effectively models time series with seasonal patterns
- The model captures:
  - **Trend**: Gradual increase in power usage over time
  - **Seasonality**: 24-hour daily cycles (office hours vs night)
  - **Weekly patterns**: Weekday vs weekend differences
  - **Holiday effects**: Lower usage on public holidays
- Provides confidence intervals for predictions (uncertainty quantification)
- Seasonal decomposition reveals underlying patterns clearly

## Future Enhancements
- Add weather data as exogenous variables (SARIMAX)
- Implement multivariate prediction (multiple devices)
- Compare with other models (Prophet, LSTM, ETS)
- Optimize SARIMA parameters using grid search
- Real-time prediction updates with rolling forecasts

## Notes
- All files use UTF-8 encoding
- Random seeds set for reproducibility
- Model training may take several minutes depending on hardware
