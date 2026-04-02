# Power Usage Anomaly Detection - Machine Learning Case Study

## Overview
This case study demonstrates the application of unsupervised machine learning algorithms for detecting anomalies in IoT power consumption data. The pipeline uses multiple detection models to identify unusual power usage patterns that may indicate equipment faults, inefficiencies, or security concerns.

## Objective
Analyze historical power usage data from multiple energy meters to:
- Detect anomalous power consumption patterns
- Identify potential equipment issues or inefficiencies
- Generate actionable insights using AI-powered analysis
- Visualize power usage trends and anomalies

## Data Characteristics

### Dataset
- **Source**: IoT energy monitoring devices (multiple meters: a1, a2, a3, a4)
- **Telemetry**: Energy readings (a1E, a2E, a3E, a4E) in Wh
- **Time Period**: Configurable (default: Nov 2025 - Mar 2026)
- **Interval**: Hourly aggregated data
- **Total Records**: ~2,880 hours (4 months)

### Features Extracted
1. **Power Consumption**: Calculated from energy differences over time
2. **Temporal Features**: Hour of day, day of week, weekend indicator
3. **Statistical Features**: Rolling statistics, rate of change
4. **Multi-channel Data**: Total power across all meters

## Pipeline Architecture

### Step 0: Configuration (`step0_config.py`)
Centralized configuration for the entire pipeline:
- **Data Paths**: Input data directory and output directory
- **Date Ranges**: Training cutoff and test period
- **Model Settings**: Contamination rate, algorithm selection
- **LLM Integration**: Grok API configuration for AI summaries
- **Visualization**: Plot settings and DPI

### Step 1: Train and Detect (`step1_train_and_detect.py`)
Main anomaly detection pipeline:
1. **Data Loading**: Reads CSV files and parses JSON telemetry
2. **Feature Engineering**: 
   - Hourly resampling of energy data
   - Power calculation from energy differences
   - Temporal feature extraction
3. **Model Training**: Trains on historical baseline data
4. **Anomaly Detection**: Applies multiple algorithms:
   - **Isolation Forest**: Global anomaly detection
   - **ECOD**: Empirical Cumulative Distribution-based detection
   - **LOF**: Local Outlier Factor for local anomalies
5. **Ensemble Voting**: Combines predictions from multiple models
6. **Visualization**: Generates comprehensive charts
7. **AI Analysis**: Uses LLM to generate insights

### Step 2: Generate Prompts (`step2_generate_prompts.py`)
Creates structured prompts for LLM analysis of detected anomalies.

### Step 3: Call LLM (`step3_call_llm.py`)
Executes LLM API calls to generate detailed explanations for anomalies.

## Machine Learning Models

### 1. Isolation Forest
- **Type**: Tree-based ensemble method
- **Strength**: Excellent for global anomalies
- **Use Case**: Detects extreme outliers in power consumption

### 2. ECOD (Empirical Cumulative Distribution Outlier Detection)
- **Type**: Statistical method
- **Strength**: Fast, parameter-free
- **Use Case**: Identifies distribution-based anomalies

### 3. LOF (Local Outlier Factor)
- **Type**: Density-based method
- **Strength**: Detects local anomalies
- **Use Case**: Finds unusual patterns in local neighborhoods

## Output Files

### Data Files
1. **trained_models.pkl**: Serialized ML models for reuse
2. **baseline_statistics.json**: Statistical baseline from training data
3. **detected_anomalies.csv**: All detected anomalies with scores
4. **all_test_data.csv**: Complete test dataset with predictions
5. **analysis_summary.json**: Summary statistics and metadata

### Visualizations
6. **step1_anomaly_detection_results.png**: Multi-panel anomaly visualization
   - Time series with anomaly markers
   - Score distributions per model

7. **hourly_power_usage.png**: Comprehensive power usage analysis
   - Time series of hourly consumption
   - Average power by hour of day (bar chart)
   - Average power by day of week (bar chart)

### Reports
8. **anomaly_detection_report.html**: Interactive HTML report
   - Embedded anomaly visualization
   - AI-generated summary with insights
   - Professional styling

9. **hourly_power_report.html**: Interactive power usage report
   - Embedded power usage charts
   - AI-generated pattern analysis
   - Daily and weekly trend insights

## Key Features

### 1. Hourly Power Calculation
- Resamples irregular data to hourly intervals
- Calculates power from energy differences: `Power = ΔEnergy / ΔTime`
- Handles missing data with forward-fill interpolation

### 2. Multi-Model Ensemble
- Combines predictions from 3 different algorithms
- Voting mechanism: Anomaly if ≥2 models agree
- Reduces false positives through consensus

### 3. AI-Powered Insights
- Integrates with Grok API for natural language summaries
- Analyzes anomaly patterns and suggests causes
- Identifies daily and weekly usage patterns
- Provides energy efficiency recommendations

### 4. Interactive HTML Reports
- Self-contained reports with embedded visualizations
- Markdown-formatted AI summaries
- Professional styling for easy sharing
- Browser-based viewing (no special software needed)

## Usage

### 1. Configure Settings
Edit `step0_config.py` to set:
```python
# Date range for analysis
TEST_START_DATE = '2025-11-01'
TEST_END_DATE = '2026-03-01'

# Model sensitivity
CONTAMINATION = 0.02  # 2% expected anomaly rate

# Enable/disable models
USE_IFOREST = True
USE_ECOD = True
USE_LOF = True

# LLM API key (in .env file)
GROK_API_KEY = "your-api-key"
```

### 2. Run Analysis
```bash
python step1_train_and_detect.py
```

### 3. View Results
- Open `output/anomaly_detection_report.html` in a browser
- Open `output/hourly_power_report.html` for usage patterns
- Review CSV files for detailed data

## Insights Generated

### Anomaly Detection Insights
- Anomaly rate and severity assessment
- Peak anomaly power consumption
- Potential causes (equipment faults, overloads, etc.)
- Temporal patterns in anomalies

### Power Usage Insights
- Peak and low usage hours
- Weekday vs weekend consumption patterns
- Daily usage curves
- Energy efficiency opportunities
- Cost optimization recommendations

## Requirements

### Python Packages
- pandas: Data manipulation
- numpy: Numerical operations
- pyod: Anomaly detection algorithms
- matplotlib: Visualization
- requests: LLM API calls
- python-dotenv: Environment configuration

### API Access
- Grok API key for AI-generated summaries (optional but recommended)

## Best Practices

1. **Training Period**: Use at least 1-2 months of normal operation data
2. **Contamination Rate**: Start with 0.02 (2%) and adjust based on results
3. **Model Selection**: Enable all three models for best accuracy
4. **Data Quality**: Ensure consistent data collection intervals
5. **Review Anomalies**: Manually verify detected anomalies for false positives

## Troubleshooting

### Empty Test Dataset
- Check date ranges in `step0_config.py`
- Ensure `TEST_END_DATE` is after `TEST_START_DATE`
- Verify data files exist for the specified period

### High False Positive Rate
- Increase `CONTAMINATION` value (e.g., 0.05 for 5%)
- Review training data for quality issues
- Consider seasonal patterns in your data

### LLM Summary Unavailable
- Check `GROK_API_KEY` in `.env` file
- Verify API endpoint accessibility
- Review rate limits and quotas

## Future Enhancements

- Real-time anomaly detection
- Automated alerting system
- Root cause analysis automation
- Predictive maintenance integration
- Multi-site comparison analysis

## License

This case study is for educational and demonstration purposes.
