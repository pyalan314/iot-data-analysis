"""
Step 1: Train anomaly detection models, detect anomalies, visualize results, and save artifacts
This script:
- Loads and preprocesses IoT power meter data
- Trains Isolation Forest and ECOD models on historical data
- Detects anomalies in test data (February 2026)
- Visualizes anomalies
- Saves trained models, baseline statistics, and detected anomalies for next steps
"""

import pandas as pd
import numpy as np
import json
import pickle
from pyod.models.iforest import IForest
from pyod.models.ecod import ECOD
from pyod.models.lof import LOF
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
import shutil
import requests
import time
import base64
from io import BytesIO

# Import configuration
import step0_config as config

warnings.filterwarnings('ignore')

# Helper function to generate LLM summary
def generate_llm_summary(prompt, max_retries=3):
    """Call LLM API to generate a text summary"""
    if not config.GROK_API_KEY:
        return "LLM summary unavailable (API key not configured)"
    
    headers = {
        "Authorization": f"Bearer {config.GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": config.GROK_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a data analyst providing concise, insightful summaries of power consumption data. Focus on key patterns, anomalies, and actionable insights."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": config.LLM_MAX_TOKENS
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(config.LLM_RATE_LIMIT_DELAY)
                continue
            return f"LLM summary unavailable (Error: {str(e)})"

# Clean and recreate output directory for fresh run
if config.OUTPUT_DIR.exists():
    print(f"Cleaning output directory: {config.OUTPUT_DIR}")
    shutil.rmtree(config.OUTPUT_DIR)
config.OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Output directory ready: {config.OUTPUT_DIR}\n")

print("="*80)
print("STEP 1: TRAIN ANOMALY DETECTION MODELS")
print("="*80)

# Print configuration
config.print_config()
if not config.validate_config():
    print("\nERROR: Invalid configuration. Please check step0_config.py")
    exit(1)
print()

# 1. LOAD AND PARSE DATA
print("\n[1/7] Loading data...")
csv_files = sorted([f for f in config.DATA_DIR.glob('202*.csv')])
print(f"Found {len(csv_files)} data files")

all_data = []
for csv_file in csv_files:
    print(f"  Loading {csv_file.name}...")
    df_month = pd.read_csv(csv_file)
    all_data.append(df_month)

df = pd.concat(all_data, ignore_index=True)
print(f"Total records loaded: {len(df):,}")

# 2. PARSE JSON FIELDS
print("\n[2/7] Parsing JSON fields...")
def parse_output_simple(json_str):
    try:
        data = json.loads(json_str)
        return pd.Series(data)
    except (json.JSONDecodeError, TypeError):
        return pd.Series()

output_data = df['outputSimple'].apply(parse_output_simple)
df = pd.concat([df, output_data], axis=1)

# Parse timestamp and convert to local timezone
df['timestamp_utc'] = pd.to_datetime(df['createdAt'], format='mixed', utc=True)
df['timestamp'] = df['timestamp_utc'].dt.tz_convert(config.TIMEZONE)
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"Timestamps converted from UTC to {config.TIMEZONE}")
print(f"Example: {df['timestamp_utc'].iloc[0]} (UTC) -> {df['timestamp'].iloc[0]} (Local)")

# 3. EXTRACT ENERGY METRICS AND RESAMPLE TO HOURLY INTERVALS
print("\n[3/7] Extracting features and resampling to hourly intervals...")
energy_cols = [col for col in df.columns if 'E' in col and col.startswith('a') and len(col) == 3]

# Convert energy columns to numeric
for col in energy_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Set timestamp as index for resampling
df_indexed = df.set_index('timestamp')

# Resample to hourly intervals, taking the last reading in each hour
# This gives us the cumulative energy at the end of each hour
df_hourly = df_indexed[energy_cols].resample('1H').last()

# Forward fill missing hours (in case some hours have no data)
df_hourly = df_hourly.fillna(method='ffill')

# Calculate hourly energy consumption (difference between consecutive hours)
power_cols = []
for col in energy_cols:
    power_col = col.replace('E', 'P_calc')  # e.g., a1E -> a1P_calc
    # Energy consumed in this hour = current hour reading - previous hour reading
    energy_diff = df_hourly[col].diff().fillna(0)
    # Handle negative differences (meter resets or errors) by setting to 0
    df_hourly[power_col] = energy_diff.where(energy_diff >= 0, 0)
    power_cols.append(power_col)

df_hourly['total_power'] = df_hourly[power_cols].sum(axis=1)

# Reset index to make timestamp a column again
df_hourly = df_hourly.reset_index()

# Use the hourly resampled data as our main dataframe
df = df_hourly.copy()

# Recreate timestamp_utc for filtering (convert back to UTC)
df['timestamp_utc'] = df['timestamp'].dt.tz_convert('UTC')

# Extract time features with cyclical encoding (using LOCAL timezone)
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute
df['hour_fraction'] = df['hour'] + df['minute'] / 60.0  # e.g., 23:30 -> 23.5

# Cyclical encoding for hour (handles midnight discontinuity)
df['hour_sin'] = np.sin(2 * np.pi * df['hour_fraction'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour_fraction'] / 24)

# Day of week features (0=Monday, 6=Sunday in local timezone)
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Binary flags for interpretability (based on LOCAL time)
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)  # Saturday=5, Sunday=6
df['is_night'] = df['hour'].isin(range(config.NIGHT_HOURS_START, config.NIGHT_HOURS_END)).astype(int)
df['is_business_hours'] = df['hour'].isin(range(config.BUSINESS_HOURS_START, config.BUSINESS_HOURS_END)).astype(int)

print(f"Data resampled to hourly intervals: {len(df):,} hours")
print(f"Energy channels: {energy_cols}")
print(f"Power channels (hourly energy consumption): {power_cols}")
print(f"Time features extracted using {config.TIMEZONE} timezone:")
print("  - Cyclical: hour_sin, hour_cos, day_sin, day_cos")
print(f"  - Binary: is_weekend, is_night ({config.NIGHT_HOURS_START:02d}:00-{config.NIGHT_HOURS_END:02d}:00), is_business_hours ({config.BUSINESS_HOURS_START:02d}:00-{config.BUSINESS_HOURS_END:02d}:00)")
print(f"Total features: total_power + {len(power_cols)} power channels + 7 time features")

# 4. TRAIN/TEST SPLIT
print("\n[4/7] Splitting train/test data...")
df_clean = df[df['total_power'] >= 0].copy()
# Use timestamp_utc for filtering (cutoffs are in UTC)
TRAIN_CUTOFF = pd.Timestamp(config.TRAIN_CUTOFF_DATE, tz='UTC')
TEST_START = pd.Timestamp(config.TEST_START_DATE, tz='UTC')
TEST_END = pd.Timestamp(config.TEST_END_DATE, tz='UTC')

df_train = df_clean[df_clean['timestamp_utc'] < TRAIN_CUTOFF].copy()
df_test = df_clean[(df_clean['timestamp_utc'] >= TEST_START) & (df_clean['timestamp_utc'] < TEST_END)].copy()

print(f"Training data: {len(df_train):,} records (before {config.TRAIN_CUTOFF_DATE})")
print(f"Test data: {len(df_test):,} records ({config.TEST_START_DATE} to {config.TEST_END_DATE})")

# Calculate baseline statistics from training data
train_stats = {
    'total_power_mean': df_train['total_power'].mean(),
    'total_power_std': df_train['total_power'].std(),
    'total_power_q25': df_train['total_power'].quantile(0.25),
    'total_power_q75': df_train['total_power'].quantile(0.75),
    'hourly_mean': df_train.groupby('hour')['total_power'].mean().to_dict(),
    'hourly_std': df_train.groupby('hour')['total_power'].std().to_dict(),
    'channel_means': {col: df_train[col].mean() for col in power_cols if col in df_train.columns},
    'channel_stds': {col: df_train[col].std() for col in power_cols if col in df_train.columns}
}

print(f"\nBaseline statistics calculated:")
print(f"  Mean power: {train_stats['total_power_mean']:.2f}W")
print(f"  Std dev: {train_stats['total_power_std']:.2f}W")
print(f"  Q25-Q75: {train_stats['total_power_q25']:.2f}W - {train_stats['total_power_q75']:.2f}W")

# 5. TRAIN MODELS
print("\n[5/7] Training anomaly detection models...")
feature_cols = ['total_power'] + [col for col in power_cols if col in df.columns and df[col].sum() > 0]
feature_cols.extend(['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is_weekend', 'is_night', 'is_business_hours'])

print(f"Feature columns: {feature_cols}")

X_train = df_train[feature_cols].values
X_test = df_test[feature_cols].values

# Train enabled models
models = {}

if config.USE_IFOREST:
    print(f"  Training Isolation Forest (contamination={config.CONTAMINATION}, n_estimators={config.IFOREST_N_ESTIMATORS})...")
    clf_iforest = IForest(contamination=config.CONTAMINATION, random_state=config.IFOREST_RANDOM_STATE, n_estimators=config.IFOREST_N_ESTIMATORS)
    clf_iforest.fit(X_train)
    models['iforest'] = clf_iforest

if config.USE_ECOD:
    print(f"  Training ECOD (contamination={config.CONTAMINATION})...")
    clf_ecod = ECOD(contamination=config.CONTAMINATION)
    clf_ecod.fit(X_train)
    models['ecod'] = clf_ecod

if config.USE_LOF:
    print(f"  Training LOF (contamination={config.CONTAMINATION}, n_neighbors={config.LOF_N_NEIGHBORS})...")
    clf_lof = LOF(contamination=config.CONTAMINATION, n_neighbors=config.LOF_N_NEIGHBORS)
    clf_lof.fit(X_train)
    models['lof'] = clf_lof

print(f"  {len(models)} model(s) trained successfully!")

# 6. DETECT ANOMALIES ON TEST DATA
print("\n[6/7] Detecting anomalies in test data...")

# Predict with each enabled model
if 'iforest' in models:
    df_test['anomaly_iforest'] = models['iforest'].predict(X_test)
    df_test['anomaly_score_iforest'] = models['iforest'].decision_function(X_test)
else:
    df_test['anomaly_iforest'] = 0
    df_test['anomaly_score_iforest'] = 0.0

if 'ecod' in models:
    df_test['anomaly_ecod'] = models['ecod'].predict(X_test)
    df_test['anomaly_score_ecod'] = models['ecod'].decision_function(X_test)
else:
    df_test['anomaly_ecod'] = 0
    df_test['anomaly_score_ecod'] = 0.0

if 'lof' in models:
    df_test['anomaly_lof'] = models['lof'].predict(X_test)
    df_test['anomaly_score_lof'] = models['lof'].decision_function(X_test)
else:
    df_test['anomaly_lof'] = 0
    df_test['anomaly_score_lof'] = 0.0

# Combine anomalies from all enabled models
df_test['anomaly_combined'] = ((df_test['anomaly_iforest'] == 1) | 
                                (df_test['anomaly_ecod'] == 1) |
                                (df_test['anomaly_lof'] == 1)).astype(int)

# Track which model(s) detected each anomaly
def get_detection_source(row):
    detected_by = []
    if row['anomaly_iforest'] == 1:
        detected_by.append('iforest')
    if row['anomaly_ecod'] == 1:
        detected_by.append('ecod')
    if row['anomaly_lof'] == 1:
        detected_by.append('lof')
    
    if len(detected_by) == 0:
        return 'none'
    elif len(detected_by) == 1:
        return detected_by[0]
    else:
        return '+'.join(detected_by)

df_test['detection_source'] = df_test.apply(get_detection_source, axis=1)

anomaly_count = df_test['anomaly_combined'].sum()

print(f"Anomalies detected: {anomaly_count} out of {len(df_test)} records ({anomaly_count/len(df_test)*100:.2f}%)")
print(f"Detection breakdown by model(s):")
for source in df_test['detection_source'].unique():
    if source != 'none':
        count = (df_test['detection_source'] == source).sum()
        print(f"  - {source}: {count}")

# 7. VISUALIZE RESULTS
print("\n[7/7] Visualizing results...")

# Determine number of plots needed (1 for time series + 1 per enabled model)
num_plots = 1 + len(models)
fig, axes = plt.subplots(num_plots, 1, figsize=(14, 5 * num_plots))

# Make axes always iterable
if num_plots == 1:
    axes = [axes]

# Plot 1: Total current over time with anomalies
ax1 = axes[0]
normal_data = df_test[df_test['anomaly_combined'] == 0]
anomaly_data = df_test[df_test['anomaly_combined'] == 1]

ax1.plot(normal_data['timestamp'], normal_data['total_power'], 
         'b.', alpha=0.5, markersize=3, label='Normal')
ax1.plot(anomaly_data['timestamp'], anomaly_data['total_power'], 
         'r.', markersize=8, label='Anomaly', alpha=0.7)
ax1.set_xlabel('Timestamp')
ax1.set_ylabel('Total Power (W)')
test_period_str = f"{config.TEST_START_DATE} to {config.TEST_END_DATE}"
ax1.set_title(f'Anomaly Detection Results - {test_period_str}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot score distributions for each enabled model
plot_idx = 1
model_titles = {
    'iforest': 'Isolation Forest',
    'ecod': 'ECOD',
    'lof': 'LOF (Local Outlier Factor)'
}

for model_name in models.keys():
    ax = axes[plot_idx]
    score_col = f'anomaly_score_{model_name}'
    
    ax.hist(df_test[df_test['anomaly_combined'] == 0][score_col], 
            bins=50, alpha=0.6, label='Normal', color='blue')
    ax.hist(df_test[df_test['anomaly_combined'] == 1][score_col], 
            bins=30, alpha=0.6, label='Anomaly', color='red')
    ax.set_xlabel(f'Anomaly Score ({model_titles[model_name]})')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{model_titles[model_name]} - Anomaly Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1

plt.tight_layout()

# Save PNG version
plot_path = config.OUTPUT_DIR / 'step1_anomaly_detection_results.png'
plt.savefig(plot_path, dpi=config.PLOT_DPI, bbox_inches='tight')
print(f"Visualization saved to: {plot_path}")

# Generate LLM summary for anomaly detection results
print("Generating LLM summary for anomaly detection...")
anomaly_summary_prompt = f"""Analyze this anomaly detection result and provide a concise summary (max 200 words):

Test Period: {config.TEST_START_DATE} to {config.TEST_END_DATE}
Total Data Points: {len(df_test):,} hours
Anomalies Detected: {anomaly_count} ({anomaly_count/len(df_test)*100:.2f}%)
Detection Models: {', '.join(models.keys())}

Power Statistics:
- Mean: {df_test['total_power'].mean():.2f} Wh/hour
- Median: {df_test['total_power'].median():.2f} Wh/hour
- Min: {df_test['total_power'].min():.2f} Wh/hour
- Max: {df_test['total_power'].max():.2f} Wh/hour
- Std Dev: {df_test['total_power'].std():.2f} Wh/hour

Anomaly Power Statistics:
- Mean: {anomaly_data['total_power'].mean():.2f} Wh/hour
- Max: {anomaly_data['total_power'].max():.2f} Wh/hour

Provide insights on: 1) What the anomaly rate suggests, 2) Key patterns in anomalous behavior, 3) Potential causes or concerns."""

anomaly_summary = generate_llm_summary(anomaly_summary_prompt)

# Save plot as base64 image for embedding in HTML
buf = BytesIO()
fig.savefig(buf, format='png', dpi=config.PLOT_DPI, bbox_inches='tight')
buf.seek(0)
img_base64 = base64.b64encode(buf.read()).decode('utf-8')
buf.close()

# Create HTML report with embedded image and Markdown-rendered summary
html_path = config.OUTPUT_DIR / 'anomaly_detection_report.html'
html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Anomaly Detection Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .visualization {{
            text-align: center;
            margin: 30px 0;
        }}
        .visualization img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .summary {{
            background-color: #f8f9fa;
            padding: 20px;
            border-left: 4px solid #3498db;
            margin: 20px 0;
            line-height: 1.6;
        }}
        .summary h2 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
            font-style: italic;
            margin-top: 20px;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Anomaly Detection Report</h1>
        
        <div class="visualization">
            <img src="data:image/png;base64,{img_base64}" alt="Anomaly Detection Visualization">
        </div>
        
        <div class="summary">
            <h2>AI-Generated Summary</h2>
            <div>{anomaly_summary.replace(chr(10), '<br>')}</div>
        </div>
        
        <div class="timestamp">
            Generated: {pd.Timestamp.now()}
        </div>
    </div>
</body>
</html>"""

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"HTML report saved to: {html_path}")

plt.close()

# Create separate hourly power usage plot with 3 subplots
print("\nCreating hourly power usage plot...")
fig2, (ax2_top, ax2_middle, ax2_bottom) = plt.subplots(3, 1, figsize=(16, 14))

# Top subplot: Time series plot
ax2_top.plot(df_test['timestamp'], df_test['total_power'], 
             linewidth=1.5, color='#2E86AB', alpha=0.8, label='Total Power')
ax2_top.set_xlabel('Time', fontsize=12, fontweight='bold')
ax2_top.set_ylabel('Power Usage (Wh/hour)', fontsize=12, fontweight='bold')
ax2_top.set_title('Hourly Power Usage Over Time', fontsize=14, fontweight='bold', pad=20)
ax2_top.grid(True, alpha=0.3, linestyle='--')
ax2_top.legend(fontsize=10)
ax2_top.tick_params(axis='x', rotation=45)

# Middle subplot: Average power by hour of day (bar chart)
hourly_avg = df_test.groupby('hour')['total_power'].mean()
hours = hourly_avg.index
avg_power = hourly_avg.values

bars_hour = ax2_middle.bar(hours, avg_power, color='#27ae60', alpha=0.7, edgecolor='#1e8449', linewidth=1.5)

# Highlight peak and lowest hours
peak_hour_idx = hourly_avg.idxmax()
lowest_hour_idx = hourly_avg.idxmin()
bars_hour[peak_hour_idx].set_color('#e74c3c')
bars_hour[peak_hour_idx].set_alpha(0.8)
bars_hour[lowest_hour_idx].set_color('#3498db')
bars_hour[lowest_hour_idx].set_alpha(0.8)

ax2_middle.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
ax2_middle.set_ylabel('Average Power (Wh/hour)', fontsize=12, fontweight='bold')
ax2_middle.set_title('Average Power Usage by Hour of Day', fontsize=14, fontweight='bold', pad=20)
ax2_middle.set_xticks(range(24))
ax2_middle.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right')
ax2_middle.grid(True, alpha=0.3, linestyle='--', axis='y')

# Add legend for highlighted bars
legend_elements_hour = [
    Patch(facecolor='#27ae60', alpha=0.7, label='Normal'),
    Patch(facecolor='#e74c3c', alpha=0.8, label=f'Peak Hour ({peak_hour_idx}:00)'),
    Patch(facecolor='#3498db', alpha=0.8, label=f'Lowest Hour ({lowest_hour_idx}:00)')
]
ax2_middle.legend(handles=legend_elements_hour, fontsize=10)

# Bottom subplot: Average power by day of week (bar chart)
daily_avg = df_test.groupby('day_of_week')['total_power'].mean()
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_avg_values = [daily_avg.get(i, 0) for i in range(7)]

bars_day = ax2_bottom.bar(day_names, day_avg_values, color='#9b59b6', alpha=0.7, edgecolor='#7d3c98', linewidth=1.5)

# Highlight peak and lowest days
peak_day_idx = daily_avg.idxmax()
lowest_day_idx = daily_avg.idxmin()
bars_day[peak_day_idx].set_color('#e74c3c')
bars_day[peak_day_idx].set_alpha(0.8)
bars_day[lowest_day_idx].set_color('#3498db')
bars_day[lowest_day_idx].set_alpha(0.8)

ax2_bottom.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
ax2_bottom.set_ylabel('Average Power (Wh/hour)', fontsize=12, fontweight='bold')
ax2_bottom.set_title('Average Power Usage by Day of Week', fontsize=14, fontweight='bold', pad=20)
ax2_bottom.tick_params(axis='x', rotation=45)
ax2_bottom.grid(True, alpha=0.3, linestyle='--', axis='y')

# Add legend for highlighted bars
legend_elements_day = [
    Patch(facecolor='#9b59b6', alpha=0.7, label='Normal'),
    Patch(facecolor='#e74c3c', alpha=0.8, label=f'Peak Day ({day_names[peak_day_idx]})'),
    Patch(facecolor='#3498db', alpha=0.8, label=f'Lowest Day ({day_names[lowest_day_idx]})')
]
ax2_bottom.legend(handles=legend_elements_day, fontsize=10)

plt.tight_layout()

# Save PNG version
hourly_plot_path = config.OUTPUT_DIR / 'hourly_power_usage.png'
plt.savefig(hourly_plot_path, dpi=config.PLOT_DPI, bbox_inches='tight')
print(f"Hourly power plot saved to: {hourly_plot_path}")

# Generate LLM summary for hourly power usage
print("Generating LLM summary for hourly power usage...")

# Calculate time-based statistics
hourly_stats = df_test.groupby('hour')['total_power'].agg(['mean', 'std', 'min', 'max'])
peak_hour = hourly_stats['mean'].idxmax()
lowest_hour = hourly_stats['mean'].idxmin()

# Weekend vs weekday comparison
weekend_mean = df_test[df_test['is_weekend'] == 1]['total_power'].mean()
weekday_mean = df_test[df_test['is_weekend'] == 0]['total_power'].mean()

hourly_summary_prompt = f"""Analyze this hourly power consumption pattern and provide a concise summary (max 200 words):

Test Period: {config.TEST_START_DATE} to {config.TEST_END_DATE}
Total Hours Analyzed: {len(df_test):,}

Overall Statistics:
- Mean Power: {df_test['total_power'].mean():.2f} Wh/hour
- Median Power: {df_test['total_power'].median():.2f} Wh/hour
- Peak Power: {df_test['total_power'].max():.2f} Wh/hour
- Minimum Power: {df_test['total_power'].min():.2f} Wh/hour
- Standard Deviation: {df_test['total_power'].std():.2f} Wh/hour

Time Patterns:
- Peak Usage Hour: {peak_hour}:00 (avg {hourly_stats.loc[peak_hour, 'mean']:.2f} Wh/hour)
- Lowest Usage Hour: {lowest_hour}:00 (avg {hourly_stats.loc[lowest_hour, 'mean']:.2f} Wh/hour)
- Weekend Average: {weekend_mean:.2f} Wh/hour
- Weekday Average: {weekday_mean:.2f} Wh/hour

Provide insights on: 1) Daily usage patterns, 2) Weekend vs weekday differences, 3) Energy efficiency opportunities."""

hourly_summary = generate_llm_summary(hourly_summary_prompt)

# Save plot as base64 image for embedding in HTML
buf2 = BytesIO()
fig2.savefig(buf2, format='png', dpi=config.PLOT_DPI, bbox_inches='tight')
buf2.seek(0)
img_base64_2 = base64.b64encode(buf2.read()).decode('utf-8')
buf2.close()

# Create HTML report with embedded image and Markdown-rendered summary
hourly_html_path = config.OUTPUT_DIR / 'hourly_power_report.html'
html_content_2 = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Hourly Power Usage Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #27ae60;
            padding-bottom: 10px;
        }}
        .visualization {{
            text-align: center;
            margin: 30px 0;
        }}
        .visualization img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .summary {{
            background-color: #f8f9fa;
            padding: 20px;
            border-left: 4px solid #27ae60;
            margin: 20px 0;
            line-height: 1.6;
        }}
        .summary h2 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
            font-style: italic;
            margin-top: 20px;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Hourly Power Usage Report</h1>
        
        <div class="visualization">
            <img src="data:image/png;base64,{img_base64_2}" alt="Hourly Power Usage Visualization">
        </div>
        
        <div class="summary">
            <h2>AI-Generated Summary</h2>
            <div>{hourly_summary.replace(chr(10), '<br>')}</div>
        </div>
        
        <div class="timestamp">
            Generated: {pd.Timestamp.now()}
        </div>
    </div>
</body>
</html>"""

with open(hourly_html_path, 'w', encoding='utf-8') as f:
    f.write(html_content_2)

print(f"HTML report saved to: {hourly_html_path}")

plt.close()

# 8. SAVE ARTIFACTS FOR NEXT STEPS
print("\n[8/8] Saving artifacts...")

# Save trained models
model_path = config.OUTPUT_DIR / 'trained_models.pkl'
models_to_save = models.copy()
models_to_save['feature_cols'] = feature_cols
models_to_save['power_cols'] = power_cols
models_to_save['energy_cols'] = energy_cols

with open(model_path, 'wb') as f:
    pickle.dump(models_to_save, f)
print(f"  Models saved to: {model_path}")

# Save baseline statistics
stats_path = config.OUTPUT_DIR / 'baseline_statistics.json'
with open(stats_path, 'w') as f:
    json.dump(train_stats, f, indent=2)
print(f"  Statistics saved to: {stats_path}")

# Save detected anomalies with all necessary context
anomalies = df_test[df_test['anomaly_combined'] == 1].copy()
anomaly_cols = ['timestamp', 'total_power', 'hour', 'hour_fraction', 'day_of_week', 
                'is_weekend', 'is_night', 'is_business_hours', 'detection_source']

# Add score columns for enabled models
if 'iforest' in models:
    anomaly_cols.append('anomaly_score_iforest')
if 'ecod' in models:
    anomaly_cols.append('anomaly_score_ecod')
if 'lof' in models:
    anomaly_cols.append('anomaly_score_lof')

# Add power channel columns
anomaly_cols += [col for col in power_cols if col in anomalies.columns]

anomalies_export = anomalies[anomaly_cols].copy()
anomalies_export['timestamp'] = anomalies_export['timestamp'].astype(str)
anomalies_path = config.OUTPUT_DIR / 'detected_anomalies.csv'
anomalies_export.to_csv(anomalies_path, index=False)
print(f"  Anomalies saved to: {anomalies_path}")

# Save all test data points (for time series visualization)
all_test_cols = ['timestamp', 'total_power', 'hour', 'hour_fraction', 'day_of_week', 
                 'is_weekend', 'is_night', 'is_business_hours', 'anomaly_combined', 'detection_source']

# Add score columns for enabled models
if 'iforest' in models:
    all_test_cols.append('anomaly_score_iforest')
if 'ecod' in models:
    all_test_cols.append('anomaly_score_ecod')
if 'lof' in models:
    all_test_cols.append('anomaly_score_lof')

# Add power channel columns
all_test_cols += [col for col in power_cols if col in df_test.columns]

all_test_export = df_test[all_test_cols].copy()
all_test_export['timestamp'] = all_test_export['timestamp'].astype(str)
all_test_path = config.OUTPUT_DIR / 'all_test_data.csv'
all_test_export.to_csv(all_test_path, index=False)
print(f"  All test data saved to: {all_test_path}")

# Save summary report
summary = {
    'timezone': config.TIMEZONE,
    'config': config.get_config_summary(),
    'train_period': f"{df_train['timestamp'].min()} to {df_train['timestamp'].max()}",
    'test_period': f"{df_test['timestamp'].min()} to {df_test['timestamp'].max()}",
    'train_records': len(df_train),
    'test_records': len(df_test),
    'anomalies_detected': int(anomaly_count),
    'anomaly_rate': float(anomaly_count / len(df_test) * 100),
    'feature_columns': feature_cols,
    'power_channels': power_cols,
    'energy_channels': energy_cols
}

summary_path = config.OUTPUT_DIR / 'analysis_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Summary saved to: {summary_path}")

print("\n" + "="*80)
print("STEP 1 COMPLETE!")
print("="*80)
print(f"\nArtifacts saved to '{config.OUTPUT_DIR}/' directory:")
print(f"  1. trained_models.pkl - Trained ML models")
print(f"  2. baseline_statistics.json - Training data statistics")
print(f"  3. detected_anomalies.csv - Anomalies found in test data")
print(f"  4. all_test_data.csv - All test data points (normal + anomalies)")
print(f"  5. analysis_summary.json - Analysis summary")
print(f"  6. step1_anomaly_detection_results.png - Anomaly detection visualization")
print(f"  7. anomaly_detection_report.html - HTML report with graph + AI summary")
print(f"  8. hourly_power_usage.png - Hourly power usage plot")
print(f"  9. hourly_power_report.html - HTML report with power usage + AI summary")
print(f"\nNext: Run step2_generate_prompts.py to create LLM prompts")
print("="*80)
