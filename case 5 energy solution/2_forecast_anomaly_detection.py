"""
Forecast-based anomaly detection using Google TimesFM
Compares actual consumption vs forecasted consumption for each floor+category
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from collections import defaultdict
from dotenv import load_dotenv
from config import (
    TEST_DATA_START,
    FLOORS,
    ANOMALY_FLOORS,
    CATEGORY_KEYS,
    CATEGORY_DISPLAY_MAP, CATEGORY_SHORT_MAP,
    DATA_DIR, OUTPUT_DIR, MODEL_DIR,
    DEVIATION_THRESHOLD,
    ENERGY_DATA_ALL
)

# Load environment variables from .env file
load_dotenv()

# Set HF_TOKEN for Hugging Face authentication (if available)
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    os.environ['HF_TOKEN'] = hf_token
    print("✓ Hugging Face token loaded from environment")

# Create output directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("="*60)
print("FORECAST-BASED ANOMALY DETECTION")
print("="*60)

# Convert TEST_DATA_START to string for pandas comparison
TEST_START_DATE = TEST_DATA_START.strftime('%Y-%m-%d')

# Load data
print("\nLoading data...")
df = pd.read_csv(ENERGY_DATA_ALL, dtype={'floor': str})  # Force floor to be string
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Data loaded: {len(df)} records")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Resample to hourly data
print("\nResampling to hourly data...")
df_hourly = df.groupby([
    pd.Grouper(key='timestamp', freq='h'),
    'floor',
    'category'
]).agg({
    'consumption_kwh': 'sum'
}).reset_index()

print(f"Hourly data: {len(df_hourly)} records")

# Split train/test
df_hourly['period'] = df_hourly['timestamp'].apply(
    lambda x: 'test' if x >= pd.Timestamp(TEST_START_DATE) else 'train'
)

train_data = df_hourly[df_hourly['period'] == 'train'].copy()
test_data = df_hourly[df_hourly['period'] == 'test'].copy()

print(f"Training: {len(train_data)} records ({train_data['timestamp'].min()} to {train_data['timestamp'].max()})")
print(f"Test: {len(test_data)} records ({test_data['timestamp'].min()} to {test_data['timestamp'].max()})")

# ============================================================================
# TimesFM Forecasting for each Floor + Category
# ============================================================================
print("\n" + "="*60)
print("FORECASTING WITH TIMESFM")
print("="*60)

try:
    import torch
    import timesfm
    
    # Initialize TimesFM
    print("\nInitializing TimesFM model...")
    torch.set_float32_matmul_precision("high")
    
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    tfm.compile(
        timesfm.ForecastConfig(
            max_context=1024,  # Use past 1024 hours (~42 days)
            max_horizon=744,   # Forecast 744 hours (31 days for March)
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    print("✓ TimesFM model loaded and compiled")
    
    TIMESFM_AVAILABLE = True
    
except (ImportError, Exception) as e:
    print(f"\n⚠ TimesFM not available ({e}). Using simple moving average forecast instead.")
    TIMESFM_AVAILABLE = False

# Get all floor+category combinations
combinations = df_hourly[['floor', 'category']].drop_duplicates().sort_values(['floor', 'category'])
print(f"\nForecasting for {len(combinations)} floor+category combinations...")

forecasts = []
deviations_summary = []

for idx, (_, row) in enumerate(combinations.iterrows(), 1):
    floor = row['floor']
    category = row['category']
    
    print(f"\n{idx}. Floor {floor} - {category}")
    
    # Get time series for this combination
    ts_train = train_data[
        (train_data['floor'] == floor) & 
        (train_data['category'] == category)
    ].sort_values('timestamp')
    
    ts_test = test_data[
        (test_data['floor'] == floor) & 
        (test_data['category'] == category)
    ].sort_values('timestamp')
    
    if len(ts_train) == 0 or len(ts_test) == 0:
        print(f"   ⚠ Insufficient data, skipping...")
        continue
    
    # Prepare training data
    train_values = ts_train['consumption_kwh'].values
    test_timestamps = ts_test['timestamp'].values
    test_actual = ts_test['consumption_kwh'].values
    
    if TIMESFM_AVAILABLE:
        # Use TimesFM for forecasting
        try:
            # Forecast using the new API
            point_forecast, quantile_forecast = tfm.forecast(
                horizon=len(test_actual),  # Forecast length
                inputs=[train_values]      # Input time series
            )
            forecast_values = point_forecast[0]  # Get point forecast for first (only) series
            
            print(f"   ✓ TimesFM forecast generated")
            
        except Exception as e:
            print(f"   ⚠ TimesFM error: {e}, using moving average")
            # Fallback to moving average
            window = min(168, len(train_values))  # 1 week or less
            forecast_values = np.full(len(test_actual), train_values[-window:].mean())
    else:
        # Use simple moving average forecast
        window = min(168, len(train_values))  # 1 week or less
        forecast_values = np.full(len(test_actual), train_values[-window:].mean())
        print(f"   ✓ Moving average forecast generated (window={window}h)")
    
    # Calculate deviations
    deviation = test_actual - forecast_values
    deviation_pct = (deviation / (forecast_values + 1e-6)) * 100
    
    # Store results
    for i in range(len(test_actual)):
        forecasts.append({
            'timestamp': test_timestamps[i],
            'floor': floor,
            'category': category,
            'actual': test_actual[i],
            'forecast': forecast_values[i],
            'deviation': deviation[i],
            'deviation_pct': deviation_pct[i]
        })
    
    # Summary statistics
    mean_deviation = deviation.mean()
    mean_deviation_pct = deviation_pct.mean()
    max_deviation = deviation.max()
    total_excess = deviation[deviation > 0].sum()
    
    deviations_summary.append({
        'floor': floor,
        'category': category,
        'mean_deviation_kwh': mean_deviation,
        'mean_deviation_pct': mean_deviation_pct,
        'max_deviation_kwh': max_deviation,
        'total_excess_kwh': total_excess,
        'anomaly_hours': np.sum(np.abs(deviation_pct) > DEVIATION_THRESHOLD * 100)
    })
    
    print(f"   Mean deviation: {mean_deviation:+.2f} kWh ({mean_deviation_pct:+.1f}%)")
    print(f"   Total excess: {total_excess:.1f} kWh")
    print(f"   Anomaly hours (>{DEVIATION_THRESHOLD*100:.0f}% deviation): {deviations_summary[-1]['anomaly_hours']}")

# Convert to DataFrames
df_forecasts = pd.DataFrame(forecasts)
df_deviations = pd.DataFrame(deviations_summary)

# Save results
df_forecasts.to_csv('data/forecast_results.csv', index=False)
df_deviations.to_csv('data/deviation_summary.csv', index=False)

print("\n✓ Forecast results saved: data/forecast_results.csv")
print("✓ Deviation summary saved: data/deviation_summary.csv")

# ============================================================================
# Anomaly Analysis
# ============================================================================
print("\n" + "="*60)
print("ANOMALY ANALYSIS")
print("="*60)

# Sort by total excess
df_deviations_sorted = df_deviations.sort_values('total_excess_kwh', ascending=False)

print("\nTop contributors to excess consumption:")
print("-"*60)
for idx, row in df_deviations_sorted.head(10).iterrows():
    cat_name = {'air_con': 'Air Con', 'computer_appliance': 'Computer', 
                'lobby_corridor_lighting': 'Lighting'}.get(row['category'], row['category'])
    print(f"Floor {row['floor']} - {cat_name:20s} | "
          f"{row['total_excess_kwh']:7.1f} kWh | "
          f"{row['mean_deviation_pct']:+6.1f}% avg | "
          f"{row['anomaly_hours']:3.0f} anomaly hours")

# Calculate total excess
total_excess_all = df_deviations['total_excess_kwh'].sum()
print(f"\nTotal excess consumption: {total_excess_all:.1f} kWh")

# Contribution by category
category_contrib = df_deviations.groupby('category')['total_excess_kwh'].sum().sort_values(ascending=False)
print("\nContribution by category:")
for cat, excess in category_contrib.items():
    cat_name = {'air_con': 'Air Conditioning', 'computer_appliance': 'Computer Appliances', 
                'lobby_corridor_lighting': 'Lobby/Corridor Lighting'}.get(cat, cat)
    pct = (excess / total_excess_all * 100) if total_excess_all > 0 else 0
    print(f"  {cat_name:30s} | {excess:7.1f} kWh | {pct:5.1f}%")

# ============================================================================
# Data-Driven Analysis Results (No Hardcoded Context)
# ============================================================================
print("\n" + "="*60)
print("DATA ANALYSIS RESULTS")
print("="*60)

# Prepare pure data insights
data_insights = {
    'total_excess_kwh': float(total_excess_all),
    'top_contributors': [],
    'category_contributions': category_contrib.to_dict(),
    'test_period_dates': {
        'start': str(test_data['timestamp'].min()),
        'end': str(test_data['timestamp'].max())
    },
    'summary_statistics': {
        'total_records_analyzed': len(df_forecasts),
        'floor_category_combinations': len(df_deviations),
        'average_deviation_pct': float(df_deviations['mean_deviation_pct'].mean())
    }
}

# Top contributors (data only, no interpretation)
for _, row in df_deviations_sorted.head(5).iterrows():
    data_insights['top_contributors'].append({
        'floor': int(row['floor']),
        'category': str(row['category']),
        'total_excess_kwh': float(row['total_excess_kwh']),
        'mean_deviation_kwh': float(row['mean_deviation_kwh']),
        'mean_deviation_pct': float(row['mean_deviation_pct']),
        'max_deviation_kwh': float(row['max_deviation_kwh']),
        'anomaly_hours': int(row['anomaly_hours']),
        'percentage_of_total': float((row['total_excess_kwh'] / total_excess_all * 100) if total_excess_all > 0 else 0)
    })

print(f"\nTotal excess consumption: {total_excess_all:.1f} kWh")
print(f"Test period: {data_insights['test_period_dates']['start']} to {data_insights['test_period_dates']['end']}")
print(f"Average deviation: {data_insights['summary_statistics']['average_deviation_pct']:.1f}%")

# Save data insights
with open('data/forecast_data_insights.json', 'w') as f:
    json.dump(data_insights, f, indent=2)

print("\n✓ Data insights saved: data/forecast_data_insights.json")

# ============================================================================
# Load Context and Generate LLM Prompt
# ============================================================================
print("\n" + "="*60)
print("GENERATING LLM PROMPT")
print("="*60)

# Load context from JSON (if exists)
context_data = None
try:
    with open('context.json', 'r') as f:
        context_data = json.load(f)
    print("\n✓ Context loaded from context.json")
except FileNotFoundError:
    print("\n⚠ context.json not found - generating prompt without context")

# Generate LLM prompt combining data + context
llm_prompt = {
    'task': 'Generate a business narrative explaining the energy consumption anomalies detected in the forecast analysis.',
    'data_insights': data_insights,
    'context': context_data,
    'instructions': [
        'Analyze the data insights provided, focusing on the top contributors to excess consumption',
        'Use the context information to provide business-relevant interpretations',
        'Explain what the deviations mean in practical terms for building management',
        'Identify patterns by time of day, floor, and equipment category',
        'Suggest potential causes and recommended actions',
        'Write in a clear, professional tone suitable for facility managers'
    ],
    'output_format': 'A narrative summary (2-3 paragraphs) followed by key findings and recommendations'
}

# Save LLM prompt
with open('data/forecast_llm_prompt.json', 'w') as f:
    json.dump(llm_prompt, f, indent=2)

print("✓ LLM prompt saved: data/forecast_llm_prompt.json")
print("\nPrompt structure:")
print(f"  - Task: {llm_prompt['task']}")
print(f"  - Data insights: {len(data_insights['top_contributors'])} top contributors")
print(f"  - Context available: {'Yes' if context_data else 'No'}")
print(f"  - Instructions: {len(llm_prompt['instructions'])} guidelines")

# ============================================================================
# Anomaly Floors Detailed Hourly Deviation Analysis
# ============================================================================

# Add day of week and hour to forecasts (only once)
if 'day_of_week' not in df_forecasts.columns:
    df_forecasts['day_of_week'] = pd.to_datetime(df_forecasts['timestamp']).dt.dayofweek
if 'hour' not in df_forecasts.columns:
    df_forecasts['hour'] = pd.to_datetime(df_forecasts['timestamp']).dt.hour

# Debug: Check floor values
print(f"\nDEBUG: Unique floors in df_forecasts: {df_forecasts['floor'].unique()}")
print(f"DEBUG: Floor data type: {df_forecasts['floor'].dtype}")
print(f"DEBUG: ANOMALY_FLOORS: {ANOMALY_FLOORS}")

# Analyze each anomaly floor
for floor_id in ANOMALY_FLOORS:
    print("\n" + "="*60)
    print(f"FLOOR {floor_id} HOURLY DEVIATION ANALYSIS (WEEKDAYS)")
    print("="*60)
    
    # Filter for this floor, weekdays only
    df_floor_weekday = df_forecasts[
        (df_forecasts['floor'] == floor_id) & 
        (df_forecasts['day_of_week'] < 5)
    ].copy()
    
    if len(df_floor_weekday) == 0:
        print(f"No data found for Floor {floor_id}")
        print(f"DEBUG: Trying to match '{floor_id}' (type: {type(floor_id)})")
        continue
    
    # Analyze by category
    floor_hourly_results = []
    
    for category in CATEGORY_KEYS:
        print(f"\n{CATEGORY_DISPLAY_MAP[category]}:")
        print("-" * 60)
        print(f"{'Hour':>6} | {'Actual':>8} | {'Forecast':>8} | {'Deviation':>10} | {'Dev %':>8}")
        print("-" * 60)
        
        cat_data = df_floor_weekday[df_floor_weekday['category'] == category]
        
        if len(cat_data) > 0:
            # Calculate average by hour
            hourly_stats = cat_data.groupby('hour').agg({
                'actual': 'mean',
                'forecast': 'mean',
                'deviation': 'mean',
                'deviation_pct': 'mean'
            }).reset_index()
            
            for _, row in hourly_stats.iterrows():
                hour = int(row['hour'])
                actual = row['actual']
                forecast = row['forecast']
                deviation = row['deviation']
                deviation_pct = row['deviation_pct']
                
                # Highlight anomaly hours based on deviation threshold
                is_anomaly = abs(deviation_pct) > (DEVIATION_THRESHOLD * 100)
                marker = " ⚠" if is_anomaly else ""
                
                print(f"{hour:>6} | {actual:>8.2f} | {forecast:>8.2f} | {deviation:>+10.2f} | {deviation_pct:>+7.1f}%{marker}")
                
                floor_hourly_results.append({
                    'category': category,
                    'hour': hour,
                    'actual_avg': actual,
                    'forecast_avg': forecast,
                    'deviation_avg': deviation,
                    'deviation_pct_avg': deviation_pct
                })
    
    # Save this floor's hourly analysis
    df_floor_hourly = pd.DataFrame(floor_hourly_results)
    df_floor_hourly.to_csv(f'{DATA_DIR}/floor{floor_id}_hourly_deviation.csv', index=False)
    print(f"\n✓ Floor {floor_id} hourly analysis saved: {DATA_DIR}/floor{floor_id}_hourly_deviation.csv")
    
    # Summary statistics for this floor
    print("\n" + "="*60)
    print(f"FLOOR {floor_id} SUMMARY BY CATEGORY")
    print("="*60)
    
    for category in CATEGORY_KEYS:
        cat_data = df_floor_hourly[df_floor_hourly['category'] == category]
        
        if len(cat_data) > 0:
            overall_dev = cat_data['deviation_pct_avg'].mean()
            max_dev_hour = cat_data.loc[cat_data['deviation_pct_avg'].idxmax()]
            
            # Identify anomaly hours based on threshold
            anomaly_hours = cat_data[abs(cat_data['deviation_pct_avg']) > (DEVIATION_THRESHOLD * 100)]
            anomaly_avg_dev = anomaly_hours['deviation_pct_avg'].mean() if len(anomaly_hours) > 0 else 0
            anomaly_hour_list = anomaly_hours['hour'].tolist() if len(anomaly_hours) > 0 else []
            
            print(f"\n{CATEGORY_DISPLAY_MAP[category]}:")
            print(f"  Overall avg deviation: {overall_dev:+.1f}%")
            print(f"  Max deviation: {max_dev_hour['deviation_pct_avg']:+.1f}% at hour {int(max_dev_hour['hour'])}")
            print(f"  Anomaly hours (>{DEVIATION_THRESHOLD*100:.0f}% threshold): {len(anomaly_hours)} hours")
            if len(anomaly_hour_list) > 0:
                print(f"    Hours: {', '.join(map(str, sorted(anomaly_hour_list)))}")
                print(f"    Avg deviation: {anomaly_avg_dev:+.1f}%")

# ============================================================================
# Visualizations
# ============================================================================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Forecast-Based Anomaly Detection Results', fontsize=16, fontweight='bold')

# 1. Top contributors
ax1 = axes[0, 0]
top_10 = df_deviations_sorted.head(10)
labels = [f"F{row['floor']} {row['category'][:8]}" for _, row in top_10.iterrows()]
ax1.barh(range(len(top_10)), top_10['total_excess_kwh'], color='#e74c3c')
ax1.set_yticks(range(len(top_10)))
ax1.set_yticklabels(labels, fontsize=8)
ax1.set_xlabel('Total Excess (kWh)')
ax1.set_title('Top 10 Contributors to Excess Consumption')
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')

# 2. Category contribution pie chart
ax2 = axes[0, 1]
cat_labels = [{'air_con': 'Air Con', 'computer_appliance': 'Computer', 
               'lobby_corridor_lighting': 'Lighting'}.get(cat, cat) for cat in category_contrib.index]
colors = ['#e74c3c', '#3498db', '#2ecc71']
ax2.pie(category_contrib.values, labels=cat_labels, autopct='%1.1f%%', colors=colors, startangle=90)
ax2.set_title('Excess Consumption by Category')

# 3. Mean deviation percentage by floor+category
ax3 = axes[0, 2]
df_dev_pivot = df_deviations.pivot(index='floor', columns='category', values='mean_deviation_pct')
df_dev_pivot.plot(kind='bar', ax=ax3, color=['#e74c3c', '#3498db', '#2ecc71'])
ax3.set_xlabel('Floor')
ax3.set_ylabel('Mean Deviation (%)')
ax3.set_title('Average Deviation by Floor & Category')
ax3.legend(title='Category', labels=['Air Con', 'Computer', 'Lighting'])
ax3.grid(True, alpha=0.3, axis='y')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 4. Time series example - worst case
ax4 = axes[1, 0]
worst_case = df_deviations_sorted.iloc[0]
worst_floor = worst_case['floor']
worst_cat = worst_case['category']
worst_data = df_forecasts[
    (df_forecasts['floor'] == worst_floor) & 
    (df_forecasts['category'] == worst_cat)
].sort_values('timestamp')

ax4.plot(worst_data['timestamp'], worst_data['actual'], label='Actual', color='#e74c3c', linewidth=1.5)
ax4.plot(worst_data['timestamp'], worst_data['forecast'], label='Forecast', color='#3498db', linestyle='--', linewidth=1.5)
ax4.fill_between(worst_data['timestamp'], worst_data['forecast'], worst_data['actual'], 
                  where=(worst_data['actual'] > worst_data['forecast']), alpha=0.3, color='#e74c3c', label='Excess')
ax4.set_xlabel('Date')
ax4.set_ylabel('Consumption (kWh/h)')
ax4.set_title(f'Worst Case: Floor {worst_floor} - {worst_cat}')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

# 5. Deviation distribution
ax5 = axes[1, 1]
all_deviations = df_forecasts['deviation_pct'].values
ax5.hist(all_deviations, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
ax5.axvline(x=0, color='black', linestyle='-', linewidth=2)
ax5.axvline(x=DEVIATION_THRESHOLD*100, color='red', linestyle='--', linewidth=2, label=f'Threshold ({DEVIATION_THRESHOLD*100:.0f}%)')
ax5.axvline(x=-DEVIATION_THRESHOLD*100, color='red', linestyle='--', linewidth=2)
ax5.set_xlabel('Deviation (%)')
ax5.set_ylabel('Frequency')
ax5.set_title('Distribution of Deviations')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Anomaly hours by floor+category
ax6 = axes[1, 2]
df_anom_pivot = df_deviations.pivot(index='floor', columns='category', values='anomaly_hours')
df_anom_pivot.plot(kind='bar', ax=ax6, color=['#e74c3c', '#3498db', '#2ecc71'])
ax6.set_xlabel('Floor')
ax6.set_ylabel('Anomaly Hours')
ax6.set_title(f'Anomaly Hours (>{DEVIATION_THRESHOLD*100:.0f}% deviation)')
ax6.legend(title='Category', labels=['Air Con', 'Computer', 'Lighting'])
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('output/2_forecast_anomaly_detection.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved: output/2_forecast_anomaly_detection.png")
plt.show()

# ============================================================================
# Individual Hourly Pattern Plots (weekdays only, one per floor+category)
# ============================================================================
print("\nGenerating hourly pattern plots (weekdays only)...")

# day_of_week and hour columns already added above
# Filter for weekdays only (Monday=0 to Friday=4)
df_forecasts_weekday = df_forecasts[df_forecasts['day_of_week'] < 5].copy()

# Create 3x3 grid for 9 combinations
fig_ts, axes_ts = plt.subplots(3, 3, figsize=(20, 12))
fig_ts.suptitle('Average Hourly Pattern: Actual vs Forecast (Weekdays Only)', fontsize=16, fontweight='bold')

category_order = ['air_con', 'computer_appliance', 'lobby_corridor_lighting']
category_names = {'air_con': 'Air Conditioning', 'computer_appliance': 'Computer Appliances', 
                  'lobby_corridor_lighting': 'Lobby/Corridor Lighting'}

for floor_idx, floor in enumerate(FLOORS):
    for cat_idx, category in enumerate(category_order):
        ax = axes_ts[floor_idx, cat_idx]
        
        # Get data for this combination (weekdays only)
        combo_data = df_forecasts_weekday[
            (df_forecasts_weekday['floor'] == floor) & 
            (df_forecasts_weekday['category'] == category)
        ]
        
        if len(combo_data) > 0:
            # Calculate average by hour
            hourly_avg = combo_data.groupby('hour').agg({
                'actual': 'mean',
                'forecast': 'mean',
                'deviation_pct': 'mean'
            }).reset_index()
            
            # Plot bars for actual - green for normal, red for anomaly hours based on threshold
            hours = hourly_avg['hour'].values
            colors = ['#e74c3c' if abs(dev) > (DEVIATION_THRESHOLD * 100) else '#2ecc71' 
                     for dev in hourly_avg['deviation_pct'].values]
            
            ax.bar(hours, hourly_avg['actual'], alpha=0.7, color=colors, 
                   width=0.8, edgecolor='black', linewidth=0.5)
            
            # Add legend entries manually
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2ecc71', alpha=0.7, edgecolor='black', label='Actual (normal)'),
                Patch(facecolor='#e74c3c', alpha=0.7, edgecolor='black', label='Actual (anomaly)'),
                ax.plot([], [], color='#3498db', linewidth=2.5, marker='o', markersize=4, label='Forecast (avg)')[0]
            ]
            
            ax.plot(hours, hourly_avg['forecast'], color='#3498db', 
                   linewidth=2.5, marker='o', markersize=4)
            
            # Get statistics for this combination
            combo_stats = df_deviations[
                (df_deviations['floor'] == floor) & 
                (df_deviations['category'] == category)
            ]
            
            if len(combo_stats) > 0:
                mean_dev = combo_stats.iloc[0]['mean_deviation_pct']
                total_excess = combo_stats.iloc[0]['total_excess_kwh']
                
                # Title with statistics
                ax.set_title(f"Floor {floor} - {category_names[category]}\n"
                           f"Avg Dev: {mean_dev:+.1f}% | Excess: {total_excess:.1f} kWh",
                           fontsize=10, fontweight='bold')
            else:
                ax.set_title(f"Floor {floor} - {category_names[category]}", fontsize=10)
            
            ax.set_xlabel('Hour of Day', fontsize=8)
            ax.set_ylabel('Average kWh/hour', fontsize=8)
            ax.set_xticks(range(0, 24, 2))
            ax.legend(handles=legend_elements, fontsize=7, loc='upper left')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xlim(-0.5, 23.5)
            ax.tick_params(axis='x', labelsize=7)
            ax.tick_params(axis='y', labelsize=7)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Floor {floor} - {category_names[category]}", fontsize=10)

plt.tight_layout()
plt.savefig('output/2_forecast_hourly_patterns.png', dpi=300, bbox_inches='tight')
print("✓ Hourly pattern plots saved: output/2_forecast_hourly_patterns.png")
plt.show()

print("\n✓ Forecast-based anomaly detection complete!")
