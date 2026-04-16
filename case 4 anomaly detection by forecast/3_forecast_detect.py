import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import torch
import timesfm
from dotenv import load_dotenv

load_dotenv()

torch.set_float32_matmul_precision("high")

print("Loading data...")
df = pd.read_csv('data/server_room_temperature.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("Loading configuration...")
with open('output/config.pkl', 'rb') as f:
    config_info = pickle.load(f)

context_len = config_info['context_len']
horizon_len = config_info['horizon_len']
train_end_idx = config_info['train_end_idx']

print("Initializing TimesFM model...")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

print("Compiling model...")
model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)

print(f"Model loaded successfully")
print(f"Context length: {context_len}")

forecast_start_idx = train_end_idx
forecast_horizon_hours = 6
steps_per_hour = 4
num_forecasts = forecast_horizon_hours * steps_per_hour
forecast_end_idx = min(forecast_start_idx + num_forecasts, len(df))
num_forecasts = forecast_end_idx - forecast_start_idx

print(f"\nForecasting from index {forecast_start_idx} ({df['timestamp'].iloc[forecast_start_idx]})")
print(f"Forecast duration: {forecast_horizon_hours} hours ({num_forecasts} steps)")

actual_values = df.iloc[forecast_start_idx:forecast_end_idx]['temperature'].values
forecast_timestamps = df.iloc[forecast_start_idx:forecast_end_idx]['timestamp'].values

print("\n" + "="*80)
print("METHOD 1: FIXED WINDOW FORECASTING")
print("="*80)
print("Forecast all steps at once using only pre-anomaly data\n")

context_data_fixed = df.iloc[forecast_start_idx-context_len:forecast_start_idx]['temperature'].values

point_forecast_fixed, quantile_forecast_fixed = model.forecast(
    horizon=num_forecasts,
    inputs=[context_data_fixed],
)

forecast_fixed = point_forecast_fixed[0]
lower_fixed = quantile_forecast_fixed[0, :, 1]
upper_fixed = quantile_forecast_fixed[0, :, 9]

residuals_fixed = actual_values - forecast_fixed
anomalies_fixed = (actual_values > upper_fixed) | (actual_values < lower_fixed)
anomaly_indices_fixed = np.where(anomalies_fixed)[0]

print(f"Fixed Window Statistics:")
print(f"Residual std: {np.std(residuals_fixed):.4f}°C")
print(f"RMSE: {np.sqrt(np.mean(residuals_fixed**2)):.4f}°C")
print(f"MAE: {np.mean(np.abs(residuals_fixed)):.4f}°C")
print(f"Anomalies detected: {np.sum(anomalies_fixed)} out of {len(anomalies_fixed)} points ({100*np.sum(anomalies_fixed)/len(anomalies_fixed):.1f}%)")

if len(anomaly_indices_fixed) > 0:
    first_idx = anomaly_indices_fixed[0]
    print(f"First anomaly at step {first_idx} ({forecast_timestamps[first_idx]})")

print("\n" + "="*80)
print("METHOD 2: MOVING WINDOW FORECASTING")
print("="*80)
print("Forecast one step at a time, updating context with each new observation\n")

forecast_moving = []
lower_moving = []
upper_moving = []

for i in range(num_forecasts):
    current_idx = forecast_start_idx + i
    
    context_start = max(0, current_idx - context_len)
    context_data = df.iloc[context_start:current_idx]['temperature'].values
    
    point_forecast, quantile_forecast = model.forecast(
        horizon=1,
        inputs=[context_data],
    )
    
    forecast_moving.append(point_forecast[0, 0])
    lower_moving.append(quantile_forecast[0, 0, 1])
    upper_moving.append(quantile_forecast[0, 0, 9])
    
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{num_forecasts} forecasts...")

forecast_moving = np.array(forecast_moving)
lower_moving = np.array(lower_moving)
upper_moving = np.array(upper_moving)

residuals_moving = actual_values - forecast_moving
anomalies_moving = (actual_values > upper_moving) | (actual_values < lower_moving)
anomaly_indices_moving = np.where(anomalies_moving)[0]

print(f"\nMoving Window Statistics:")
print(f"Residual std: {np.std(residuals_moving):.4f}°C")
print(f"RMSE: {np.sqrt(np.mean(residuals_moving**2)):.4f}°C")
print(f"MAE: {np.mean(np.abs(residuals_moving)):.4f}°C")
print(f"Anomalies detected: {np.sum(anomalies_moving)} out of {len(anomalies_moving)} points ({100*np.sum(anomalies_moving)/len(anomalies_moving):.1f}%)")

if len(anomaly_indices_moving) > 0:
    first_idx = anomaly_indices_moving[0]
    print(f"First anomaly at step {first_idx} ({forecast_timestamps[first_idx]})")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"Fixed Window  - RMSE: {np.sqrt(np.mean(residuals_fixed**2)):.4f}°C, Anomalies: {np.sum(anomalies_fixed)}")
print(f"Moving Window - RMSE: {np.sqrt(np.mean(residuals_moving**2)):.4f}°C, Anomalies: {np.sum(anomalies_moving)}")

results_df = pd.DataFrame({
    'timestamp': forecast_timestamps,
    'actual': actual_values,
    'forecast_fixed': forecast_fixed,
    'upper_fixed': upper_fixed,
    'lower_fixed': lower_fixed,
    'residual_fixed': residuals_fixed,
    'anomaly_fixed': anomalies_fixed,
    'forecast_moving': forecast_moving,
    'upper_moving': upper_moving,
    'lower_moving': lower_moving,
    'residual_moving': residuals_moving,
    'anomaly_moving': anomalies_moving,
})

os.makedirs('output', exist_ok=True)

results_df.to_csv('output/forecast_results.csv', index=False)
print("\nForecast results saved to: output/forecast_results.csv")

seven_days_samples = 7 * 24 * 4
display_start_idx = max(0, forecast_start_idx - seven_days_samples)

fig, axes = plt.subplots(3, 1, figsize=(16, 14))

historical_data = df.iloc[display_start_idx:forecast_start_idx]

ax1 = axes[0]
ax1.plot(historical_data['timestamp'], historical_data['temperature'], 
         label='Historical (Past 7 Days)', color='blue', linewidth=1, alpha=0.7)
ax1.plot(forecast_timestamps, actual_values, 
         label='Actual', color='black', linewidth=1.5, alpha=0.8)
ax1.plot(forecast_timestamps, forecast_fixed, 
         label='Fixed Window Forecast', color='green', linewidth=1.5, linestyle='--')
ax1.fill_between(forecast_timestamps, lower_fixed, upper_fixed, 
                  alpha=0.2, color='green', label='Fixed 80% Interval')
if len(anomaly_indices_fixed) > 0:
    ax1.scatter(forecast_timestamps[anomaly_indices_fixed], 
                actual_values[anomaly_indices_fixed],
                color='red', s=50, zorder=5, label='Fixed Anomalies', marker='x', linewidths=2)
ax1.axvline(x=forecast_timestamps[0], color='orange', linestyle=':', 
            label='Forecast Start', linewidth=2)
ax1.set_xlabel('Timestamp')
ax1.set_ylabel('Temperature (°C)')
ax1.set_title('Method 1: Fixed Window Forecasting')
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

ax2 = axes[1]
ax2.plot(historical_data['timestamp'], historical_data['temperature'], 
         label='Historical (Past 7 Days)', color='blue', linewidth=1, alpha=0.7)
ax2.plot(forecast_timestamps, actual_values, 
         label='Actual', color='black', linewidth=1.5, alpha=0.8)
ax2.plot(forecast_timestamps, forecast_moving, 
         label='Moving Window Forecast', color='purple', linewidth=1.5, linestyle='--')
ax2.fill_between(forecast_timestamps, lower_moving, upper_moving, 
                  alpha=0.2, color='purple', label='Moving 80% Interval')
if len(anomaly_indices_moving) > 0:
    ax2.scatter(forecast_timestamps[anomaly_indices_moving], 
                actual_values[anomaly_indices_moving],
                color='red', s=50, zorder=5, label='Moving Anomalies', marker='x', linewidths=2)
ax2.axvline(x=forecast_timestamps[0], color='orange', linestyle=':', 
            label='Forecast Start', linewidth=2)
ax2.set_xlabel('Timestamp')
ax2.set_ylabel('Temperature (°C)')
ax2.set_title('Method 2: Moving Window Forecasting')
ax2.legend(loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

ax3 = axes[2]
ax3.plot(forecast_timestamps, residuals_fixed, color='green', linewidth=1, marker='o', 
         markersize=3, label='Fixed Window Residuals', alpha=0.7)
ax3.plot(forecast_timestamps, residuals_moving, color='purple', linewidth=1, marker='s', 
         markersize=3, label='Moving Window Residuals', alpha=0.7)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
if len(anomaly_indices_fixed) > 0:
    ax3.scatter(forecast_timestamps[anomaly_indices_fixed], 
                residuals_fixed[anomaly_indices_fixed],
                color='green', s=50, zorder=5, marker='x', linewidths=2, alpha=0.7)
if len(anomaly_indices_moving) > 0:
    ax3.scatter(forecast_timestamps[anomaly_indices_moving], 
                residuals_moving[anomaly_indices_moving],
                color='purple', s=50, zorder=5, marker='x', linewidths=2, alpha=0.7)
ax3.set_xlabel('Timestamp')
ax3.set_ylabel('Residual (°C)')
ax3.set_title('Residual Comparison: Fixed vs Moving Window')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('output/3_forecast_anomaly_detection.png', dpi=150)
plt.show()

print("Forecast visualization saved to: output/3_forecast_anomaly_detection.png")

fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))

ax_hist1 = axes2[0, 0]
ax_hist1.hist(residuals_fixed, bins=20, color='green', alpha=0.6, edgecolor='black', label='Fixed')
ax_hist1.set_xlabel('Residual (°C)')
ax_hist1.set_ylabel('Frequency')
ax_hist1.set_title('Fixed Window - Residual Distribution')
ax_hist1.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax_hist1.legend()
ax_hist1.grid(True, alpha=0.3)

ax_hist2 = axes2[0, 1]
ax_hist2.hist(residuals_moving, bins=20, color='purple', alpha=0.6, edgecolor='black', label='Moving')
ax_hist2.set_xlabel('Residual (°C)')
ax_hist2.set_ylabel('Frequency')
ax_hist2.set_title('Moving Window - Residual Distribution')
ax_hist2.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax_hist2.legend()
ax_hist2.grid(True, alpha=0.3)

ax_scatter1 = axes2[1, 0]
ax_scatter1.scatter(forecast_fixed, actual_values, alpha=0.5, s=20, color='green', label='Fixed')
if len(anomaly_indices_fixed) > 0:
    ax_scatter1.scatter(forecast_fixed[anomaly_indices_fixed], 
                       actual_values[anomaly_indices_fixed],
                       color='red', s=50, marker='x', linewidths=2, label='Anomalies')
min_val = min(forecast_fixed.min(), actual_values.min())
max_val = max(forecast_fixed.max(), actual_values.max())
ax_scatter1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
ax_scatter1.set_xlabel('Forecast Temperature (°C)')
ax_scatter1.set_ylabel('Actual Temperature (°C)')
ax_scatter1.set_title('Fixed Window - Forecast vs Actual')
ax_scatter1.legend()
ax_scatter1.grid(True, alpha=0.3)

ax_table = axes2[1, 1]
ax_table.axis('off')
summary_stats = [
    ['Metric', 'Fixed', 'Moving'],
    ['Total Points', f'{len(actual_values)}', f'{len(actual_values)}'],
    ['Anomalies', f'{np.sum(anomalies_fixed)}', f'{np.sum(anomalies_moving)}'],
    ['Anomaly Rate', f'{100*np.sum(anomalies_fixed)/len(anomalies_fixed):.1f}%', f'{100*np.sum(anomalies_moving)/len(anomalies_moving):.1f}%'],
    ['RMSE', f'{np.sqrt(np.mean(residuals_fixed**2)):.4f}°C', f'{np.sqrt(np.mean(residuals_moving**2)):.4f}°C'],
    ['MAE', f'{np.mean(np.abs(residuals_fixed)):.4f}°C', f'{np.mean(np.abs(residuals_moving)):.4f}°C'],
    ['Residual Std', f'{np.std(residuals_fixed):.4f}°C', f'{np.std(residuals_moving):.4f}°C'],
]
table = ax_table.table(cellText=summary_stats, cellLoc='center', loc='center',
                       colWidths=[0.4, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
for i in range(len(summary_stats)):
    if i == 0:
        table[(i, 0)].set_facecolor('#4CAF50')
        table[(i, 1)].set_facecolor('#4CAF50')
        table[(i, 2)].set_facecolor('#4CAF50')
        table[(i, 0)].set_text_props(weight='bold', color='white')
        table[(i, 1)].set_text_props(weight='bold', color='white')
        table[(i, 2)].set_text_props(weight='bold', color='white')
    else:
        table[(i, 0)].set_facecolor('#f0f0f0')
ax_table.set_title('Comparison Summary', fontsize=12, weight='bold', pad=20)

plt.tight_layout()
plt.savefig('output/3_detailed_analysis.png', dpi=150)
plt.show()

print("Detailed analysis saved to: output/3_detailed_analysis.png")
print("\n=== Anomaly Detection Complete ===")
