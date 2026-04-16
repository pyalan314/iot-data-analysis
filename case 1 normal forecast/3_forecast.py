import pandas as pd
import matplotlib.pyplot as plt
import torch
from neuralforecast import NeuralForecast
import timesfm
from pathlib import Path

if Path('.env').exists():
    from dotenv import load_dotenv
    load_dotenv()

df = pd.read_csv('data/office_power_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

weekday_dates = df[df['timestamp'].dt.weekday < 5]['timestamp'].dt.date.unique()

forecast_start_idx = None
for date in reversed(weekday_dates):
    same_day = df[df['timestamp'].dt.date == date]
    after_11 = same_day[same_day['timestamp'].dt.hour >= 11]
    before_18 = same_day[same_day['timestamp'].dt.hour < 18]
    
    if len(after_11) > 0 and len(before_18[before_18['timestamp'].dt.hour >= 17]) > 0:
        forecast_start_idx = after_11.index[0]
        forecast_start_time = df.iloc[forecast_start_idx]['timestamp']
        print(f"Last complete weekday found: {date}")
        break

if forecast_start_idx is None:
    raise ValueError("No suitable weekday found with data from 11:00 to 18:00")

print(f"Forecast start: {forecast_start_time}")
print(f"Forecast end: 18:00 (28 steps = 7 hours)")

df_nf = df.rename(columns={
    'timestamp': 'ds',
    'power_kw': 'y'
})
df_nf['unique_id'] = 'office_tower'
df_nf = df_nf[['unique_id', 'ds', 'y']]

df_train = df_nf.iloc[:forecast_start_idx].copy()

nf = NeuralForecast.load(path='model/')

print("\nGenerating DLinear forecast...")
forecast_df = nf.predict(df=df_train)
forecast_df = forecast_df.reset_index()
print(f"DLinear forecast shape: {forecast_df.shape}")

print("\nGenerating TimesFM forecast...")
torch.set_float32_matmul_precision("high")

tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

tfm.compile(
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

historical_values = df_train['y'].values[-672:]
point_forecast, quantile_forecast = tfm.forecast(
    horizon=28,
    inputs=[historical_values]
)
forecast_timesfm_values = point_forecast[0, :]
forecast_timesfm_q10 = quantile_forecast[0, :, 1]
forecast_timesfm_q90 = quantile_forecast[0, :, 9]

seven_days_ago = forecast_start_time - pd.Timedelta(days=7)
historical_data = df[(df['timestamp'] >= seven_days_ago) & (df['timestamp'] < forecast_start_time)].copy()

forecast_timestamps = pd.date_range(
    start=forecast_start_time,
    periods=28,
    freq='15min'
)

dlinear_std = df_train['y'].std() * 0.15

forecast_plot_df = pd.DataFrame({
    'timestamp': forecast_timestamps,
    'DLinear': forecast_df['DLinear'].values,
    'DLinear_lower': forecast_df['DLinear'].values - 1.645 * dlinear_std,
    'DLinear_upper': forecast_df['DLinear'].values + 1.645 * dlinear_std,
    'TimesFM': forecast_timesfm_values,
    'TimesFM_lower': forecast_timesfm_q10,
    'TimesFM_upper': forecast_timesfm_q90
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

ax1.plot(historical_data['timestamp'], 
         historical_data['power_kw'], 
         label='Historical (Past 7 days)', 
         color='blue', 
         linewidth=1.5,
         alpha=0.7)

ax1.plot(forecast_plot_df['timestamp'], 
         forecast_plot_df['DLinear'], 
         label='DLinear Forecast', 
         color='red', 
         linewidth=2,
         linestyle='--',
         marker='o',
         markersize=3)

ax1.fill_between(forecast_plot_df['timestamp'],
                  forecast_plot_df['DLinear_lower'],
                  forecast_plot_df['DLinear_upper'],
                  color='red',
                  alpha=0.2,
                  label='90% Confidence Interval')

ax1.axvline(x=forecast_start_time, 
            color='green', 
            linestyle=':', 
            linewidth=2, 
            label='Forecast Start')

ax1.set_xlabel('Timestamp', fontsize=11)
ax1.set_ylabel('Power (kW)', fontsize=11)
ax1.set_title('DLinear Model Forecast with Confidence Interval', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

ax2.plot(historical_data['timestamp'], 
         historical_data['power_kw'], 
         label='Historical (Past 7 days)', 
         color='blue', 
         linewidth=1.5,
         alpha=0.7)

ax2.plot(forecast_plot_df['timestamp'], 
         forecast_plot_df['TimesFM'], 
         label='TimesFM Forecast', 
         color='purple', 
         linewidth=2,
         linestyle='--',
         marker='s',
         markersize=3)

ax2.fill_between(forecast_plot_df['timestamp'],
                  forecast_plot_df['TimesFM_lower'],
                  forecast_plot_df['TimesFM_upper'],
                  color='purple',
                  alpha=0.2,
                  label='10th-90th Percentile')

ax2.axvline(x=forecast_start_time, 
            color='green', 
            linestyle=':', 
            linewidth=2, 
            label='Forecast Start')

ax2.set_xlabel('Timestamp', fontsize=11)
ax2.set_ylabel('Power (kW)', fontsize=11)
ax2.set_title('TimesFM Model Forecast with Quantile Prediction', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9, loc='best')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()

output_path = 'output/forecast_visualization.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")
plt.show()

forecast_csv_path = 'output/forecast_results.csv'
forecast_plot_df.to_csv(forecast_csv_path, index=False)
print(f"Forecast data saved to: {forecast_csv_path}")

print("\n=== Forecast Summary ===")
print(f"Forecast period: {forecast_plot_df['timestamp'].min()} to {forecast_plot_df['timestamp'].max()}")
print(f"\nDLinear:")
print(f"  Point forecast range: {forecast_plot_df['DLinear'].min():.2f} - {forecast_plot_df['DLinear'].max():.2f} kW")
print(f"  Average predicted power: {forecast_plot_df['DLinear'].mean():.2f} kW")
print(f"  90% CI width: {(forecast_plot_df['DLinear_upper'] - forecast_plot_df['DLinear_lower']).mean():.2f} kW")
print(f"\nTimesFM:")
print(f"  Point forecast range: {forecast_plot_df['TimesFM'].min():.2f} - {forecast_plot_df['TimesFM'].max():.2f} kW")
print(f"  Average predicted power: {forecast_plot_df['TimesFM'].mean():.2f} kW")
print(f"  10th-90th percentile width: {(forecast_plot_df['TimesFM_upper'] - forecast_plot_df['TimesFM_lower']).mean():.2f} kW")
print(f"\nMean Absolute Difference: {abs(forecast_plot_df['DLinear'] - forecast_plot_df['TimesFM']).mean():.2f} kW")
