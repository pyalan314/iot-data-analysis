import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import timesfm
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

torch.set_float32_matmul_precision("high")

print("Loading data...")
df = pd.read_csv('data/server_room_temperature.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

anomaly_start_idx = int(len(df) * 0.85)
train_df = df.iloc[:anomaly_start_idx].copy()

print(f"Training data: {len(train_df)} samples")
print(f"Date range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")

print("\nInitializing TimesFM model...")
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

print("\nValidating model with sample data...")
temperature_values = train_df['temperature'].values

context_window = 672
forecast_horizon = 96

test_context = temperature_values[-context_window:]
print(f"Testing forecast with context length: {context_window}, horizon: {forecast_horizon}")

point_forecast, quantile_forecast = model.forecast(
    horizon=forecast_horizon,
    inputs=[test_context],
)

print(f"\nModel validation successful!")
print(f"Point forecast shape: {point_forecast.shape}")
print(f"Quantile forecast shape: {quantile_forecast.shape}")
print(f"Forecast range: {point_forecast[0].min():.2f}°C to {point_forecast[0].max():.2f}°C")

config_info = {
    'context_len': context_window,
    'horizon_len': forecast_horizon,
    'train_end_idx': anomaly_start_idx,
    'train_stats': {
        'mean': train_df['temperature'].mean(),
        'std': train_df['temperature'].std(),
        'min': train_df['temperature'].min(),
        'max': train_df['temperature'].max(),
    }
}

os.makedirs('output', exist_ok=True)

with open('output/config.pkl', 'wb') as f:
    pickle.dump(config_info, f)

print("\nConfiguration saved to: output/config.pkl")

plt.figure(figsize=(15, 8))

plt.subplot(2, 1, 1)
plt.plot(train_df['timestamp'], train_df['temperature'], linewidth=0.8)
plt.xlabel('Timestamp')
plt.ylabel('Temperature (°C)')
plt.title('Training Data - Server Room Temperature (Pre-Anomaly)')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.subplot(2, 1, 2)
test_timestamps = train_df['timestamp'].iloc[-context_window:]
plt.plot(test_timestamps, test_context, label='Context', linewidth=0.8)
plt.xlabel('Timestamp')
plt.ylabel('Temperature (°C)')
plt.title('Sample Context Window for Validation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('output/2_training_results.png', dpi=150)
plt.show()

print("Validation visualization saved to: output/2_training_results.png")
