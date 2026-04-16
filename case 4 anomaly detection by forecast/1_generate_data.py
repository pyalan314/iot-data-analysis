import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

np.random.seed(42)

start_date = datetime(2024, 3, 1, 0, 0, 0)
end_date = start_date + timedelta(days=30)
freq = '15min'

date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

base_temp = 22.0
daily_amplitude = 2.0
hourly_pattern = np.sin(2 * np.pi * np.arange(len(date_range)) / (24 * 4)) * daily_amplitude

noise = np.random.normal(0, 0.3, len(date_range))

temperature = base_temp + hourly_pattern + noise

anomaly1_start_idx = int(len(date_range) * 0.35)
anomaly1_length = 24
anomaly1_end_idx = min(anomaly1_start_idx + anomaly1_length, len(date_range))
anomaly1_magnitude = np.linspace(0, 3, anomaly1_end_idx - anomaly1_start_idx)
temperature[anomaly1_start_idx:anomaly1_end_idx] += anomaly1_magnitude

anomaly2_start_idx = int(len(date_range) * 0.60)
anomaly2_length = 36
anomaly2_end_idx = min(anomaly2_start_idx + anomaly2_length, len(date_range))
anomaly2_magnitude = np.linspace(0, 5, anomaly2_end_idx - anomaly2_start_idx)
temperature[anomaly2_start_idx:anomaly2_end_idx] += anomaly2_magnitude

anomaly3_start_idx = int(len(date_range) * 0.85)
anomaly3_length = 48
anomaly3_end_idx = min(anomaly3_start_idx + anomaly3_length, len(date_range))
anomaly3_magnitude = np.linspace(0, 8, anomaly3_end_idx - anomaly3_start_idx)
temperature[anomaly3_start_idx:anomaly3_end_idx] += anomaly3_magnitude

df = pd.DataFrame({
    'timestamp': date_range,
    'temperature': temperature
})

os.makedirs('data', exist_ok=True)
os.makedirs('output', exist_ok=True)

df.to_csv('data/server_room_temperature.csv', index=False)
print(f"Generated {len(df)} temperature readings")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Temperature range: {df['temperature'].min():.2f}°C to {df['temperature'].max():.2f}°C")
print(f"\nAnomalies injected:")
print(f"  Anomaly 1 [MILD]:     index {anomaly1_start_idx} ({df['timestamp'].iloc[anomaly1_start_idx]}) - +3°C surge (6h)")
print(f"  Anomaly 2 [MODERATE]: index {anomaly2_start_idx} ({df['timestamp'].iloc[anomaly2_start_idx]}) - +5°C surge (9h)")
print(f"  Anomaly 3 [SEVERE]:   index {anomaly3_start_idx} ({df['timestamp'].iloc[anomaly3_start_idx]}) - +8°C surge (12h)")

plt.figure(figsize=(15, 6))
plt.plot(df['timestamp'], df['temperature'], linewidth=0.8, alpha=0.8)
plt.axvline(x=df['timestamp'].iloc[anomaly1_start_idx], color='yellow', linestyle='--', 
            label='Anomaly 1 [MILD]', alpha=0.7, linewidth=2)
plt.axvline(x=df['timestamp'].iloc[anomaly2_start_idx], color='orange', linestyle='--', 
            label='Anomaly 2 [MODERATE]', alpha=0.7, linewidth=2)
plt.axvline(x=df['timestamp'].iloc[anomaly3_start_idx], color='red', linestyle='--', 
            label='Anomaly 3 [SEVERE]', alpha=0.7, linewidth=2)
plt.xlabel('Timestamp')
plt.ylabel('Temperature (°C)')
plt.title('Server Room Temperature - 1 Month (15-min intervals)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('output/1_generated_data.png', dpi=150)
plt.show()

print("\nData saved to: data/server_room_temperature.csv")
print("Visualization saved to: output/1_generated_data.png")
