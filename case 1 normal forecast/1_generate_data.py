import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

np.random.seed(42)

start_date = datetime(2025, 4, 15)
end_date = start_date + timedelta(days=365)
date_range = pd.date_range(start=start_date, end=end_date, freq='15min')

data = []

for timestamp in date_range:
    hour = timestamp.hour
    minute = timestamp.minute
    is_weekday = timestamp.weekday() < 5
    
    base_load = 100
    
    if is_weekday:
        if 9 <= hour < 18:
            time_factor = 1.0 - abs(hour - 13.5) / 9
            office_load = 300 + 150 * time_factor
        elif 7 <= hour < 9:
            ramp_up = (hour - 7 + minute / 60) / 2
            office_load = 100 + 200 * ramp_up
        elif 18 <= hour < 20:
            ramp_down = 1 - (hour - 18 + minute / 60) / 2
            office_load = 100 + 200 * ramp_down
        else:
            office_load = 50
    else:
        if 10 <= hour < 16:
            office_load = 80 + 40 * np.sin((hour - 10) * np.pi / 6)
        else:
            office_load = 50
    
    seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * timestamp.dayofyear / 365)
    
    noise = np.random.normal(0, 15)
    
    power = base_load + office_load * seasonal_factor + noise
    power = max(50, power)
    
    data.append({
        'timestamp': timestamp,
        'power_kw': round(power, 2)
    })

df = pd.DataFrame(data)

df.to_csv('data/office_power_data.csv', index=False)

print(f"Generated {len(df)} records")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Power range: {df['power_kw'].min():.2f} to {df['power_kw'].max():.2f} kW")
print(f"\nData saved to: data/office_power_data.csv")

weekday_office_hours = df[(pd.to_datetime(df['timestamp']).dt.weekday < 5) & 
                           (pd.to_datetime(df['timestamp']).dt.hour >= 9) & 
                           (pd.to_datetime(df['timestamp']).dt.hour < 18)]
weekend_data = df[pd.to_datetime(df['timestamp']).dt.weekday >= 5]

print(f"\nWeekday office hours (9-18) avg: {weekday_office_hours['power_kw'].mean():.2f} kW")
print(f"Weekend avg: {weekend_data['power_kw'].mean():.2f} kW")

sample_week = df[(df['timestamp'] >= '2025-04-14') & (df['timestamp'] < '2025-04-21')]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

ax1.plot(sample_week['timestamp'], sample_week['power_kw'], linewidth=1)
ax1.set_xlabel('Timestamp', fontsize=11)
ax1.set_ylabel('Power (kW)', fontsize=11)
ax1.set_title('Sample Week - Power Consumption Pattern', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

for day in range(14, 21):
    day_start = pd.Timestamp(f'2025-04-{day} 00:00:00')
    if day_start in sample_week['timestamp'].values:
        weekday_name = day_start.strftime('%A')
        ax1.axvline(x=day_start, color='gray', linestyle='--', alpha=0.5)
        ax1.text(day_start, ax1.get_ylim()[1], weekday_name, rotation=0, ha='left', fontsize=9)

weekday_sample = sample_week[sample_week['timestamp'].dt.weekday < 5]
weekend_sample = sample_week[sample_week['timestamp'].dt.weekday >= 5]

ax2.plot(weekday_sample['timestamp'].dt.hour + weekday_sample['timestamp'].dt.minute/60, 
         weekday_sample['power_kw'], 'o', alpha=0.3, markersize=2, label='Weekday', color='blue')
ax2.plot(weekend_sample['timestamp'].dt.hour + weekend_sample['timestamp'].dt.minute/60, 
         weekend_sample['power_kw'], 'o', alpha=0.3, markersize=2, label='Weekend', color='orange')
ax2.set_xlabel('Hour of Day', fontsize=11)
ax2.set_ylabel('Power (kW)', fontsize=11)
ax2.set_title('Weekday vs Weekend Pattern', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 24)

plt.tight_layout()
plt.savefig('output/data_generation_overview.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to: output/data_generation_overview.png")
plt.show()
