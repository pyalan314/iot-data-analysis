import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

np.random.seed(42)

def create_directories():
    os.makedirs('data', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('model', exist_ok=True)

def generate_office_energy_data():
    start_date = datetime(2024, 1, 1, 0, 0)
    end_date = datetime(2024, 3, 31, 23, 45)
    
    timestamps = []
    current = start_date
    while current <= end_date:
        timestamps.append(current)
        current += timedelta(minutes=15)
    
    data = []
    
    for ts in timestamps:
        hour = ts.hour
        is_weekday = ts.weekday() < 5
        is_office_hours = 9 <= hour < 18
        
        if is_weekday and is_office_hours:
            base_usage = 80
            noise = np.random.normal(0, 5)
        elif is_weekday and not is_office_hours:
            base_usage = 10
            noise = np.random.normal(0, 2)
        else:
            base_usage = 10
            noise = np.random.normal(0, 2)
        
        usage = max(0, base_usage + noise)
        
        data.append({
            'timestamp': ts,
            'energy_usage': usage,
            'is_anomaly': 0,
            'anomaly_type': 'normal'
        })
    
    df = pd.DataFrame(data)
    
    tuesday_night = datetime(2024, 3, 26, 23, 0)
    tuesday_night_end = datetime(2024, 3, 27, 0, 0)
    mask1 = (df['timestamp'] >= tuesday_night) & (df['timestamp'] < tuesday_night_end)
    df.loc[mask1, 'energy_usage'] = 50 + np.random.normal(0, 3, mask1.sum())
    df.loc[mask1, 'is_anomaly'] = 1
    df.loc[mask1, 'anomaly_type'] = 'tuesday_night'
    
    friday_afternoon = datetime(2024, 3, 29, 14, 0)
    friday_afternoon_end = datetime(2024, 3, 29, 16, 0)
    mask2 = (df['timestamp'] >= friday_afternoon) & (df['timestamp'] < friday_afternoon_end)
    df.loc[mask2, 'energy_usage'] = 100 + np.random.normal(0, 5, mask2.sum())
    df.loc[mask2, 'is_anomaly'] = 1
    df.loc[mask2, 'anomaly_type'] = 'friday_afternoon'
    
    saturday_evening = datetime(2024, 3, 30, 19, 0)
    saturday_evening_end = datetime(2024, 3, 30, 22, 0)
    mask3 = (df['timestamp'] >= saturday_evening) & (df['timestamp'] < saturday_evening_end)
    df.loc[mask3, 'energy_usage'] = 30 + np.random.normal(0, 2, mask3.sum())
    df.loc[mask3, 'is_anomaly'] = 1
    df.loc[mask3, 'anomaly_type'] = 'saturday_evening'
    
    return df

def visualize_data(df):
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    axes[0].plot(df['timestamp'], df['energy_usage'], linewidth=0.5, alpha=0.7)
    anomaly_data = df[df['is_anomaly'] == 1]
    axes[0].scatter(anomaly_data['timestamp'], anomaly_data['energy_usage'], 
                    color='red', s=20, label='Injected Anomalies', zorder=5)
    axes[0].set_xlabel('Timestamp')
    axes[0].set_ylabel('Energy Usage (kWh)')
    axes[0].set_title('Office Energy Usage - 3 Months (Jan-Mar 2024)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    last_week = df[df['timestamp'] >= datetime(2024, 3, 25, 0, 0)]
    axes[1].plot(last_week['timestamp'], last_week['energy_usage'], linewidth=1, alpha=0.7)
    last_week_anomaly = last_week[last_week['is_anomaly'] == 1]
    axes[1].scatter(last_week_anomaly['timestamp'], last_week_anomaly['energy_usage'], 
                    color='red', s=30, label='Injected Anomalies', zorder=5)
    axes[1].set_xlabel('Timestamp')
    axes[1].set_ylabel('Energy Usage (kWh)')
    axes[1].set_title('Office Energy Usage - Last Week (Detection Period)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    df['hour'] = df['timestamp'].dt.hour
    df['is_weekday'] = df['timestamp'].dt.weekday < 5
    
    weekday_data = df[df['is_weekday'] == True]
    weekend_data = df[df['is_weekday'] == False]
    
    hourly_weekday = weekday_data.groupby('hour')['energy_usage'].mean()
    hourly_weekend = weekend_data.groupby('hour')['energy_usage'].mean()
    
    axes[2].plot(hourly_weekday.index, hourly_weekday.values, marker='o', label='Weekday Average', linewidth=2)
    axes[2].plot(hourly_weekend.index, hourly_weekend.values, marker='s', label='Weekend Average', linewidth=2)
    axes[2].axvspan(9, 18, alpha=0.2, color='yellow', label='Office Hours')
    axes[2].set_xlabel('Hour of Day')
    axes[2].set_ylabel('Average Energy Usage (kWh)')
    axes[2].set_title('Average Energy Usage by Hour')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    plt.savefig('output/1_generated_data.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to output/1_generated_data.png")
    plt.show()
    
    print("\n=== Data Statistics ===")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Injected anomalies: {df['is_anomaly'].sum()}")
    print(f"\nAnomaly breakdown:")
    print(df[df['is_anomaly'] == 1]['anomaly_type'].value_counts())
    print(f"\nEnergy usage statistics:")
    print(df['energy_usage'].describe())

if __name__ == "__main__":
    create_directories()
    
    print("Generating office energy usage data...")
    df = generate_office_energy_data()
    
    df.to_csv('data/office_energy_data.csv', index=False)
    print("Data saved to data/office_energy_data.csv")
    
    visualize_data(df)
