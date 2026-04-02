import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# HK Public Holidays in 2026 (Jan 1 - Mar 31)
HK_HOLIDAYS_2026 = [
    '2026-01-01',  # New Year's Day
    '2026-01-29',  # Lunar New Year Day 1
    '2026-01-30',  # Lunar New Year Day 2
    '2026-01-31',  # Lunar New Year Day 3
    '2026-02-02',  # Lunar New Year Day 4 (in lieu)
]

def is_holiday(date):
    """Check if date is a HK public holiday"""
    date_str = date.strftime('%Y-%m-%d')
    return date_str in HK_HOLIDAYS_2026

def is_weekend(date):
    """Check if date is weekend (Saturday=5, Sunday=6)"""
    return date.weekday() >= 5

def generate_power_usage_data():
    """
    Generate synthetic power usage data with realistic patterns:
    - Time pattern: High usage during office hours (9am-6pm)
    - Weekday pattern: Higher usage on weekdays
    - Holiday pattern: Lower usage on HK public holidays
    """
    
    # Generate timestamps: Jan 1, 2026 00:00 to Mar 31, 2026 23:59 (1-minute intervals)
    start_date = datetime(2026, 1, 1, 0, 0)
    end_date = datetime(2026, 3, 31, 23, 59)
    
    timestamps = []
    current = start_date
    while current <= end_date:
        timestamps.append(current)
        current += timedelta(minutes=1)
    
    # Initialize data
    data = []
    
    for ts in timestamps:
        hour = ts.hour
        
        # Base power usage (kW)
        base_power = 20.0
        
        # Time-of-day pattern (office hours: 9am-6pm)
        if 9 <= hour < 18:
            # Peak hours: higher usage
            time_factor = 3.0 + 0.5 * np.sin((hour - 9) * np.pi / 9)
        elif 6 <= hour < 9:
            # Morning ramp-up
            time_factor = 1.0 + 2.0 * (hour - 6) / 3
        elif 18 <= hour < 21:
            # Evening ramp-down
            time_factor = 3.0 - 2.0 * (hour - 18) / 3
        else:
            # Night time: minimal usage
            time_factor = 0.5 + 0.3 * np.sin(hour * np.pi / 12)
        
        # Weekday vs weekend pattern
        if is_holiday(ts):
            # Holiday: very low usage
            weekday_factor = 0.3
        elif is_weekend(ts):
            # Weekend: lower usage
            weekday_factor = 0.5
        else:
            # Weekday: normal usage
            weekday_factor = 1.0
        
        # Calculate power usage
        power = base_power * time_factor * weekday_factor
        
        # Add some random noise (±5%)
        noise = np.random.normal(0, power * 0.05)
        power = max(0, power + noise)
        
        # Add slight trend (gradual increase over time)
        days_elapsed = (ts - start_date).days
        trend = 0.01 * days_elapsed
        power += trend
        
        data.append({
            'timestamp': ts,
            'device_id': 'DEVICE_001',
            'power_usage': round(power, 2)
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV with UTF-8 encoding
    output_file = 'power_usage_data.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"Generated {len(df)} records")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Power usage range: {df['power_usage'].min():.2f} - {df['power_usage'].max():.2f} kW")
    print(f"Data saved to: {output_file}")
    
    # Display sample statistics
    print("\nSample statistics:")
    print(df.describe())
    
    return df

if __name__ == "__main__":
    df = generate_power_usage_data()
