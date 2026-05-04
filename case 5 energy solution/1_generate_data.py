"""
Generate energy consumption data for 3-floor office building
- 3 floors, each with 3 categories: Computer Appliance, Air Con, Lobby/Corridor Lighting
- 15-minute interval data
- Date range: 2025-12-01 to 2026-02-28
- Weekday office hours (9am-7pm): configurable consumption per floor
- Other times: configurable consumption
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from config import (
    START_DATE, END_DATE, INTERVAL_MINUTES, TEST_DATA_START,
    FLOORS, ANOMALY_FLOORS,
    CATEGORY_KEYS, CATEGORY_NAMES,
    DATA_DIR, OUTPUT_DIR,
    RANDOM_SEED
)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)

NORMAL_WEEKDAY = {
    0: (10, 7, 2),  # 12am - 1am
    1: (10, 7, 2),  # 1am - 2am
    2: (10, 7, 2),  # 2am - 3am
    3: (10, 7, 2),  # 3am - 4am
    4: (10, 7, 2),  # 4am - 5am
    5: (10, 7, 2),  # 5am - 6am
    6: (10, 7, 2),  # 6am - 7am
    7: (16, 11, 2),  # 7am - 8am
    8: (33, 18, 3),  # 8am - 9am
    9: (50, 25, 5),  # 9am - 10am
    10: (50, 25, 5),  # 10am - 11am
    11: (50, 25, 5),  # 11am - 12pm
    12: (50, 25, 5),  # 12pm - 1pm
    13: (50, 25, 5),  # 1pm - 2pm
    14: (50, 25, 5),  # 2pm - 3pm
    15: (50, 25, 5),  # 3pm - 4pm
    16: (50, 25, 5),  # 4pm - 5pm
    17: (50, 25, 5),  # 5pm - 6pm
    18: (33, 18, 4),  # 6pm - 7pm
    19: (16, 11, 3),  # 7pm - 8pm
    20: (10, 7, 2),  # 8pm - 9pm
    21: (10, 7, 2),  # 9pm - 10pm
    22: (10, 7, 2),  # 10pm - 11pm
    23: (10, 7, 2),  # 11pm - 12am
}

NORMAL_WEEKEND = {
    0: (10, 7, 2),  # 12am - 1am
    1: (10, 7, 2),  # 1am - 2am
    2: (10, 7, 2),  # 2am - 3am
    3: (10, 7, 2),  # 3am - 4am
    4: (10, 7, 2),  # 4am - 5am
    5: (10, 7, 2),  # 5am - 6am
    6: (10, 7, 2),  # 6am - 7am
    7: (10, 7, 2),  # 7am - 8am
    8: (10, 7, 2),  # 8am - 9am
    9: (10, 7, 2),  # 9am - 10am
    10: (10, 7, 2),  # 10am - 11am
    11: (10, 7, 2),  # 11am - 12pm
    12: (10, 7, 2),  # 12pm - 1pm
    13: (10, 7, 2),  # 1pm - 2pm
    14: (10, 7, 2),  # 2pm - 3pm
    15: (10, 7, 2),  # 3pm - 4pm
    16: (10, 7, 2),  # 4pm - 5pm
    17: (10, 7, 2),  # 5pm - 6pm
    18: (10, 7, 2),  # 6pm - 7pm
    19: (10, 7, 2),  # 7pm - 8pm
    20: (10, 7, 2),  # 8pm - 9pm
    21: (10, 7, 2),  # 9pm - 10pm
    22: (10, 7, 2),  # 10pm - 11pm
    23: (10, 7, 2),  # 11pm - 12am
}

ABNORMAL_WEEKDAY = {
    0: (25, 15, 2),  # 12am - 1am
    1: (24, 14, 2),  # 1am - 2am
    2: (23, 14, 2),  # 2am - 3am
    3: (22, 13, 2),  # 3am - 4am
    4: (21, 13, 2),  # 4am - 5am
    5: (20, 12, 2),  # 5am - 6am
    6: (19, 12, 2),  # 6am - 7am
    7: (25, 15, 2),  # 7am - 8am
    8: (33, 18, 3),  # 8am - 9am
    9: (50, 25, 5),  # 9am - 10am
    10: (50, 25, 5),  # 10am - 11am
    11: (50, 25, 5),  # 11am - 12pm
    12: (50, 25, 5),  # 12pm - 1pm
    13: (50, 25, 5),  # 1pm - 2pm
    14: (50, 25, 5),  # 2pm - 3pm
    15: (50, 25, 5),  # 3pm - 4pm
    16: (50, 25, 5),  # 4pm - 5pm
    17: (50, 25, 5),  # 5pm - 6pm
    18: (33, 18, 4),  # 6pm - 7pm
    19: (32, 17, 3),  # 7pm - 8pm
    20: (31, 17, 2),  # 8pm - 9pm
    21: (31, 16, 2),  # 9pm - 10pm
    22: (30, 16, 2),  # 10pm - 11pm
    23: (28, 15, 2),  # 11pm - 12am
}

timestamps = []
current = START_DATE
while current <= END_DATE:
    timestamps.append(current)
    current += timedelta(minutes=INTERVAL_MINUTES)

TOTAL_INTERVALS = len(timestamps)

print("="*60)
print("CONFIGURATION")
print("="*60)
print(f"Training data: {START_DATE.date()} to {(TEST_DATA_START - timedelta(days=1)).date()}")
print(f"Test data: {TEST_DATA_START.date()} to {END_DATE.date()}")
print(f"Interval: {INTERVAL_MINUTES} minutes")
print(f"\nData generation method: Explicit hourly targets per category")
print(f"Categories: {', '.join(CATEGORY_NAMES)}")
print(f"\nNormal weekday office hours (9am-6pm): {sum(NORMAL_WEEKDAY[9])} kWh/hour total")
print(f"Normal weekday other times: {sum(NORMAL_WEEKDAY[0])} kWh/hour total")
print(f"Abnormal weekday evening (9pm-12am): {sum(ABNORMAL_WEEKDAY[21])} kWh/hour total")
print(f"\nGenerating data for {TOTAL_INTERVALS} intervals...")
print("="*60 + "\n")

def get_category_consumption(hour, day_of_week, floor, category, timestamp):
    """
    Calculate consumption for each category using explicit hourly targets
    Values are in kWh/hour for all 3 floors combined, divided by 3 for per-floor
    
    For test data (March 2026), use ABNORMAL_WEEKDAY pattern
    Additional anomaly: Floor 2 computer appliances spike on Friday evenings (7pm-11pm)
    """
    
    is_weekend = day_of_week >= 5
    is_test_period = timestamp >= TEST_DATA_START
    is_friday = day_of_week == 4
    
    if is_weekend:
        targets = NORMAL_WEEKEND[hour]
    elif is_test_period and not is_weekend and floor in ANOMALY_FLOORS:
        targets = ABNORMAL_WEEKDAY[hour]
    else:
        targets = NORMAL_WEEKDAY[hour]
    
    per_floor_per_15min = {
        cat: (kwh / len(FLOORS)) / 4 
        for cat, kwh in zip(CATEGORY_KEYS, targets)
    }
    
    base = per_floor_per_15min[category]
    
    # Add Friday evening spike for Floor 2 computer appliances in test period
    if (is_test_period and is_friday and floor == "2" and 
        category == 'computer_appliance' and 19 <= hour <= 22):
        # Spike: 3x normal consumption during Friday evening (7pm-11pm)
        spike_multiplier = 3.0
        base = base * spike_multiplier
    
    noise = np.random.normal(0, 0.5)
    return max(0, base * (1 + noise))

data = []

for ts in timestamps:
    hour = ts.hour
    day_of_week = ts.weekday()
    
    for floor in FLOORS:
        for category in CATEGORY_KEYS:
            consumption = get_category_consumption(hour, day_of_week, floor, category, ts)
            
            data.append({
                'timestamp': ts,
                'floor': floor,
                'category': category,
                'consumption_kwh': consumption,
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': 1 if day_of_week >= 5 else 0,
                'is_office_hours': 1 if (9 <= hour < 19 and day_of_week < 5) else 0
            })

df = pd.DataFrame(data)

# Split into train and test
df_train = df[df['timestamp'] < TEST_DATA_START].copy()
df_test = df[df['timestamp'] >= TEST_DATA_START].copy()

# Save in long format (one row per category)
df_train.to_csv(f'{DATA_DIR}/energy_consumption_train.csv', index=False)
df_test.to_csv(f'{DATA_DIR}/energy_consumption_test.csv', index=False)
df.to_csv(f'{DATA_DIR}/energy_consumption_all.csv', index=False)

# Create pivot for visualization
df_pivot = df.pivot_table(
    index=['timestamp', 'hour', 'day_of_week', 'is_weekend', 'is_office_hours', 'floor'],
    columns='category',
    values='consumption_kwh'
).reset_index()

df_pivot['total'] = (df_pivot['computer_appliance'] + 
                     df_pivot['air_con'] + 
                     df_pivot['lobby_corridor_lighting'])

df_train_pivot = df_pivot[df_pivot['timestamp'] < TEST_DATA_START].copy()
df_test_pivot = df_pivot[df_pivot['timestamp'] >= TEST_DATA_START].copy()

print(f"✓ Generated {len(df)} total records")
print(f"  Training: {len(df_train)} records ({df_train['timestamp'].min()} to {df_train['timestamp'].max()})")
print(f"  Test: {len(df_test)} records ({df_test['timestamp'].min()} to {df_test['timestamp'].max()})")
print(f"  Floors: {sorted(df_pivot['floor'].unique())}")

fig_train, axes_train = plt.subplots(2, 2, figsize=(16, 10))
fig_train.suptitle('Training Data Overview (Dec 2025 - Feb 2026)', fontsize=16, fontweight='bold')

ax1 = axes_train[0, 0]
df_daily_train = df_train_pivot.groupby([df_train_pivot['timestamp'].dt.date, 'floor'])['total'].sum().reset_index()
df_daily_train.columns = ['date', 'floor', 'total_kwh']
for floor in FLOORS:
    floor_data = df_daily_train[df_daily_train['floor'] == floor]
    ax1.plot(floor_data['date'], floor_data['total_kwh'], label=f'Floor {floor}', alpha=0.7)
ax1.set_title('Daily Energy Consumption by Floor')
ax1.set_xlabel('Date')
ax1.set_ylabel('kWh per day')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

ax2 = axes_train[0, 1]
df_train_weekday = df_train[df_train['day_of_week'] < 5]
df_hourly_train = df_train_weekday.groupby(['hour', 'category'])['consumption_kwh'].mean().reset_index()
df_hourly_train_pivot = df_hourly_train.pivot(index='hour', columns='category', values='consumption_kwh')
df_hourly_train_pivot = df_hourly_train_pivot * 4 * len(FLOORS)
df_hourly_train_pivot.plot(kind='bar', stacked=True, ax=ax2, color=['#3498db', '#2ecc71', '#f39c12'], width=0.8)
ax2.set_title('Hourly Consumption - Weekdays')
ax2.set_xlabel('Hour of Day')
ax2.set_ylabel('kWh per hour')
ax2.legend(title='Category', labels=['Air Con', 'Computer Appliance', 'Lobby/Corridor Lighting'])
ax2.grid(True, alpha=0.3, axis='y')
for i in range(24):
    if 9 <= i < 19:
        ax2.axvspan(i-0.5, i+0.5, alpha=0.1, color='green')
ax2.set_xticklabels(range(24), rotation=0)

ax3 = axes_train[1, 0]
df_category_train = df_train.groupby('category')['consumption_kwh'].sum()
category_totals = [df_category_train[cat] for cat in CATEGORY_KEYS]
pd.DataFrame({'Total kWh': category_totals}, index=CATEGORY_NAMES).plot(kind='bar', ax=ax3, color=['#3498db', '#2ecc71', '#f39c12'], legend=False)
ax3.set_title('Total Consumption by Category')
ax3.set_ylabel('Total kWh')
ax3.set_xlabel('Category')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3, axis='y')

ax4 = axes_train[1, 1]
df_heatmap_train = df_train_pivot.groupby(['day_of_week', 'hour'])['total'].mean().reset_index()
heatmap_train_pivot = df_heatmap_train.pivot(index='day_of_week', columns='hour', values='total')
sns.heatmap(heatmap_train_pivot * 4, ax=ax4, cmap='YlOrRd', cbar_kws={'label': 'kWh/hour'})
ax4.set_title('Average Building Consumption Heatmap')
ax4.set_xlabel('Hour of Day')
ax4.set_ylabel('Day of Week (0=Mon, 6=Sun)')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/1_train_overview.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved training visualization: {OUTPUT_DIR}/1_train_overview.png")
plt.show()

fig_test, axes_test = plt.subplots(2, 2, figsize=(16, 10))
fig_test.suptitle('Test Data Overview - With Anomaly (March 2026)', fontsize=16, fontweight='bold')

ax_t1 = axes_test[0, 0]
df_daily_test = df_test_pivot.groupby([df_test_pivot['timestamp'].dt.date, 'floor'])['total'].sum().reset_index()
df_daily_test.columns = ['date', 'floor', 'total_kwh']
for floor in FLOORS:
    floor_data = df_daily_test[df_daily_test['floor'] == floor]
    ax_t1.plot(floor_data['date'], floor_data['total_kwh'], label=f'Floor {floor}', alpha=0.7, marker='o')
ax_t1.set_title('Daily Energy Consumption by Floor')
ax_t1.set_xlabel('Date')
ax_t1.set_ylabel('kWh per day')
ax_t1.legend()
ax_t1.grid(True, alpha=0.3)
ax_t1.tick_params(axis='x', rotation=45)

ax_t2 = axes_test[0, 1]
df_test_weekday = df_test[df_test['day_of_week'] < 5]
df_hourly_test = df_test_weekday.groupby(['hour', 'category'])['consumption_kwh'].mean().reset_index()
df_hourly_test_pivot = df_hourly_test.pivot(index='hour', columns='category', values='consumption_kwh')
df_hourly_test_pivot = df_hourly_test_pivot * 4 * len(FLOORS)
df_hourly_test_pivot.plot(kind='bar', stacked=True, ax=ax_t2, color=['#3498db', '#2ecc71', '#f39c12'], width=0.8)
ax_t2.set_title('Hourly Consumption - Weekdays')
ax_t2.set_xlabel('Hour of Day')
ax_t2.set_ylabel('kWh per hour')
ax_t2.legend(title='Category', labels=['Air Con', 'Computer Appliance', 'Lobby/Corridor Lighting'])
ax_t2.grid(True, alpha=0.3, axis='y')
for i in range(24):
    if 21 <= i < 24:
        ax_t2.axvspan(i-0.5, i+0.5, alpha=0.2, color='red', label='Anomaly' if i == 21 else '')
ax_t2.set_xticklabels(range(24), rotation=0)

ax_t3 = axes_test[1, 0]
df_test_evening = df_test_pivot[(df_test_pivot['hour'] >= 21) & (df_test_pivot['hour'] < 24) & (df_test_pivot['day_of_week'] < 5)]
df_train_evening = df_train_pivot[(df_train_pivot['hour'] >= 21) & (df_train_pivot['hour'] < 24) & (df_train_pivot['day_of_week'] < 5)]
comparison_data = pd.DataFrame({
    'Training': [df_train_evening['air_con'].mean() * 4 * len(FLOORS),
                 df_train_evening['computer_appliance'].mean() * 4 * len(FLOORS),
                 df_train_evening['lobby_corridor_lighting'].mean() * 4 * len(FLOORS)],
    'Test (Anomaly)': [df_test_evening['air_con'].mean() * 4 * len(FLOORS),
                       df_test_evening['computer_appliance'].mean() * 4 * len(FLOORS),
                       df_test_evening['lobby_corridor_lighting'].mean() * 4 * len(FLOORS)]
}, index=['Air Con', 'Computer\nAppliance', 'Lobby/Corridor\nLighting'])
comparison_data.plot(kind='bar', ax=ax_t3, color=['#3498db', '#e74c3c'], width=0.7)
ax_t3.set_title('Evening Consumption Comparison (9pm-12am, Weekdays)')
ax_t3.set_ylabel('kWh per hour')
ax_t3.set_xlabel('Category')
ax_t3.legend(title='Period')
ax_t3.tick_params(axis='x', rotation=0)
ax_t3.grid(True, alpha=0.3, axis='y')

ax_t4 = axes_test[1, 1]
df_heatmap_test = df_test_pivot.groupby(['day_of_week', 'hour'])['total'].mean().reset_index()
heatmap_test_pivot = df_heatmap_test.pivot(index='day_of_week', columns='hour', values='total')
sns.heatmap(heatmap_test_pivot * 4, ax=ax_t4, cmap='YlOrRd', cbar_kws={'label': 'kWh/hour'})
ax_t4.set_title('Average Building Consumption Heatmap')
ax_t4.set_xlabel('Hour of Day')
ax_t4.set_ylabel('Day of Week (0=Mon, 6=Sun)')
for i in range(21, 24):
    ax_t4.axvline(x=i+0.5, color='red', linewidth=2, alpha=0.5)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/1_test_overview.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved test visualization: {OUTPUT_DIR}/1_test_overview.png")
plt.show()

print("\n" + "="*60)
print("DATA SUMMARY")
print("="*60)

total_by_floor = df_pivot.groupby('floor')['total'].sum()
print("\nTotal consumption by floor:")
for floor, kwh in total_by_floor.items():
    print(f"  Floor {floor}: {kwh:,.1f} kWh")

total_by_category = df.groupby('category')['consumption_kwh'].sum()
print("\nTotal consumption by category:")
for category, kwh in total_by_category.items():
    print(f"  {category}: {kwh:,.1f} kWh")

office_hours_avg = df_pivot[df_pivot['is_office_hours'] == 1]['total'].mean() * 4
other_times_avg = df_pivot[df_pivot['is_office_hours'] == 0]['total'].mean() * 4
print(f"\nAverage consumption rates:")
print(f"  Weekday office hours (9am-7pm): {office_hours_avg:.1f} kWh/hour")
print(f"  Other times: {other_times_avg:.1f} kWh/hour")

print(f"\nTotal building consumption: {total_by_floor.sum():,.1f} kWh")
days = (END_DATE - START_DATE).days + 1
print(f"  Period: {days} days ({START_DATE.date()} to {END_DATE.date()})")
print(f"  Average daily: {total_by_floor.sum() / days:,.1f} kWh/day")

df_test = df_pivot[df_pivot['timestamp'] >= TEST_DATA_START]
df_train = df_pivot[df_pivot['timestamp'] < TEST_DATA_START]

print("\n" + "="*60)
print("TRAINING vs TEST DATA COMPARISON")
print("="*60)
print(f"Training period: {len(df_train)} records ({START_DATE.date()} to {(TEST_DATA_START - timedelta(days=1)).date()})")
print(f"Test period: {len(df_test)} records ({TEST_DATA_START.date()} to {END_DATE.date()})")

test_evening = df_test[(df_test['hour'] >= 19) & (df_test['hour'] < 24) & (df_test['day_of_week'] < 5)]
train_evening = df_train[(df_train['hour'] >= 19) & (df_train['hour'] < 24) & (df_train['day_of_week'] < 5)]

if len(test_evening) > 0 and len(train_evening) > 0:
    print(f"\nWeekday evening (7pm-12am) average consumption:")
    print(f"  Training: {train_evening['total'].mean() * 4:.1f} kWh/hour")
    print(f"  Test (with anomaly): {test_evening['total'].mean() * 4:.1f} kWh/hour")
    print(f"  Increase: {((test_evening['total'].mean() / train_evening['total'].mean()) - 1) * 100:.1f}%")

# ============================================================================
# Floor 2 Computer Appliance Analysis (Test Period)
# ============================================================================
print("\nGenerating Floor 2 computer appliance analysis...")

# Get Floor 2 computer data from test period
df_floor2_computer_test = df[
    (df['timestamp'] >= TEST_DATA_START) & 
    (df['floor'] == "2") & 
    (df['category'] == 'computer_appliance')
].copy()

# Add hour and day of week
df_floor2_computer_test['hour'] = df_floor2_computer_test['timestamp'].dt.hour
df_floor2_computer_test['day_of_week'] = df_floor2_computer_test['timestamp'].dt.dayofweek
df_floor2_computer_test['day_name'] = df_floor2_computer_test['timestamp'].dt.day_name()

# Resample to hourly and calculate average by day of week and hour
hourly_avg = df_floor2_computer_test.groupby(['day_of_week', 'day_name', 'hour']).agg({
    'consumption_kwh': 'sum'
}).reset_index()

# Convert to kWh/hour (sum of 4x 15-min intervals)
hourly_avg['kwh_per_hour'] = hourly_avg['consumption_kwh']

# Create visualization
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle('Floor 2 - Computer Appliances (Test Period: March 2026)', fontsize=16, fontweight='bold')

# Plot 1: Heatmap by day of week and hour
ax1 = axes[0]
pivot_data = hourly_avg.pivot(index='day_name', columns='hour', values='kwh_per_hour')
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pivot_data = pivot_data.reindex(day_order)

sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax1, 
            cbar_kws={'label': 'kWh/hour'}, linewidths=0.5, linecolor='gray')
ax1.set_title('Average Hourly Consumption by Day of Week', fontsize=12, fontweight='bold')
ax1.set_xlabel('Hour of Day', fontsize=10)
ax1.set_ylabel('Day of Week', fontsize=10)

# Highlight Friday evening spike area
from matplotlib.patches import Rectangle
# Friday is row 4 (0-indexed), hours 19-22
rect = Rectangle((19, 4), 4, 1, linewidth=3, edgecolor='blue', facecolor='none', linestyle='--')
ax1.add_patch(rect)
ax1.text(21, 4.5, 'Friday Spike', ha='center', va='center', 
         fontsize=10, fontweight='bold', color='blue', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='blue'))

# Plot 2: Line chart comparing weekdays
ax2 = axes[1]
weekday_data = hourly_avg[hourly_avg['day_of_week'] < 5].copy()

for day_num in range(5):
    day_data = weekday_data[weekday_data['day_of_week'] == day_num]
    day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][day_num]
    
    # Use different style for Friday
    if day_num == 4:
        ax2.plot(day_data['hour'], day_data['kwh_per_hour'], 
                marker='o', linewidth=3, markersize=6, label=day_name, color='red')
    else:
        ax2.plot(day_data['hour'], day_data['kwh_per_hour'], 
                marker='o', linewidth=1.5, markersize=4, label=day_name, alpha=0.6)

ax2.set_xlabel('Hour of Day', fontsize=10)
ax2.set_ylabel('Consumption (kWh/hour)', fontsize=10)
ax2.set_title('Hourly Consumption Pattern by Weekday (Friday Highlighted)', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(0, 24, 2))

# Highlight Friday evening spike window
ax2.axvspan(19, 23, alpha=0.2, color='red', label='Friday Spike Window')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/1_floor2_computer_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved Floor 2 computer analysis: {OUTPUT_DIR}/1_floor2_computer_analysis.png")
plt.show()

print("\n✓ Data generation complete!")
