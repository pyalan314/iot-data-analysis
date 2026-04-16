import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from datetime import timedelta

os.makedirs('output', exist_ok=True)

print("Loading trained model and scaler...")
model = joblib.load('model/isolation_forest.pkl')
scaler = joblib.load('model/scaler.pkl')
print("Model and scaler loaded successfully!")

df = pd.read_csv('data/server_room_temperature.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"\nTotal data: {len(df)} samples")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

last_7_days_start = df['timestamp'].max() - timedelta(days=7)
test_df = df[df['timestamp'] >= last_7_days_start].copy()

print(f"\nTest data (last 7 days): {len(test_df)} samples")
print(f"Test period: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

test_df['hour'] = test_df['timestamp'].dt.hour
test_df['minute'] = test_df['timestamp'].dt.minute
test_df['day_of_week'] = test_df['timestamp'].dt.dayofweek

test_df['temp_lag_1'] = test_df['temperature'].shift(1)
test_df['temp_lag_4'] = test_df['temperature'].shift(4)
test_df['temp_lag_12'] = test_df['temperature'].shift(12)
test_df['temp_lag_24'] = test_df['temperature'].shift(24)

test_df['temp_change_1'] = test_df['temperature'] - test_df['temp_lag_1']
test_df['temp_change_4'] = test_df['temperature'] - test_df['temp_lag_4']

test_df['temp_rolling_mean_6h'] = test_df['temperature'].rolling(window=24, min_periods=1).mean()
test_df['temp_rolling_std_6h'] = test_df['temperature'].rolling(window=24, min_periods=1).std()
test_df['temp_rolling_max_6h'] = test_df['temperature'].rolling(window=24, min_periods=1).max()
test_df['temp_rolling_min_6h'] = test_df['temperature'].rolling(window=24, min_periods=1).min()

features = ['temperature', 'hour', 'minute', 
            'temp_lag_1', 'temp_lag_4', 'temp_lag_12', 'temp_lag_24',
            'temp_change_1', 'temp_change_4',
            'temp_rolling_mean_6h', 'temp_rolling_std_6h',
            'temp_rolling_max_6h', 'temp_rolling_min_6h']
X_test = test_df[features].fillna(method='bfill')

X_test_scaled = scaler.transform(X_test)

print("\nDetecting anomalies...")
predictions = model.predict(X_test_scaled)
anomaly_scores = model.score_samples(X_test_scaled)

test_df['anomaly_score'] = anomaly_scores
test_df['is_anomaly'] = predictions == -1

anomalies_only = test_df[test_df['is_anomaly']].copy()
if len(anomalies_only) > 0:
    anomaly_score_min = anomalies_only['anomaly_score'].min()
    anomaly_score_max = anomalies_only['anomaly_score'].max()
    anomaly_score_range = anomaly_score_max - anomaly_score_min
    
    threshold_mild = anomaly_score_min + anomaly_score_range * 0.33
    threshold_moderate = anomaly_score_min + anomaly_score_range * 0.67
    
    def classify_severity(row):
        if not row['is_anomaly']:
            return 'Normal'
        score = row['anomaly_score']
        if score >= threshold_moderate:
            return 'Mild'
        elif score >= threshold_mild:
            return 'Moderate'
        else:
            return 'Severe'
    
    test_df['severity'] = test_df.apply(classify_severity, axis=1)
else:
    test_df['severity'] = 'Normal'

anomaly_df = test_df[test_df['is_anomaly']].copy()
normal_df = test_df[~test_df['is_anomaly']].copy()
mild_df = test_df[test_df['severity'] == 'Mild'].copy()
moderate_df = test_df[test_df['severity'] == 'Moderate'].copy()
severe_df = test_df[test_df['severity'] == 'Severe'].copy()

print(f"\nDetection Results:")
print(f"Total samples analyzed: {len(test_df)}")
print(f"Normal samples: {len(normal_df)} ({len(normal_df)/len(test_df)*100:.2f}%)")
print(f"Anomalies detected: {len(anomaly_df)} ({len(anomaly_df)/len(test_df)*100:.2f}%)")
if len(anomaly_df) > 0:
    print(f"  - Mild Anomaly: {len(mild_df)} ({len(mild_df)/len(test_df)*100:.2f}%)")
    print(f"  - Moderate Anomaly: {len(moderate_df)} ({len(moderate_df)/len(test_df)*100:.2f}%)")
    print(f"  - Severe Anomaly: {len(severe_df)} ({len(severe_df)/len(test_df)*100:.2f}%)")

if len(anomaly_df) > 0:
    print(f"\nAnomaly Details:")
    print(f"First anomaly: {anomaly_df['timestamp'].min()}")
    print(f"Last anomaly: {anomaly_df['timestamp'].max()}")
    print(f"Temperature range in anomalies: {anomaly_df['temperature'].min():.2f}°C to {anomaly_df['temperature'].max():.2f}°C")
    print(f"Average anomaly score: {anomaly_df['anomaly_score'].mean():.4f}")

result_df = test_df[['timestamp', 'temperature', 'anomaly_score', 'is_anomaly', 'severity']].copy()
result_df.to_csv('data/detection_results.csv', index=False)
print(f"\nDetection results saved to: data/detection_results.csv")

fig, axes = plt.subplots(4, 1, figsize=(16, 14))

axes[0].plot(test_df['timestamp'], test_df['temperature'], 
            linewidth=1.2, alpha=0.7, color='blue', label='Temperature')
if len(mild_df) > 0:
    axes[0].scatter(mild_df['timestamp'], mild_df['temperature'], 
                   color='yellow', s=50, alpha=0.8, label=f'Mild Anomaly ({len(mild_df)})', zorder=5)
if len(moderate_df) > 0:
    axes[0].scatter(moderate_df['timestamp'], moderate_df['temperature'], 
                   color='orange', s=50, alpha=0.8, label=f'Moderate Anomaly ({len(moderate_df)})', zorder=5)
if len(severe_df) > 0:
    axes[0].scatter(severe_df['timestamp'], severe_df['temperature'], 
                   color='red', s=50, alpha=0.8, label=f'Severe Anomaly ({len(severe_df)})', zorder=5)
axes[0].set_xlabel('Timestamp')
axes[0].set_ylabel('Temperature (°C)')
axes[0].set_title('Last 7 Days - Temperature with Anomaly Detection', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

axes[1].plot(test_df['timestamp'], test_df['anomaly_score'], 
            linewidth=1.0, alpha=0.7, color='green', label='Anomaly Score')
threshold = np.percentile(anomaly_scores, 5)
axes[1].axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
               alpha=0.7, label=f'Threshold ({threshold:.4f})')
if len(anomaly_df) > 0:
    axes[1].scatter(anomaly_df['timestamp'], anomaly_df['anomaly_score'], 
                   color='red', s=50, alpha=0.8, zorder=5)
axes[1].set_xlabel('Timestamp')
axes[1].set_ylabel('Anomaly Score')
axes[1].set_title('Anomaly Scores Over Time', fontsize=12, fontweight='bold')
axes[1].legend(loc='upper left')
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='x', rotation=45)

daily_anomalies = test_df.groupby(test_df['timestamp'].dt.date)['is_anomaly'].sum()
axes[2].bar(range(len(daily_anomalies)), daily_anomalies.values, 
           alpha=0.7, color='coral', edgecolor='black')
axes[2].set_xlabel('Day')
axes[2].set_ylabel('Number of Anomalies')
axes[2].set_title('Daily Anomaly Count', fontsize=12, fontweight='bold')
axes[2].set_xticks(range(len(daily_anomalies)))
axes[2].set_xticklabels([str(d) for d in daily_anomalies.index], rotation=45)
axes[2].grid(True, alpha=0.3, axis='y')

temp_bins = np.linspace(test_df['temperature'].min(), test_df['temperature'].max(), 30)
axes[3].hist(normal_df['temperature'], bins=temp_bins, alpha=0.6, 
            color='blue', label=f'Normal ({len(normal_df)})', edgecolor='black')
if len(mild_df) > 0:
    axes[3].hist(mild_df['temperature'], bins=temp_bins, alpha=0.6, 
                color='yellow', label=f'Mild Anomaly ({len(mild_df)})', edgecolor='black')
if len(moderate_df) > 0:
    axes[3].hist(moderate_df['temperature'], bins=temp_bins, alpha=0.6, 
                color='orange', label=f'Moderate Anomaly ({len(moderate_df)})', edgecolor='black')
if len(severe_df) > 0:
    axes[3].hist(severe_df['temperature'], bins=temp_bins, alpha=0.6, 
                color='red', label=f'Severe Anomaly ({len(severe_df)})', edgecolor='black')
axes[3].set_xlabel('Temperature (°C)')
axes[3].set_ylabel('Frequency')
axes[3].set_title('Temperature Distribution: Normal vs Anomalies', fontsize=12, fontweight='bold')
axes[3].legend()
axes[3].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('output/3_detection_results.png', dpi=150)
plt.show()

print("Visualization saved to: output/3_detection_results.png")

fig2, axes2 = plt.subplots(2, 1, figsize=(16, 10))

axes2[0].plot(test_df['timestamp'], test_df['temperature'], 
             linewidth=1.5, alpha=0.8, color='steelblue', label='Temperature')
if len(mild_df) > 0:
    axes2[0].scatter(mild_df['timestamp'], mild_df['temperature'], 
                    color='yellow', s=80, alpha=0.9, label='Mild Anomaly', 
                    edgecolors='orange', linewidths=1.5, zorder=5)
if len(moderate_df) > 0:
    axes2[0].scatter(moderate_df['timestamp'], moderate_df['temperature'], 
                    color='orange', s=80, alpha=0.9, label='Moderate Anomaly', 
                    edgecolors='darkorange', linewidths=1.5, zorder=5)
if len(severe_df) > 0:
    axes2[0].scatter(severe_df['timestamp'], severe_df['temperature'], 
                    color='red', s=80, alpha=0.9, label='Severe Anomaly', 
                    edgecolors='darkred', linewidths=1.5, zorder=5)
if len(anomaly_df) > 0:
    for idx, row in anomaly_df.iterrows():
        severity_color = {'Mild': 'yellow', 'Moderate': 'orange', 'Severe': 'red'}[row['severity']]
        axes2[0].annotate(f"{row['temperature']:.1f}°C\n{row['severity']}", 
                         xy=(row['timestamp'], row['temperature']),
                         xytext=(10, 10), textcoords='offset points',
                         fontsize=7, alpha=0.8,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor=severity_color, alpha=0.6))
axes2[0].set_xlabel('Timestamp', fontsize=11)
axes2[0].set_ylabel('Temperature (°C)', fontsize=11)
axes2[0].set_title('Anomaly Detection - Detailed View (Last 7 Days)', 
                   fontsize=14, fontweight='bold')
axes2[0].legend(loc='upper left', fontsize=10)
axes2[0].grid(True, alpha=0.3)
axes2[0].tick_params(axis='x', rotation=45)

severity_color_map = {'Normal': 'green', 'Mild': 'yellow', 'Moderate': 'orange', 'Severe': 'red'}
score_colors = [severity_color_map[sev] for sev in test_df['severity']]
axes2[1].scatter(test_df['timestamp'], test_df['anomaly_score'], 
                c=score_colors, s=10, alpha=0.6)
axes2[1].axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                alpha=0.8, label=f'Anomaly Threshold')
axes2[1].set_xlabel('Timestamp', fontsize=11)
axes2[1].set_ylabel('Anomaly Score', fontsize=11)
axes2[1].set_title('Anomaly Score Timeline', fontsize=12, fontweight='bold')
axes2[1].legend(loc='upper left', fontsize=10)
axes2[1].grid(True, alpha=0.3)
axes2[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('output/3_detection_detailed.png', dpi=150)
plt.show()

print("Detailed visualization saved to: output/3_detection_detailed.png")

print("\n" + "="*70)
print("ANOMALY DETECTION SUMMARY")
print("="*70)
print(f"Analysis period: Last 7 days")
print(f"Total samples: {len(test_df)}")
print(f"Normal samples: {len(normal_df)} ({len(normal_df)/len(test_df)*100:.2f}%)")
print(f"Anomalies detected: {len(anomaly_df)} ({len(anomaly_df)/len(test_df)*100:.2f}%)")
if len(anomaly_df) > 0:
    print(f"\nSeverity Distribution:")
    print(f"  - Mild Anomaly: {len(mild_df)} ({len(mild_df)/len(test_df)*100:.2f}%)")
    print(f"  - Moderate Anomaly: {len(moderate_df)} ({len(moderate_df)/len(test_df)*100:.2f}%)")
    print(f"  - Severe Anomaly: {len(severe_df)} ({len(severe_df)/len(test_df)*100:.2f}%)")
    print(f"\nAnomaly period: {anomaly_df['timestamp'].min()} to {anomaly_df['timestamp'].max()}")
    print(f"Peak anomaly temperature: {anomaly_df['temperature'].max():.2f}°C")
    print(f"Lowest anomaly score: {anomaly_df['anomaly_score'].min():.4f}")
    if len(severe_df) > 0:
        print(f"\nSevere Anomaly Details:")
        print(f"  First occurrence: {severe_df['timestamp'].min()}")
        print(f"  Peak temperature: {severe_df['temperature'].max():.2f}°C")
        print(f"  Lowest score: {severe_df['anomaly_score'].min():.4f}")
print("="*70)
