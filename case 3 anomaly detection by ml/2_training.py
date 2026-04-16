import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

os.makedirs('model', exist_ok=True)
os.makedirs('output', exist_ok=True)

df = pd.read_csv('data/server_room_temperature.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Loaded {len(df)} temperature readings")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

training_end_idx = int(len(df) * 0.80)
train_df = df.iloc[:training_end_idx].copy()

print(f"\nTraining data: {len(train_df)} samples")
print(f"Training period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")

train_df['hour'] = train_df['timestamp'].dt.hour
train_df['minute'] = train_df['timestamp'].dt.minute
train_df['day_of_week'] = train_df['timestamp'].dt.dayofweek

train_df['temp_lag_1'] = train_df['temperature'].shift(1)
train_df['temp_lag_4'] = train_df['temperature'].shift(4)
train_df['temp_lag_12'] = train_df['temperature'].shift(12)
train_df['temp_lag_24'] = train_df['temperature'].shift(24)

train_df['temp_change_1'] = train_df['temperature'] - train_df['temp_lag_1']
train_df['temp_change_4'] = train_df['temperature'] - train_df['temp_lag_4']

train_df['temp_rolling_mean_6h'] = train_df['temperature'].rolling(window=24, min_periods=1).mean()
train_df['temp_rolling_std_6h'] = train_df['temperature'].rolling(window=24, min_periods=1).std()
train_df['temp_rolling_max_6h'] = train_df['temperature'].rolling(window=24, min_periods=1).max()
train_df['temp_rolling_min_6h'] = train_df['temperature'].rolling(window=24, min_periods=1).min()

features = ['temperature', 'hour', 'minute', 
            'temp_lag_1', 'temp_lag_4', 'temp_lag_12', 'temp_lag_24',
            'temp_change_1', 'temp_change_4',
            'temp_rolling_mean_6h', 'temp_rolling_std_6h',
            'temp_rolling_max_6h', 'temp_rolling_min_6h']
X_train = train_df[features].fillna(method='bfill')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = IsolationForest(
    contamination=0.05,
    random_state=42,
    n_estimators=100,
    max_samples='auto',
    max_features=1.0
)

print("\nTraining Isolation Forest model...")
model.fit(X_train_scaled)

train_predictions = model.predict(X_train_scaled)
train_scores = model.score_samples(X_train_scaled)

train_df['anomaly_score'] = train_scores
train_df['is_anomaly'] = train_predictions == -1

anomaly_count = train_df['is_anomaly'].sum()
print(f"\nTraining completed!")
print(f"Anomalies detected in training data: {anomaly_count} ({anomaly_count/len(train_df)*100:.2f}%)")

joblib.dump(model, 'model/isolation_forest.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
print("\nModel saved to: model/isolation_forest.pkl")
print("Scaler saved to: model/scaler.pkl")

fig, axes = plt.subplots(3, 1, figsize=(15, 12))

axes[0].plot(train_df['timestamp'], train_df['temperature'], 
            linewidth=0.8, alpha=0.7, label='Temperature')
anomaly_points = train_df[train_df['is_anomaly']]
axes[0].scatter(anomaly_points['timestamp'], anomaly_points['temperature'], 
               color='red', s=20, alpha=0.6, label=f'Anomalies ({len(anomaly_points)})')
axes[0].set_xlabel('Timestamp')
axes[0].set_ylabel('Temperature (°C)')
axes[0].set_title('Training Data - Temperature with Detected Anomalies')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

axes[1].plot(train_df['timestamp'], train_df['anomaly_score'], 
            linewidth=0.8, alpha=0.7, color='blue')
axes[1].axhline(y=train_df['anomaly_score'].quantile(0.05), 
               color='red', linestyle='--', alpha=0.7, 
               label=f'Threshold (5th percentile)')
axes[1].set_xlabel('Timestamp')
axes[1].set_ylabel('Anomaly Score')
axes[1].set_title('Anomaly Scores Over Time')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='x', rotation=45)

axes[2].hist(train_df['anomaly_score'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[2].axvline(x=train_df['anomaly_score'].quantile(0.05), 
               color='red', linestyle='--', linewidth=2, 
               label=f'Threshold (5th percentile)')
axes[2].set_xlabel('Anomaly Score')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Distribution of Anomaly Scores')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/2_training_results.png', dpi=150)
plt.show()

print("Visualization saved to: output/2_training_results.png")

print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"Model: Isolation Forest")
print(f"Number of features: {len(features)}")
print(f"Features used:")
print(f"  - Time features: hour, minute")
print(f"  - Lag features: 1, 4, 12, 24 steps (15min, 1h, 3h, 6h)")
print(f"  - Change features: 1-step, 4-step changes")
print(f"  - Rolling features: mean, std, max, min (6h window)")
print(f"Training samples: {len(train_df)}")
print(f"Contamination rate: 5%")
print(f"Anomaly score range: [{train_df['anomaly_score'].min():.4f}, {train_df['anomaly_score'].max():.4f}]")
print(f"Anomaly threshold: {train_df['anomaly_score'].quantile(0.05):.4f}")
print("="*60)
