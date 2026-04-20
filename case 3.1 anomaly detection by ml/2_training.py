import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyod.models.iforest import IForest
import joblib
import os

def create_directories():
    os.makedirs('model', exist_ok=True)
    os.makedirs('output', exist_ok=True)

def create_features(df):
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekday'] = (df['day_of_week'] < 5).astype(int)
    df['is_office_hours'] = ((df['hour'] >= 9) & (df['hour'] < 18)).astype(int)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

def prepare_training_data(df):
    training_end = pd.Timestamp('2024-03-24 23:45:00')
    train_df = df[df['timestamp'] <= training_end].copy()
    
    train_df = create_features(train_df)
    
    feature_cols = [
        'energy_usage', 
        # 'hour',
        # 'day_of_week',
        'is_weekday', 
        'is_office_hours', 
        # 'hour_sin', 
        # 'hour_cos', 
        # 'day_sin', 
        # 'day_cos'
    ]
    
    X_train = train_df[feature_cols].values
    
    return X_train, train_df, feature_cols

def train_iforest(X_train):
    print("Training Isolation Forest model...")
    
    clf = IForest(
        n_estimators=100,
        max_samples='auto',
        contamination=0.01,
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_train)
    
    print(f"\nModel trained successfully")
    print(f"Number of estimators: {clf.n_estimators}")
    print(f"Contamination: {clf.contamination}")
    print(f"Training samples: {X_train.shape[0]}")
    
    return clf

def visualize_training_results(clf, X_train, train_df):
    train_scores = clf.decision_function(X_train)
    train_labels = clf.labels_
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(train_df['timestamp'], train_df['energy_usage'], 
                    linewidth=0.5, alpha=0.7, label='Energy Usage')
    anomaly_idx = train_labels == 1
    axes[0, 0].scatter(train_df[anomaly_idx]['timestamp'], 
                       train_df[anomaly_idx]['energy_usage'],
                       color='red', s=10, alpha=0.6, label='Detected Anomalies')
    axes[0, 0].set_xlabel('Timestamp')
    axes[0, 0].set_ylabel('Energy Usage (kWh)')
    axes[0, 0].set_title('Training Data with Detected Anomalies')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(train_scores, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=clf.threshold_, color='red', linestyle='--', 
                       linewidth=2, label=f'Threshold: {clf.threshold_:.3f}')
    axes[0, 1].set_xlabel('Anomaly Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Anomaly Scores (Training Data)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    train_df_with_scores = train_df.copy()
    train_df_with_scores['anomaly_score'] = train_scores
    train_df_with_scores['predicted_anomaly'] = train_labels
    
    hourly_avg = train_df_with_scores.groupby('hour').agg({
        'energy_usage': 'mean',
        'anomaly_score': 'mean'
    })
    
    ax2 = axes[1, 0]
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(hourly_avg.index, hourly_avg['energy_usage'], 
                     marker='o', color='blue', label='Avg Energy Usage', linewidth=2)
    line2 = ax2_twin.plot(hourly_avg.index, hourly_avg['anomaly_score'], 
                          marker='s', color='red', label='Avg Anomaly Score', linewidth=2)
    
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Average Energy Usage (kWh)', color='blue')
    ax2_twin.set_ylabel('Average Anomaly Score', color='red')
    ax2.set_title('Hourly Patterns in Training Data')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(0, 24, 2))
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    
    weekday_scores = train_df_with_scores[train_df_with_scores['is_weekday'] == 1]['anomaly_score']
    weekend_scores = train_df_with_scores[train_df_with_scores['is_weekday'] == 0]['anomaly_score']
    
    axes[1, 1].boxplot([weekday_scores, weekend_scores], 
                        labels=['Weekday', 'Weekend'],
                        patch_artist=True)
    axes[1, 1].set_ylabel('Anomaly Score')
    axes[1, 1].set_title('Anomaly Score Distribution: Weekday vs Weekend')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('output/2_training_results.png', dpi=300, bbox_inches='tight')
    print("Training visualization saved to output/2_training_results.png")
    plt.show()
    
    print("\n=== Training Statistics ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Anomalies detected in training: {train_labels.sum()} ({train_labels.sum()/len(train_labels)*100:.2f}%)")
    print(f"Anomaly score range: [{train_scores.min():.3f}, {train_scores.max():.3f}]")
    print(f"Anomaly threshold: {clf.threshold_:.3f}")
    print(f"Mean anomaly score: {train_scores.mean():.3f}")
    print(f"Std anomaly score: {train_scores.std():.3f}")

if __name__ == "__main__":
    create_directories()
    
    print("Loading data...")
    df = pd.read_csv('data/office_energy_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    X_train, train_df, feature_cols = prepare_training_data(df)
    
    clf = train_iforest(X_train)
    
    joblib.dump(clf, 'model/iforest_model.pkl')
    print("\nModel saved to model/iforest_model.pkl")
    
    pd.DataFrame({'feature': feature_cols}).to_csv('model/feature_columns.csv', index=False)
    print("Feature columns saved to model/feature_columns.csv")
    
    visualize_training_results(clf, X_train, train_df)
