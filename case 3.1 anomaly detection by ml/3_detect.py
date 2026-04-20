import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

def create_directories():
    os.makedirs('output', exist_ok=True)
    os.makedirs('data', exist_ok=True)

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

def classify_anomaly_severity(scores, threshold):
    anomaly_scores_only = scores[scores > threshold]
    
    if len(anomaly_scores_only) == 0:
        return scores, None, None
    
    q75 = np.percentile(anomaly_scores_only, 75)
    
    severity = np.where(scores <= threshold, 'normal',
                       np.where(scores > q75, 'severe', 'moderate'))
    
    return severity, threshold, q75

def detect_anomalies(clf, df, feature_cols):
    detect_start = pd.Timestamp('2024-03-24 00:00:00')
    detect_df = df[df['timestamp'] >= detect_start].copy()
    
    detect_df = create_features(detect_df)
    
    X_detect = detect_df[feature_cols].values
    
    anomaly_scores = clf.decision_function(X_detect)
    
    severity, threshold, severe_threshold = classify_anomaly_severity(
        anomaly_scores, clf.threshold_
    )
    
    detect_df['anomaly_score'] = anomaly_scores
    detect_df['severity'] = severity
    
    return detect_df, threshold, severe_threshold

def visualize_detection_results(detect_df, threshold, severe_threshold):
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    normal_data = detect_df[detect_df['severity'] == 'normal']
    moderate_data = detect_df[detect_df['severity'] == 'moderate']
    severe_data = detect_df[detect_df['severity'] == 'severe']
    
    axes[0].plot(detect_df['timestamp'], detect_df['energy_usage'], 
                 linewidth=1, alpha=0.5, color='gray', label='Energy Usage')
    axes[0].scatter(moderate_data['timestamp'], moderate_data['energy_usage'], 
                    s=100, alpha=0.9, label='Moderate Anomaly', color='orange', 
                    edgecolors='black', linewidths=2, zorder=5, marker='o')
    axes[0].scatter(severe_data['timestamp'], severe_data['energy_usage'], 
                    s=150, alpha=0.9, label='Severe Anomaly', color='red', 
                    edgecolors='darkred', linewidths=2, zorder=6, marker='X')
    
    for _, row in moderate_data.iterrows():
        axes[0].annotate('M', xy=(row['timestamp'], row['energy_usage']), 
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=8, fontweight='bold', color='orange')
    
    for _, row in severe_data.iterrows():
        axes[0].annotate('S', xy=(row['timestamp'], row['energy_usage']), 
                        xytext=(0, 15), textcoords='offset points',
                        ha='center', fontsize=9, fontweight='bold', color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    axes[0].set_xlabel('Timestamp')
    axes[0].set_ylabel('Energy Usage (kWh)')
    axes[0].set_title('Anomaly Detection Results - Testing Period (Mar 24-31) (M=Moderate, S=Severe)')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(detect_df['timestamp'], detect_df['anomaly_score'], 
                 linewidth=1, alpha=0.7, color='blue', label='Anomaly Score')
    axes[1].axhline(y=threshold, color='orange', linestyle='--', 
                    linewidth=2, label=f'Anomaly Threshold: {threshold:.3f}')
    axes[1].axhline(y=severe_threshold, color='red', linestyle='--', 
                    linewidth=2, label=f'Severe Threshold: {severe_threshold:.3f}')
    axes[1].fill_between(detect_df['timestamp'], threshold, 
                         detect_df['anomaly_score'].max(), 
                         where=(detect_df['anomaly_score'] > threshold),
                         alpha=0.2, color='orange', label='Anomaly Region')
    axes[1].fill_between(detect_df['timestamp'], severe_threshold, 
                         detect_df['anomaly_score'].max(), 
                         where=(detect_df['anomaly_score'] > severe_threshold),
                         alpha=0.3, color='red', label='Severe Region')
    axes[1].set_xlabel('Timestamp')
    axes[1].set_ylabel('Anomaly Score')
    axes[1].set_title('Anomaly Scores Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    severity_counts = detect_df['severity'].value_counts()
    colors_map = {'normal': 'green', 'moderate': 'orange', 'severe': 'red'}
    colors = [colors_map.get(x, 'gray') for x in severity_counts.index]
    
    bars = axes[2].bar(severity_counts.index, severity_counts.values, 
                       color=colors, alpha=0.7, edgecolor='black')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Distribution of Anomaly Severity')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({height/len(detect_df)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/3_detection_results.png', dpi=300, bbox_inches='tight')
    print("Detection visualization saved to output/3_detection_results.png")
    plt.show()

def visualize_anomaly_details(detect_df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    moderate_data = detect_df[detect_df['severity'] == 'moderate']
    severe_data = detect_df[detect_df['severity'] == 'severe']
    
    if len(moderate_data) > 0:
        axes[0, 0].scatter(moderate_data['timestamp'], moderate_data['energy_usage'],
                          c=moderate_data['anomaly_score'], cmap='YlOrRd', 
                          s=100, alpha=0.7, edgecolors='black')
        axes[0, 0].set_xlabel('Timestamp')
        axes[0, 0].set_ylabel('Energy Usage (kWh)')
        axes[0, 0].set_title('Moderate Anomalies (colored by score)')
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0], label='Anomaly Score')
    else:
        axes[0, 0].text(0.5, 0.5, 'No Moderate Anomalies', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
    
    if len(severe_data) > 0:
        axes[0, 1].scatter(severe_data['timestamp'], severe_data['energy_usage'],
                          c=severe_data['anomaly_score'], cmap='Reds', 
                          s=150, alpha=0.8, edgecolors='black', linewidths=2)
        axes[0, 1].set_xlabel('Timestamp')
        axes[0, 1].set_ylabel('Energy Usage (kWh)')
        axes[0, 1].set_title('Severe Anomalies (colored by score)')
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1], label='Anomaly Score')
    else:
        axes[0, 1].text(0.5, 0.5, 'No Severe Anomalies', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
    
    hourly_severity = detect_df.groupby(['hour', 'severity']).size().unstack(fill_value=0)
    
    if 'normal' in hourly_severity.columns:
        axes[1, 0].plot(hourly_severity.index, hourly_severity['normal'], 
                       marker='o', label='Normal', color='green', linewidth=2)
    if 'moderate' in hourly_severity.columns:
        axes[1, 0].plot(hourly_severity.index, hourly_severity['moderate'], 
                       marker='s', label='Moderate', color='orange', linewidth=2)
    if 'severe' in hourly_severity.columns:
        axes[1, 0].plot(hourly_severity.index, hourly_severity['severe'], 
                       marker='^', label='Severe', color='red', linewidth=2)
    
    axes[1, 0].set_xlabel('Hour of Day')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Anomaly Severity by Hour')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(range(0, 24, 2))
    
    daily_severity = detect_df.groupby([detect_df['timestamp'].dt.date, 'severity']).size().unstack(fill_value=0)
    
    x_pos = np.arange(len(daily_severity))
    width = 0.25
    
    if 'normal' in daily_severity.columns:
        axes[1, 1].bar(x_pos - width, daily_severity['normal'], 
                      width, label='Normal', color='green', alpha=0.7)
    if 'moderate' in daily_severity.columns:
        axes[1, 1].bar(x_pos, daily_severity['moderate'], 
                      width, label='Moderate', color='orange', alpha=0.7)
    if 'severe' in daily_severity.columns:
        axes[1, 1].bar(x_pos + width, daily_severity['severe'], 
                      width, label='Severe', color='red', alpha=0.7)
    
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Anomaly Severity by Day')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([str(d) for d in daily_severity.index], rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('output/3_anomaly_details.png', dpi=300, bbox_inches='tight')
    print("Anomaly details visualization saved to output/3_anomaly_details.png")
    plt.show()

def visualize_threshold_comparison(detect_df):
    fig, axes = plt.subplots(3, 1, figsize=(15, 14))
    
    fixed_threshold = 90
    
    moderate_data = detect_df[detect_df['severity'] == 'moderate']
    severe_data = detect_df[detect_df['severity'] == 'severe']
    all_ml_anomalies = detect_df[detect_df['severity'] != 'normal']
    
    axes[0].plot(detect_df['timestamp'], detect_df['energy_usage'], 
                 linewidth=1, alpha=0.7, color='blue', label='Energy Usage')
    axes[0].axhline(y=fixed_threshold, color='red', linestyle='--', 
                    linewidth=2, label=f'Fixed Threshold: {fixed_threshold} kWh')
    
    threshold_alerts = detect_df[detect_df['energy_usage'] > fixed_threshold]
    axes[0].scatter(threshold_alerts['timestamp'], threshold_alerts['energy_usage'],
                   color='red', s=100, alpha=0.6, marker='x', 
                   label=f'Fixed Threshold Alerts ({len(threshold_alerts)})', zorder=5)
    
    axes[0].set_xlabel('Timestamp')
    axes[0].set_ylabel('Energy Usage (kWh)')
    axes[0].set_title('Panel A: Fixed Threshold Approach (90 kWh)')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(detect_df['timestamp'], detect_df['energy_usage'], 
                 linewidth=1, alpha=0.5, color='gray', label='Energy Usage')
    axes[1].scatter(moderate_data['timestamp'], moderate_data['energy_usage'], 
                    s=100, alpha=0.9, label=f'Moderate Anomaly ({len(moderate_data)})', 
                    color='orange', edgecolors='black', linewidths=2, zorder=5, marker='o')
    axes[1].scatter(severe_data['timestamp'], severe_data['energy_usage'], 
                    s=150, alpha=0.9, label=f'Severe Anomaly ({len(severe_data)})', 
                    color='red', edgecolors='darkred', linewidths=2, zorder=6, marker='X')
    
    axes[1].set_xlabel('Timestamp')
    axes[1].set_ylabel('Energy Usage (kWh)')
    axes[1].set_title('Panel B: Anomaly Detection Approach (ML-based)')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(detect_df['timestamp'], detect_df['energy_usage'], 
                 linewidth=1, alpha=0.3, color='gray', label='Energy Usage')
    axes[2].axhline(y=fixed_threshold, color='red', linestyle='--', 
                    linewidth=1.5, alpha=0.5, label=f'Fixed Threshold: {fixed_threshold} kWh')
    
    detected_by_threshold = all_ml_anomalies[all_ml_anomalies['energy_usage'] > fixed_threshold]
    missed_by_threshold = all_ml_anomalies[all_ml_anomalies['energy_usage'] <= fixed_threshold]
    
    axes[2].scatter(detected_by_threshold['timestamp'], detected_by_threshold['energy_usage'],
                   color='green', s=120, alpha=0.8, marker='o', 
                   label=f'Detected by Both ({len(detected_by_threshold)})', 
                   edgecolors='darkgreen', linewidths=2, zorder=5)
    
    axes[2].scatter(missed_by_threshold['timestamp'], missed_by_threshold['energy_usage'],
                   color='red', s=150, alpha=0.9, marker='X', 
                   label=f'MISSED by Fixed Threshold ({len(missed_by_threshold)})', 
                   edgecolors='darkred', linewidths=2.5, zorder=6)
    
    for _, row in missed_by_threshold.iterrows():
        axes[2].annotate('MISSED!', 
                        xy=(row['timestamp'], row['energy_usage']), 
                        xytext=(0, 20), textcoords='offset points',
                        ha='center', fontsize=9, fontweight='bold', color='red',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                                 edgecolor='red', alpha=0.8, linewidth=2),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    axes[2].set_xlabel('Timestamp')
    axes[2].set_ylabel('Energy Usage (kWh)')
    axes[2].set_title('Panel C: Direct Comparison - Highlighting Missed Anomalies')
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/3_threshold_comparison.png', dpi=300, bbox_inches='tight')
    print("Threshold comparison visualization saved to output/3_threshold_comparison.png")
    plt.show()
    
    print("\n=== Comparison Analysis ===")
    threshold_alert_count = len(threshold_alerts)
    ml_anomaly_count = len(all_ml_anomalies)
    
    print(f"Fixed Threshold (>{fixed_threshold} kWh):")
    print(f"  Total alerts: {threshold_alert_count}")
    print(f"  Detected anomalies: {len(detected_by_threshold)}")
    print(f"  MISSED anomalies: {len(missed_by_threshold)}")
    
    print(f"\nAnomaly Detection (ML-based):")
    print(f"  Total anomalies: {ml_anomaly_count}")
    print(f"  Moderate: {len(moderate_data)}")
    print(f"  Severe: {len(severe_data)}")
    
    print(f"\nMissed Anomalies Details:")
    for _, row in missed_by_threshold.iterrows():
        print(f"  {row['timestamp']}: {row['energy_usage']:.2f} kWh (Severity: {row['severity']})")
    
    print(f"\nKey Insight: Fixed threshold missed {len(missed_by_threshold)} out of {ml_anomaly_count} anomalies ({len(missed_by_threshold)/ml_anomaly_count*100:.1f}%)")

def print_detection_summary(detect_df, threshold, severe_threshold):
    print("\n=== Detection Summary ===")
    print(f"Detection period: {detect_df['timestamp'].min()} to {detect_df['timestamp'].max()}")
    print(f"Total records analyzed: {len(detect_df)}")
    print(f"\nAnomaly Thresholds:")
    print(f"  Normal/Anomaly threshold: {threshold:.3f}")
    print(f"  Moderate/Severe threshold: {severe_threshold:.3f}")
    
    print(f"\nSeverity Distribution:")
    severity_counts = detect_df['severity'].value_counts()
    for severity in ['normal', 'moderate', 'severe']:
        count = severity_counts.get(severity, 0)
        percentage = count / len(detect_df) * 100
        print(f"  {severity.capitalize()}: {count} ({percentage:.2f}%)")
    
    print(f"\nAnomaly Score Statistics:")
    print(f"  Min: {detect_df['anomaly_score'].min():.3f}")
    print(f"  Max: {detect_df['anomaly_score'].max():.3f}")
    print(f"  Mean: {detect_df['anomaly_score'].mean():.3f}")
    print(f"  Std: {detect_df['anomaly_score'].std():.3f}")
    
    moderate_data = detect_df[detect_df['severity'] == 'moderate']
    severe_data = detect_df[detect_df['severity'] == 'severe']
    
    if len(moderate_data) > 0:
        print(f"\nModerate Anomalies:")
        print(f"  Count: {len(moderate_data)}")
        print(f"  Score range: [{moderate_data['anomaly_score'].min():.3f}, {moderate_data['anomaly_score'].max():.3f}]")
        print(f"  Energy usage range: [{moderate_data['energy_usage'].min():.2f}, {moderate_data['energy_usage'].max():.2f}] kWh")
        print(f"  Time periods:")
        for _, row in moderate_data.iterrows():
            print(f"    {row['timestamp']}: {row['energy_usage']:.2f} kWh (score: {row['anomaly_score']:.3f})")
    
    if len(severe_data) > 0:
        print(f"\nSevere Anomalies:")
        print(f"  Count: {len(severe_data)}")
        print(f"  Score range: [{severe_data['anomaly_score'].min():.3f}, {severe_data['anomaly_score'].max():.3f}]")
        print(f"  Energy usage range: [{severe_data['energy_usage'].min():.2f}, {severe_data['energy_usage'].max():.2f}] kWh")
        print(f"  Time periods:")
        for _, row in severe_data.iterrows():
            print(f"    {row['timestamp']}: {row['energy_usage']:.2f} kWh (score: {row['anomaly_score']:.3f})")

if __name__ == "__main__":
    create_directories()
    
    print("Loading model and data...")
    clf = joblib.load('model/iforest_model.pkl')
    feature_cols_df = pd.read_csv('model/feature_columns.csv')
    feature_cols = feature_cols_df['feature'].tolist()
    
    df = pd.read_csv('data/office_energy_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("Detecting anomalies in last week...")
    detect_df, threshold, severe_threshold = detect_anomalies(clf, df, feature_cols)
    
    detect_df.to_csv('data/detection_results.csv', index=False)
    print("Detection results saved to data/detection_results.csv")
    
    print_detection_summary(detect_df, threshold, severe_threshold)
    
    visualize_detection_results(detect_df, threshold, severe_threshold)
    
    visualize_anomaly_details(detect_df)
    
    visualize_threshold_comparison(detect_df)
    
    print("\n=== Detection Complete ===")
