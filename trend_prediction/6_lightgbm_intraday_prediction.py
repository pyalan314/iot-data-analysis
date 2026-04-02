import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# HK Public Holidays 2026
HK_HOLIDAYS_2026 = [
    '2026-01-01', '2026-01-29', '2026-01-30', '2026-01-31', '2026-02-02',
    '2026-04-03', '2026-04-06'
]

def load_data():
    """Load power usage data"""
    df = pd.read_csv('power_usage_data.csv', encoding='utf-8')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def create_features(df):
    """Create time-based features for LightGBM"""
    df = df.copy()
    
    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding for hour and minute
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Office hours indicator
    df['is_office_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 18) & 
                            (df['is_weekend'] == 0)).astype(int)
    
    # Holiday indicator
    df['is_holiday'] = df['timestamp'].dt.strftime('%Y-%m-%d').isin(HK_HOLIDAYS_2026).astype(int)
    
    # Time of day categories
    df['time_of_day'] = pd.cut(df['hour'], 
                                bins=[0, 6, 9, 12, 18, 21, 24],
                                labels=[0, 1, 2, 3, 4, 5],
                                include_lowest=True).astype(int)
    
    # Lag features (previous values)
    df['lag_1h'] = df['power_usage'].shift(60)  # 1 hour ago
    df['lag_24h'] = df['power_usage'].shift(60 * 24)  # 24 hours ago
    df['lag_1week'] = df['power_usage'].shift(60 * 24 * 7)  # 1 week ago
    
    # Rolling statistics
    df['rolling_mean_1h'] = df['power_usage'].rolling(window=60, min_periods=1).mean()
    df['rolling_std_1h'] = df['power_usage'].rolling(window=60, min_periods=1).std()
    df['rolling_mean_3h'] = df['power_usage'].rolling(window=180, min_periods=1).mean()
    
    return df

def evaluate_model(y_true, y_pred, dataset_name=""):
    """Calculate evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{dataset_name} Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R2:   {r2:.4f}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def main():
    print("=" * 80)
    print("Intraday Power Usage Prediction using LightGBM (Per-Minute Data)")
    print("Scenario: Predict 31/3 12pm to 11:59pm using data up to 31/3 12pm")
    print("=" * 80)
    
    # Load data
    print("\n[1] Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Data frequency: Per-minute (60 records/hour)")
    
    # Create features
    print("\n[2] Creating features...")
    df = create_features(df)
    print("Features created:")
    print("  - Time features: hour, minute, day_of_week, month")
    print("  - Cyclical encodings: hour_sin/cos, minute_sin/cos, day_sin/cos")
    print("  - Binary features: is_weekend, is_office_hour, is_holiday")
    print("  - Lag features: 1h, 24h, 1week ago")
    print("  - Rolling statistics: mean and std over 1h and 3h windows")
    
    # Filter data up to 31/3 12pm (noon)
    cutoff_time = pd.Timestamp('2026-03-31 12:00:00')
    train_data = df[df['timestamp'] <= cutoff_time].copy()
    
    # Drop rows with NaN (from lag features at the beginning)
    train_data = train_data.dropna()
    
    print(f"\n[3] Using data up to {cutoff_time} (inclusive)...")
    print(f"Training data: {len(train_data)} records (after removing NaN)")
    print(f"Last training timestamp: {train_data['timestamp'].max()}")
    
    # Prepare features and target
    feature_cols = ['hour', 'minute', 'day_of_week', 'day_of_month', 'month',
                    'is_weekend', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
                    'day_sin', 'day_cos', 'is_office_hour', 'is_holiday', 'time_of_day',
                    'lag_1h', 'lag_24h', 'lag_1week', 
                    'rolling_mean_1h', 'rolling_std_1h', 'rolling_mean_3h']
    
    X_train = train_data[feature_cols]
    y_train = train_data['power_usage']
    
    # Build LightGBM model
    print("\n[4] Building LightGBM model...")
    print("Model parameters:")
    print("  - Objective: regression")
    print("  - Boosting type: gbdt")
    print("  - Learning rate: 0.05")
    print("  - Max depth: 8")
    print("  - Num leaves: 64")
    print("  - Feature fraction: 0.8")
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'max_depth': 8,
        'num_leaves': 64,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    # Create dataset
    train_dataset = lgb.Dataset(X_train, label=y_train)
    
    # Train model
    print("\n[5] Training LightGBM model...")
    print("This may take a few minutes...")
    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=500,
        valid_sets=[train_dataset],
        valid_names=['train'],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    print(f"Model training complete! Best iteration: {model.best_iteration}")
    
    # Evaluate on training set (last 10% for validation)
    val_size = int(len(train_data) * 0.1)
    X_val = X_train.tail(val_size)
    y_val = y_train.tail(val_size)
    
    val_predictions = model.predict(X_val, num_iteration=model.best_iteration)
    evaluate_model(y_val, val_predictions, "Validation Set")
    
    # Create future timestamps for prediction (12pm to 11:59pm = 12 hours = 720 minutes)
    print("\n[6] Predicting 31/3 12pm to 11:59pm (720 minutes)...")
    
    last_timestamp = cutoff_time
    future_timestamps = [last_timestamp + timedelta(minutes=i) for i in range(1, 721)]
    
    # Create future dataframe
    future_df = pd.DataFrame({'timestamp': future_timestamps})
    
    # We need to iteratively predict because we use lag features
    predictions = []
    
    # Get the last known values for lag features
    last_known_data = df[df['timestamp'] <= cutoff_time].copy()
    
    print("Generating predictions iteratively (using lag features)...")
    for i, future_time in enumerate(future_timestamps):
        # Create features for this timestamp
        temp_df = pd.DataFrame({'timestamp': [future_time]})
        temp_df['hour'] = temp_df['timestamp'].dt.hour
        temp_df['minute'] = temp_df['timestamp'].dt.minute
        temp_df['day_of_week'] = temp_df['timestamp'].dt.dayofweek
        temp_df['day_of_month'] = temp_df['timestamp'].dt.day
        temp_df['month'] = temp_df['timestamp'].dt.month
        temp_df['is_weekend'] = (temp_df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding
        temp_df['hour_sin'] = np.sin(2 * np.pi * temp_df['hour'] / 24)
        temp_df['hour_cos'] = np.cos(2 * np.pi * temp_df['hour'] / 24)
        temp_df['minute_sin'] = np.sin(2 * np.pi * temp_df['minute'] / 60)
        temp_df['minute_cos'] = np.cos(2 * np.pi * temp_df['minute'] / 60)
        temp_df['day_sin'] = np.sin(2 * np.pi * temp_df['day_of_week'] / 7)
        temp_df['day_cos'] = np.cos(2 * np.pi * temp_df['day_of_week'] / 7)
        
        temp_df['is_office_hour'] = ((temp_df['hour'] >= 9) & (temp_df['hour'] <= 18) & 
                                      (temp_df['is_weekend'] == 0)).astype(int)
        temp_df['is_holiday'] = temp_df['timestamp'].dt.strftime('%Y-%m-%d').isin(HK_HOLIDAYS_2026).astype(int)
        temp_df['time_of_day'] = pd.cut(temp_df['hour'], 
                                         bins=[0, 6, 9, 12, 18, 21, 24],
                                         labels=[0, 1, 2, 3, 4, 5],
                                         include_lowest=True).astype(int)
        
        # Get lag features from historical + predicted data
        combined_data = pd.concat([last_known_data[['timestamp', 'power_usage']], 
                                   pd.DataFrame({'timestamp': future_timestamps[:i], 
                                                'power_usage': predictions})], 
                                  ignore_index=True)
        
        # Lag features
        if len(combined_data) >= 60:
            temp_df['lag_1h'] = combined_data.iloc[-60]['power_usage']
        else:
            temp_df['lag_1h'] = last_known_data.iloc[-1]['power_usage']
        
        if len(combined_data) >= 60 * 24:
            temp_df['lag_24h'] = combined_data.iloc[-60*24]['power_usage']
        else:
            temp_df['lag_24h'] = last_known_data.iloc[-60*24]['power_usage'] if len(last_known_data) >= 60*24 else last_known_data.iloc[0]['power_usage']
        
        if len(combined_data) >= 60 * 24 * 7:
            temp_df['lag_1week'] = combined_data.iloc[-60*24*7]['power_usage']
        else:
            temp_df['lag_1week'] = last_known_data.iloc[-60*24*7]['power_usage'] if len(last_known_data) >= 60*24*7 else last_known_data.iloc[0]['power_usage']
        
        # Rolling statistics
        recent_values = combined_data.tail(180)['power_usage'].values
        temp_df['rolling_mean_1h'] = recent_values[-60:].mean() if len(recent_values) >= 60 else recent_values.mean()
        temp_df['rolling_std_1h'] = recent_values[-60:].std() if len(recent_values) >= 60 else recent_values.std()
        temp_df['rolling_mean_3h'] = recent_values.mean()
        
        # Predict
        X_future = temp_df[feature_cols]
        pred = model.predict(X_future, num_iteration=model.best_iteration)[0]
        predictions.append(pred)
        
        if (i + 1) % 60 == 0:  # Every hour
            print(f"  Predicted {i+1}/{len(future_timestamps)} minutes...")
    
    # Create prediction DataFrame - include 12pm point for smooth connection
    cutoff_value = last_known_data[last_known_data['timestamp'] == cutoff_time]['power_usage'].values[0]
    
    all_timestamps = [cutoff_time] + future_timestamps
    all_predictions = [cutoff_value] + predictions
    
    prediction_df = pd.DataFrame({
        'timestamp': all_timestamps,
        'predicted_power_usage': all_predictions
    })
    
    print(f"\nGenerated {len(prediction_df)} per-minute predictions (including 12pm connection point)")
    print(f"Prediction period: {prediction_df['timestamp'].min()} to {prediction_df['timestamp'].max()}")
    
    # Save predictions
    output_file = 'output/6_predictions_lightgbm_intraday.csv'
    prediction_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Predictions saved to: {output_file}")
    
    # Get historical data from 26/3 onwards
    history_start = pd.Timestamp('2026-03-26 00:00:00')
    historical_data = df[
        (df['timestamp'] >= history_start) & 
        (df['timestamp'] <= cutoff_time)
    ].copy()
    
    print("\n[7] Historical data for plotting...")
    print(f"Period: {historical_data['timestamp'].min()} to {historical_data['timestamp'].max()}")
    print(f"Records: {len(historical_data)}")
    
    # Feature importance
    print("\n[8] Top 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:20s}: {row['importance']:.0f}")
    
    # Visualizations
    print("\n[9] Creating visualizations...")
    
    fig = plt.figure(figsize=(18, 14))
    
    # Plot 1: Main timeline
    ax1 = plt.subplot(4, 2, 1)
    hist_plot = historical_data.iloc[::10]
    pred_plot = prediction_df.iloc[::10]
    
    ax1.plot(hist_plot['timestamp'], hist_plot['power_usage'], 
             label='Historical (26/3 - 31/3 12pm)', color='blue', linewidth=1, alpha=0.7)
    ax1.plot(pred_plot['timestamp'], pred_plot['predicted_power_usage'], 
             label='Predicted (31/3 12pm - 11:59pm)', color='red', linewidth=1, alpha=0.8)
    ax1.axvline(x=cutoff_time, color='green', linestyle='--', 
                linewidth=2, label='Prediction Start (31/3 12pm)')
    ax1.set_title('Historical Data + Intraday LightGBM Predictions', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date & Time')
    ax1.set_ylabel('Power Usage (kW)')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: March 31 full day
    ax2 = plt.subplot(4, 2, 2)
    march31_hist = historical_data[historical_data['timestamp'].dt.date == pd.Timestamp('2026-03-31').date()]
    march31_plot = march31_hist.iloc[::5]
    pred31_plot = prediction_df.iloc[::5]
    
    ax2.plot(march31_plot['timestamp'], march31_plot['power_usage'], 
             label='Historical (31/3 00:00-12:00)', color='blue', linewidth=1.5, alpha=0.8)
    ax2.plot(pred31_plot['timestamp'], pred31_plot['predicted_power_usage'], 
             label='Predicted (31/3 12:00-23:59)', color='red', linewidth=1.5, alpha=0.8)
    ax2.axvline(x=cutoff_time, color='green', linestyle='--', linewidth=2, label='Cutoff (12pm)')
    ax2.set_title('Focus: March 31 (Full Day)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Power Usage (kW)')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Transition detail (11am - 2pm)
    ax3 = plt.subplot(4, 2, 3)
    transition_start = pd.Timestamp('2026-03-31 11:00:00')
    transition_end = pd.Timestamp('2026-03-31 14:00:00')
    
    hist_transition = historical_data[
        (historical_data['timestamp'] >= transition_start) & 
        (historical_data['timestamp'] <= cutoff_time)
    ]
    pred_transition = prediction_df[
        (prediction_df['timestamp'] >= cutoff_time) & 
        (prediction_df['timestamp'] <= transition_end)
    ]
    
    ax3.plot(hist_transition['timestamp'], hist_transition['power_usage'], 
             label='Historical', color='blue', linewidth=1, alpha=0.7)
    ax3.plot(pred_transition['timestamp'], pred_transition['predicted_power_usage'], 
             label='Predicted', color='red', linewidth=1, alpha=0.7)
    ax3.axvline(x=cutoff_time, color='green', linestyle='--', linewidth=2, label='Cutoff (12pm)')
    ax3.set_title('Transition Detail (11am - 2pm)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Power Usage (kW)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Last 24 hours + predictions
    ax4 = plt.subplot(4, 2, 4)
    last_24h = historical_data.tail(60 * 24)
    last_24h_plot = last_24h.iloc[::1]
    pred_plot2 = prediction_df.iloc[::1]
    
    ax4.plot(last_24h_plot['timestamp'], last_24h_plot['power_usage'], 
             label='Historical (Last 24h)', color='blue', linewidth=1.5, alpha=0.7)
    ax4.plot(pred_plot2['timestamp'], pred_plot2['predicted_power_usage'], 
             label='Predicted', color='red', linewidth=1.5, alpha=0.7)
    ax4.axvline(x=cutoff_time, color='green', linestyle='--', linewidth=2, label='Cutoff')
    ax4.set_title('Last 24 Hours + Predictions', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Date & Time')
    ax4.set_ylabel('Power Usage (kW)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # Plot 5: Feature importance
    ax5 = plt.subplot(4, 2, 5)
    top_features = feature_importance.head(15)
    ax5.barh(range(len(top_features)), top_features['importance'], color='steelblue', alpha=0.7)
    ax5.set_yticks(range(len(top_features)))
    ax5.set_yticklabels(top_features['feature'])
    ax5.set_xlabel('Importance (Gain)')
    ax5.set_title('Top 15 Feature Importance', fontsize=12, fontweight='bold')
    ax5.grid(True, axis='x', alpha=0.3)
    ax5.invert_yaxis()
    
    # Plot 6: Hourly pattern comparison
    ax6 = plt.subplot(4, 2, 6)
    prediction_df['hour'] = prediction_df['timestamp'].dt.hour
    historical_data['hour'] = historical_data['timestamp'].dt.hour
    
    hist_hourly_avg = historical_data.groupby('hour')['power_usage'].mean()
    pred_hourly_avg = prediction_df.groupby('hour')['predicted_power_usage'].mean()
    
    ax6.plot(hist_hourly_avg.index, hist_hourly_avg.values, 
             marker='o', label='Historical Avg', color='blue', linewidth=2, markersize=6)
    ax6.plot(pred_hourly_avg.index, pred_hourly_avg.values, 
             marker='s', label='Predicted Avg', color='red', linewidth=2, markersize=6)
    ax6.axvspan(9, 18, alpha=0.2, color='green', label='Office Hours')
    ax6.set_title('Average Hourly Pattern', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Hour of Day')
    ax6.set_ylabel('Average Power Usage (kW)')
    ax6.set_xticks(range(0, 24, 2))
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Residuals on validation set
    ax7 = plt.subplot(4, 2, 7)
    residuals = y_val - val_predictions
    ax7.scatter(val_predictions, residuals, alpha=0.3, s=1)
    ax7.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax7.set_xlabel('Predicted Power Usage (kW)')
    ax7.set_ylabel('Residuals (kW)')
    ax7.set_title('Residual Plot (Validation Set)', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Statistics summary
    ax8 = plt.subplot(4, 2, 8)
    ax8.axis('off')
    
    summary_text = f"""
LIGHTGBM INTRADAY FORECAST

Training Data:
  • Records: {len(train_data):,} minutes
  • Features: {len(feature_cols)}
  • Best iteration: {model.best_iteration}

Predictions:
  • Period: 31/3 12:00 - 23:59
  • Records: {len(prediction_df):,} minutes
  • Mean: {prediction_df['predicted_power_usage'].mean():.2f} kW
  • Std: {prediction_df['predicted_power_usage'].std():.2f} kW
  • Range: {prediction_df['predicted_power_usage'].min():.2f} - {prediction_df['predicted_power_usage'].max():.2f} kW

Model Performance:
  • Validation RMSE: {np.sqrt(mean_squared_error(y_val, val_predictions)):.4f}
  • Validation MAE: {mean_absolute_error(y_val, val_predictions):.4f}
  • Validation R²: {r2_score(y_val, val_predictions):.4f}

Top Features:
  1. {feature_importance.iloc[0]['feature']}
  2. {feature_importance.iloc[1]['feature']}
  3. {feature_importance.iloc[2]['feature']}
    """
    
    ax8.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
             verticalalignment='center', transform=ax8.transAxes)
    ax8.set_title('Model Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('Intraday Power Usage Prediction - LightGBM Model (Per-Minute Data)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plot_file = 'output/6_lightgbm_intraday_results.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {plot_file}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY (LIGHTGBM - INTRADAY)")
    print("=" * 80)
    
    print("\nScenario:")
    print(f"  - Training data: Up to {cutoff_time}")
    print(f"  - Prediction window: {prediction_df['timestamp'].min()} to {prediction_df['timestamp'].max()}")
    print("  - Period: 31/3 12pm to 11:59pm (same day)")
    print(f"  - Granularity: Per-minute ({len(prediction_df):,} predictions)")
    
    print(f"\nHistorical Context (26/3 - 31/3 12pm):")
    print(f"  Mean: {historical_data['power_usage'].mean():.2f} kW")
    print(f"  Std:  {historical_data['power_usage'].std():.2f} kW")
    print(f"  Min:  {historical_data['power_usage'].min():.2f} kW")
    print(f"  Max:  {historical_data['power_usage'].max():.2f} kW")
    
    print(f"\nPredicted Values (31/3 12pm - 11:59pm):")
    print(f"  Mean: {prediction_df['predicted_power_usage'].mean():.2f} kW")
    print(f"  Std:  {prediction_df['predicted_power_usage'].std():.2f} kW")
    print(f"  Min:  {prediction_df['predicted_power_usage'].min():.2f} kW")
    print(f"  Max:  {prediction_df['predicted_power_usage'].max():.2f} kW")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    
    plt.show()

if __name__ == "__main__":
    main()
