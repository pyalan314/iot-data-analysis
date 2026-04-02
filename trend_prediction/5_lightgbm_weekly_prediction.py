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
    
    # Lag features (previous values) - for 30-minute intervals
    df['lag_1h'] = df['power_usage'].shift(2)  # 2 intervals = 1 hour
    df['lag_24h'] = df['power_usage'].shift(48)  # 48 intervals = 24 hours
    df['lag_1week'] = df['power_usage'].shift(336)  # 336 intervals = 7 days
    
    # Rolling statistics - for 30-minute intervals
    df['rolling_mean_1h'] = df['power_usage'].rolling(window=2, min_periods=1).mean()  # 2 intervals = 1 hour
    df['rolling_std_1h'] = df['power_usage'].rolling(window=2, min_periods=1).std()
    df['rolling_mean_3h'] = df['power_usage'].rolling(window=6, min_periods=1).mean()  # 6 intervals = 3 hours
    
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
    print("Weekly Power Usage Prediction using LightGBM (Per-30-Minute Data)")
    print("Scenario: Predict 1/4 to 7/4 using data up to 31/3 11:59pm")
    print("=" * 80)
    
    # Load data
    print("\n[1] Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")
    print("Data frequency: Per-minute (will resample to 30-minute intervals)")
    
    # Resample to 30-minute intervals
    print("\n[2] Resampling to 30-minute intervals...")
    df_30min = df.set_index('timestamp')[['power_usage']].resample('30min').mean()
    df_30min.reset_index(inplace=True)
    print(f"Resampled to {len(df_30min)} records (30-minute intervals)")
    
    # Create features
    print("\n[3] Creating features...")
    df_30min = create_features(df_30min)
    print("Features created:")
    print("  - Time features: hour, minute, day_of_week, month")
    print("  - Cyclical encodings: hour_sin/cos, minute_sin/cos, day_sin/cos")
    print("  - Binary features: is_weekend, is_office_hour, is_holiday")
    print("  - Lag features: 1h, 24h, 1week ago")
    print("  - Rolling statistics: mean and std over 1h and 3h windows")
    
    # Filter data up to 31/3 11:59pm
    cutoff_time = pd.Timestamp('2026-03-31 23:30:00')  # Last 30-min interval of the day
    train_data = df_30min[df_30min['timestamp'] <= cutoff_time].copy()
    
    # Drop rows with NaN
    print(f"\n[4] Data before dropping NaN: {len(train_data)} records")
    train_data = train_data.dropna()
    
    print(f"Using data up to {cutoff_time} (inclusive)...")
    print(f"Training data: {len(train_data)} records (after removing NaN)")
    
    if len(train_data) == 0:
        print("\nERROR: No training data available after removing NaN values!")
        print("This typically happens when lag features require more historical data than available.")
        print("The 1-week lag feature requires at least 7 days of data before the cutoff date.")
        return
    
    print(f"Last training timestamp: {train_data['timestamp'].max()}")
    
    # Prepare features and target
    feature_cols = ['hour', 'minute', 'day_of_week', 'day_of_month', 'month',
                    'is_weekend', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
                    'day_sin', 'day_cos', 'is_office_hour', 'is_holiday', 'time_of_day',
                    'lag_1h', 'lag_24h', 'lag_1week', 
                    'rolling_mean_1h', 'rolling_std_1h', 'rolling_mean_3h']
    
    X_train = train_data[feature_cols]
    y_train = train_data['power_usage']
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    # Build LightGBM model
    print("\n[5] Building LightGBM model...")
    print("Model parameters:")
    print("  - Objective: regression")
    print("  - Boosting type: gbdt")
    print("  - Learning rate: 0.05")
    print("  - Max depth: 8")
    print("  - Num leaves: 64")
    
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
    
    train_dataset = lgb.Dataset(X_train, label=y_train)
    
    # Train model
    print("\n[6] Training LightGBM model...")
    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=500,
        valid_sets=[train_dataset],
        valid_names=['train'],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    print(f"Model training complete! Best iteration: {model.best_iteration}")
    
    # Evaluate on validation set
    val_size = int(len(train_data) * 0.1)
    X_val = X_train.tail(val_size)
    y_val = y_train.tail(val_size)
    
    val_predictions = model.predict(X_val, num_iteration=model.best_iteration)
    evaluate_model(y_val, val_predictions, "Validation Set")
    
    # Create future timestamps for prediction (1/4 00:00 to 7/4 23:30 = 7 days = 336 intervals)
    print("\n[7] Predicting 1/4 00:00 to 7/4 23:30 (336 30-minute intervals = 7 days)...")
    
    prediction_start = pd.Timestamp('2026-04-01 00:00:00')
    prediction_end = pd.Timestamp('2026-04-07 23:30:00')
    
    future_timestamps = pd.date_range(start=prediction_start, end=prediction_end, freq='30min').tolist()
    
    # Iterative prediction
    predictions = []
    last_known_data = df_30min[df_30min['timestamp'] <= cutoff_time].copy()
    
    print("Generating predictions iteratively (using lag features)...")
    print("Processing 336 30-minute intervals...")
    
    for i, future_time in enumerate(future_timestamps):
        temp_df = pd.DataFrame({'timestamp': [future_time]})
        temp_df['hour'] = temp_df['timestamp'].dt.hour
        temp_df['minute'] = temp_df['timestamp'].dt.minute
        temp_df['day_of_week'] = temp_df['timestamp'].dt.dayofweek
        temp_df['day_of_month'] = temp_df['timestamp'].dt.day
        temp_df['month'] = temp_df['timestamp'].dt.month
        temp_df['is_weekend'] = (temp_df['day_of_week'] >= 5).astype(int)
        
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
        
        # Get lag features (adjusted for 30-min intervals: 2 intervals/hour, 48 intervals/day)
        combined_data = pd.concat([last_known_data[['timestamp', 'power_usage']], 
                                   pd.DataFrame({'timestamp': future_timestamps[:i], 
                                                'power_usage': predictions})], 
                                  ignore_index=True)
        
        if len(combined_data) >= 2:  # 1 hour = 2 intervals
            temp_df['lag_1h'] = combined_data.iloc[-2]['power_usage']
        else:
            temp_df['lag_1h'] = last_known_data.iloc[-1]['power_usage']
        
        if len(combined_data) >= 48:  # 24 hours = 48 intervals
            temp_df['lag_24h'] = combined_data.iloc[-48]['power_usage']
        else:
            temp_df['lag_24h'] = last_known_data.iloc[-48]['power_usage'] if len(last_known_data) >= 48 else last_known_data.iloc[0]['power_usage']
        
        if len(combined_data) >= 336:  # 1 week = 336 intervals
            temp_df['lag_1week'] = combined_data.iloc[-336]['power_usage']
        else:
            temp_df['lag_1week'] = last_known_data.iloc[-336]['power_usage'] if len(last_known_data) >= 336 else last_known_data.iloc[0]['power_usage']
        
        recent_values = combined_data.tail(6)['power_usage'].values  # Last 3 hours = 6 intervals
        temp_df['rolling_mean_1h'] = recent_values[-2:].mean() if len(recent_values) >= 2 else recent_values.mean()
        temp_df['rolling_std_1h'] = recent_values[-2:].std() if len(recent_values) >= 2 else recent_values.std()
        temp_df['rolling_mean_3h'] = recent_values.mean()
        
        X_future = temp_df[feature_cols]
        pred = model.predict(X_future, num_iteration=model.best_iteration)[0]
        predictions.append(pred)
        
        if (i + 1) % 48 == 0:  # Every 24 hours (48 intervals)
            days_done = (i + 1) / 48
            print(f"  Predicted {days_done:.0f} days ({i+1}/{len(future_timestamps)} intervals)...")
    
    prediction_df = pd.DataFrame({
        'timestamp': future_timestamps,
        'predicted_power_usage': predictions
    })
    
    print(f"\nGenerated {len(prediction_df)} 30-minute interval predictions")
    print(f"Prediction period: {prediction_df['timestamp'].min()} to {prediction_df['timestamp'].max()}")
    
    # Save predictions
    output_file = 'output/5_predictions_lightgbm_weekly.csv'
    prediction_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Predictions saved to: {output_file}")
    
    # Get historical data from 26/3 onwards
    history_start = pd.Timestamp('2026-03-26 00:00:00')
    historical_data = df_30min[
        (df_30min['timestamp'] >= history_start) & 
        (df_30min['timestamp'] <= cutoff_time)
    ].copy()
    
    print("\n[8] Historical data for plotting...")
    print(f"Period: {historical_data['timestamp'].min()} to {historical_data['timestamp'].max()}")
    print(f"Records: {len(historical_data)}")
    
    # Feature importance
    print("\n[9] Top 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:20s}: {row['importance']:.0f}")
    
    # Visualizations
    print("\n[10] Creating visualizations...")
    
    plt.figure(figsize=(18, 14))
    
    # Plot 1: Full week prediction
    ax1 = plt.subplot(4, 2, 1)
    hist_plot = historical_data.iloc[::2]  # Every hour
    pred_plot = prediction_df.iloc[::2]  # Every hour
    
    ax1.plot(hist_plot['timestamp'], hist_plot['power_usage'], 
             label='Historical (26/3 - 31/3)', color='blue', linewidth=1, alpha=0.7)
    ax1.plot(pred_plot['timestamp'], pred_plot['predicted_power_usage'], 
             label='Predicted (1/4 - 7/4)', color='red', linewidth=1, alpha=0.8)
    ax1.axvline(x=cutoff_time, color='green', linestyle='--', linewidth=2, label='Cutoff (31/3 23:30)')
    ax1.set_title('7-Day Weekly Forecast (1/4 - 7/4)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date & Time')
    ax1.set_ylabel('Power Usage (kW)')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Daily averages
    ax2 = plt.subplot(4, 2, 2)
    prediction_df['date'] = prediction_df['timestamp'].dt.date
    daily_avg = prediction_df.groupby('date')['predicted_power_usage'].mean()
    daily_std = prediction_df.groupby('date')['predicted_power_usage'].std()
    
    dates = [pd.Timestamp(d) for d in daily_avg.index]
    ax2.bar(dates, daily_avg.values, color='steelblue', alpha=0.7, width=0.8)
    ax2.errorbar(dates, daily_avg.values, yerr=daily_std.values, fmt='none', color='black', capsize=5)
    ax2.set_title('Daily Average Power Usage (1/4 - 7/4)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Average Power Usage (kW)')
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Weekday vs Weekend
    ax3 = plt.subplot(4, 2, 3)
    prediction_df['day_of_week'] = prediction_df['timestamp'].dt.dayofweek
    
    weekday_pred = prediction_df[prediction_df['day_of_week'] == 0].head(48)  # Monday (48 intervals)
    weekend_pred = prediction_df[prediction_df['day_of_week'] == 5].head(48)  # Saturday (48 intervals)
    
    if len(weekday_pred) > 0:
        weekday_pred['hour_min'] = weekday_pred['timestamp'].dt.hour + weekday_pred['timestamp'].dt.minute / 60
        ax3.plot(weekday_pred['hour_min'], weekday_pred['predicted_power_usage'], 
                 label='Weekday (Mon 1/4)', color='blue', linewidth=2, alpha=0.8, marker='o', markersize=4)
    
    if len(weekend_pred) > 0:
        weekend_pred['hour_min'] = weekend_pred['timestamp'].dt.hour + weekend_pred['timestamp'].dt.minute / 60
        ax3.plot(weekend_pred['hour_min'], weekend_pred['predicted_power_usage'], 
                 label='Weekend (Sat 6/4)', color='red', linewidth=2, alpha=0.8, marker='s', markersize=4)
    
    ax3.axvspan(9, 18, alpha=0.2, color='green', label='Office Hours')
    ax3.set_title('Weekday vs Weekend Pattern', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Power Usage (kW)')
    ax3.set_xticks(range(0, 25, 3))
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Hourly heatmap
    ax4 = plt.subplot(4, 2, 4)
    prediction_df['hour'] = prediction_df['timestamp'].dt.hour
    heatmap_data = prediction_df.pivot_table(values='predicted_power_usage', 
                                              index='hour', 
                                              columns='date', 
                                              aggfunc='mean')
    
    im = ax4.imshow(heatmap_data.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax4.set_yticks(range(24))
    ax4.set_yticklabels(range(24))
    ax4.set_xticks(range(len(heatmap_data.columns)))
    ax4.set_xticklabels([d.strftime('%m/%d') for d in heatmap_data.columns], rotation=45)
    ax4.set_title('Hourly Power Usage Heatmap', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Hour of Day')
    plt.colorbar(im, ax=ax4, label='Power Usage (kW)')
    
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
    
    # Plot 6: Day-by-day comparison
    ax6 = plt.subplot(4, 2, 6)
    for date in daily_avg.index:
        day_data = prediction_df[prediction_df['date'] == date]
        if len(day_data) > 0:
            day_data['hour_min'] = day_data['timestamp'].dt.hour + day_data['timestamp'].dt.minute / 60
            day_name = pd.Timestamp(date).strftime('%a %d')
            ax6.plot(day_data['hour_min'], day_data['predicted_power_usage'], 
                     label=day_name, linewidth=1.5, alpha=0.7, marker='o', markersize=2)
    
    ax6.set_title('Day-by-Day Comparison', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Hour of Day')
    ax6.set_ylabel('Power Usage (kW)')
    ax6.set_xticks(range(0, 25, 3))
    ax6.legend(fontsize=7, loc='best')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Residuals
    ax7 = plt.subplot(4, 2, 7)
    residuals = y_val - val_predictions
    ax7.scatter(val_predictions, residuals, alpha=0.3, s=1)
    ax7.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax7.set_xlabel('Predicted Power Usage (kW)')
    ax7.set_ylabel('Residuals (kW)')
    ax7.set_title('Residual Plot (Validation Set)', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Summary
    ax8 = plt.subplot(4, 2, 8)
    ax8.axis('off')
    
    summary_text = f"""
WEEKLY FORECAST SUMMARY

Training Period:
  • Up to: 31/3 23:30
  • Records: {len(train_data):,} intervals
  • Features: {len(feature_cols)}

Prediction Period:
  • 1/4 00:00 - 7/4 23:30 (7 days)
  • Records: {len(prediction_df):,} (30-min intervals)
  • Mean: {prediction_df['predicted_power_usage'].mean():.2f} kW
  • Std: {prediction_df['predicted_power_usage'].std():.2f} kW

Model Performance:
  • RMSE: {np.sqrt(mean_squared_error(y_val, val_predictions)):.4f}
  • MAE: {mean_absolute_error(y_val, val_predictions):.4f}
  • R²: {r2_score(y_val, val_predictions):.4f}

Top Features:
  1. {feature_importance.iloc[0]['feature']}
  2. {feature_importance.iloc[1]['feature']}
  3. {feature_importance.iloc[2]['feature']}
    """
    
    ax8.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
             verticalalignment='center', transform=ax8.transAxes)
    ax8.set_title('Model Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('Weekly Power Usage Forecast (1/4 - 7/4) - LightGBM Model', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plot_file = 'output/5_lightgbm_weekly_results.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {plot_file}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("WEEKLY FORECAST SUMMARY")
    print("=" * 80)
    
    print("\nPrediction Period: 1/4 00:00 to 7/4 23:30 (7 full days)")
    print(f"Total predictions: {len(prediction_df):,} 30-minute intervals")
    print(f"Frequency: 48 intervals per day, 336 intervals per week")
    
    print(f"\nPredicted Values:")
    print(f"  Mean: {prediction_df['predicted_power_usage'].mean():.2f} kW")
    print(f"  Std:  {prediction_df['predicted_power_usage'].std():.2f} kW")
    print(f"  Min:  {prediction_df['predicted_power_usage'].min():.2f} kW")
    print(f"  Max:  {prediction_df['predicted_power_usage'].max():.2f} kW")
    
    print("\nDaily Averages:")
    for date, avg in daily_avg.items():
        day_name = pd.Timestamp(date).strftime('%a %d/%m')
        print(f"  {day_name}: {avg:.2f} kW")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    
    plt.show()

if __name__ == "__main__":
    main()
