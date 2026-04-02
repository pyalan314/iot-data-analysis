import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load power usage data"""
    df = pd.read_csv('power_usage_data.csv', encoding='utf-8')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
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

def resample_to_hourly(df):
    """Resample minute data to hourly"""
    df_hourly = df[['power_usage']].resample('h').mean()
    return df_hourly

def create_exogenous_features(df):
    """Create exogenous features for weekday/weekend effect"""
    df_features = pd.DataFrame(index=df.index)
    df_features['is_weekday'] = (df.index.dayofweek < 5).astype(int)
    df_features['hour'] = df.index.hour
    df_features['is_office_hour'] = ((df_features['hour'] >= 9) & 
                                      (df_features['hour'] <= 18) & 
                                      (df_features['is_weekday'] == 1)).astype(int)
    return df_features

def main():
    print("=" * 80)
    print("SARIMAX Weekly Prediction (Apr 1-7, 2026)")
    print("=" * 80)
    
    print("\n[1] Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} records from {df.index.min()} to {df.index.max()}")
    
    print("\n[2] Resampling to hourly data...")
    df_hourly = resample_to_hourly(df)
    print(f"Resampled to {len(df_hourly)} hourly records")
    
    print("\n[3] Creating exogenous features...")
    exog_features = create_exogenous_features(df_hourly)
    print("Features created:")
    print("  - is_weekday: 1 for Mon-Fri, 0 for Sat-Sun")
    print("  - hour: Hour of day (0-23)")
    print("  - is_office_hour: 1 for 9AM-6PM on weekdays")
    
    print("\n[4] Splitting data into train/validation sets...")
    train_size = int(len(df_hourly) * 0.8)
    train_data = df_hourly.iloc[:train_size]
    val_data = df_hourly.iloc[train_size:]
    train_exog = exog_features.iloc[:train_size]
    val_exog = exog_features.iloc[train_size:]
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    print("\n[5] Building SARIMAX model with exogenous variables...")
    print("Model parameters: SARIMAX(1,1,1)x(1,1,1,168)")
    print("  - Weekly seasonality: 168 hours")
    print("  - Exogenous variables: is_weekday, hour, is_office_hour")
    
    model = SARIMAX(train_data['power_usage'],
                    exog=train_exog,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 168),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    
    print("\n[6] Training SARIMAX model...")
    print("This may take a few minutes...")
    fitted_model = model.fit(disp=False, maxiter=200)
    print("Model training complete!")
    
    print("\n[7] Evaluating model on validation set...")
    val_predictions = fitted_model.forecast(steps=len(val_data), exog=val_exog)
    evaluate_model(val_data['power_usage'].values, val_predictions.values, "Validation Set")
    
    print("\n[8] Predicting future power usage (Apr 1-7, 2026)...")
    
    print("Refitting model on full dataset...")
    full_model = SARIMAX(df_hourly['power_usage'],
                         exog=exog_features,
                         order=(1, 1, 1),
                         seasonal_order=(1, 1, 1, 168),
                         enforce_stationarity=False,
                         enforce_invertibility=False)
    full_fitted = full_model.fit(disp=False, maxiter=200)
    
    forecast_steps = 24 * 7
    last_timestamp = df_hourly.index[-1]
    future_timestamps = pd.date_range(start=last_timestamp + timedelta(hours=1),
                                     periods=forecast_steps,
                                     freq='h')
    
    future_exog = pd.DataFrame(index=future_timestamps)
    future_exog['is_weekday'] = (future_exog.index.dayofweek < 5).astype(int)
    future_exog['hour'] = future_exog.index.hour
    future_exog['is_office_hour'] = ((future_exog['hour'] >= 9) & 
                                      (future_exog['hour'] <= 18) & 
                                      (future_exog['is_weekday'] == 1)).astype(int)
    
    future_predictions = full_fitted.forecast(steps=forecast_steps, exog=future_exog)
    
    prediction_df = pd.DataFrame({
        'timestamp': future_timestamps,
        'predicted_power_usage': future_predictions.values
    })
    
    output_file = 'output/1_predictions_sarimax_weekly.csv'
    prediction_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Predictions saved to: {output_file}")
    
    print("\n[9] Analyzing weekday/weekend patterns...")
    prediction_df['is_weekend'] = prediction_df['timestamp'].dt.dayofweek >= 5
    
    weekday_pred = prediction_df[~prediction_df['is_weekend']]['predicted_power_usage']
    weekend_pred = prediction_df[prediction_df['is_weekend']]['predicted_power_usage']
    
    print("\nPredicted Patterns:")
    print(f"  Weekday - Mean: {weekday_pred.mean():.2f} kW, Std: {weekday_pred.std():.2f} kW")
    print(f"  Weekend - Mean: {weekend_pred.mean():.2f} kW, Std: {weekend_pred.std():.2f} kW")
    print(f"  Weekday/Weekend Ratio: {weekday_pred.mean() / weekend_pred.mean():.2f}x")
    print(f"  Difference: {weekday_pred.mean() - weekend_pred.mean():.2f} kW")
    
    print("\n[10] Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    ax1 = axes[0, 0]
    weekday_mask = ~prediction_df['is_weekend']
    weekend_mask = prediction_df['is_weekend']
    
    ax1.plot(prediction_df.loc[weekday_mask, 'timestamp'], 
             prediction_df.loc[weekday_mask, 'predicted_power_usage'],
             'o-', label='Weekday', color='blue', alpha=0.7, markersize=3)
    ax1.plot(prediction_df.loc[weekend_mask, 'timestamp'], 
             prediction_df.loc[weekend_mask, 'predicted_power_usage'],
             's-', label='Weekend', color='red', alpha=0.7, markersize=3)
    ax1.set_title('SARIMAX Weekly Predictions (Apr 1-7)', fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Power Usage (kW)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    ax2 = axes[0, 1]
    prediction_df['hour'] = prediction_df['timestamp'].dt.hour
    
    weekday_hourly = prediction_df[~prediction_df['is_weekend']].groupby('hour')['predicted_power_usage'].mean()
    weekend_hourly = prediction_df[prediction_df['is_weekend']].groupby('hour')['predicted_power_usage'].mean()
    
    ax2.plot(weekday_hourly.index, weekday_hourly.values, 'o-', 
             label='Weekday', color='blue', linewidth=2, markersize=6)
    ax2.plot(weekend_hourly.index, weekend_hourly.values, 's-', 
             label='Weekend', color='red', linewidth=2, markersize=6)
    ax2.axvspan(9, 18, alpha=0.1, color='green', label='Office Hours')
    ax2.set_title('Hourly Pattern: Weekday vs Weekend', fontweight='bold')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Average Power Usage (kW)')
    ax2.set_xticks(range(0, 24, 2))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.plot(val_data.index, val_data['power_usage'], label='Actual', alpha=0.7, linewidth=2)
    ax3.plot(val_data.index, val_predictions, label='Predicted', alpha=0.7, linewidth=2)
    ax3.set_title('Validation Set Performance', fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Power Usage (kW)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    ax4 = axes[1, 1]
    prediction_df['date'] = prediction_df['timestamp'].dt.date
    daily_avg = prediction_df.groupby('date')['predicted_power_usage'].mean()
    
    ax4.bar(range(len(daily_avg)), daily_avg.values, alpha=0.7, color='steelblue')
    ax4.set_title('Predicted Daily Average Power Usage', fontweight='bold')
    ax4.set_xlabel('Day')
    ax4.set_ylabel('Average Power Usage (kW)')
    ax4.set_xticks(range(len(daily_avg)))
    ax4.set_xticklabels([pd.Timestamp(d).strftime('%m-%d\n%a') for d in daily_avg.index], rotation=0)
    ax4.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_file = 'output/1_sarimax_weekly_results.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {plot_file}")
    
    print("\n" + "=" * 80)
    print("SARIMAX WEEKLY PREDICTION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
