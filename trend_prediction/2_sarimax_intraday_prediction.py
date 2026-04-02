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

def resample_to_15min(df):
    """Resample minute data to 15-minute intervals"""
    df_15min = df[['power_usage']].resample('15min').mean()
    return df_15min

def create_exogenous_features(df):
    """Create exogenous features for intraday patterns"""
    df_features = pd.DataFrame(index=df.index)
    df_features['is_weekday'] = (df.index.dayofweek < 5).astype(int)
    df_features['hour'] = df.index.hour
    df_features['minute'] = df.index.minute
    df_features['is_office_hour'] = ((df_features['hour'] >= 9) & 
                                      (df_features['hour'] <= 18) & 
                                      (df_features['is_weekday'] == 1)).astype(int)
    return df_features

def main():
    print("=" * 80)
    print("SARIMAX Intraday Prediction (Next 24 Hours)")
    print("=" * 80)
    
    print("\n[1] Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} records from {df.index.min()} to {df.index.max()}")
    
    print("\n[2] Resampling to 15-minute intervals...")
    df_15min = resample_to_15min(df)
    print(f"Resampled to {len(df_15min)} 15-minute records")
    
    print("\n[3] Creating exogenous features...")
    exog_features = create_exogenous_features(df_15min)
    print("Features created:")
    print("  - is_weekday: 1 for Mon-Fri, 0 for Sat-Sun")
    print("  - hour: Hour of day (0-23)")
    print("  - minute: Minute of hour")
    print("  - is_office_hour: 1 for 9AM-6PM on weekdays")
    
    print("\n[4] Splitting data into train/validation sets...")
    train_size = int(len(df_15min) * 0.8)
    train_data = df_15min.iloc[:train_size]
    val_data = df_15min.iloc[train_size:]
    train_exog = exog_features.iloc[:train_size]
    val_exog = exog_features.iloc[train_size:]
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    print("\n[5] Building SARIMAX model for intraday prediction...")
    print("Model parameters: SARIMAX(1,1,1)x(1,1,1,96)")
    print("  - Daily seasonality: 96 intervals (24 hours * 4 per hour)")
    print("  - Exogenous variables: is_weekday, hour, minute, is_office_hour")
    
    model = SARIMAX(train_data['power_usage'],
                    exog=train_exog,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 96),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    
    print("\n[6] Training SARIMAX model...")
    print("This may take a few minutes...")
    fitted_model = model.fit(disp=False, maxiter=200)
    print("Model training complete!")
    
    print("\n[7] Evaluating model on validation set...")
    val_predictions = fitted_model.forecast(steps=len(val_data), exog=val_exog)
    evaluate_model(val_data['power_usage'].values, val_predictions.values, "Validation Set")
    
    print("\n[8] Predicting next 24 hours...")
    
    print("Refitting model on full dataset...")
    full_model = SARIMAX(df_15min['power_usage'],
                         exog=exog_features,
                         order=(1, 1, 1),
                         seasonal_order=(1, 1, 1, 96),
                         enforce_stationarity=False,
                         enforce_invertibility=False)
    full_fitted = full_model.fit(disp=False, maxiter=200)
    
    forecast_steps = 24 * 4
    last_timestamp = df_15min.index[-1]
    future_timestamps = pd.date_range(start=last_timestamp + timedelta(minutes=15),
                                     periods=forecast_steps,
                                     freq='15min')
    
    future_exog = pd.DataFrame(index=future_timestamps)
    future_exog['is_weekday'] = (future_exog.index.dayofweek < 5).astype(int)
    future_exog['hour'] = future_exog.index.hour
    future_exog['minute'] = future_exog.index.minute
    future_exog['is_office_hour'] = ((future_exog['hour'] >= 9) & 
                                      (future_exog['hour'] <= 18) & 
                                      (future_exog['is_weekday'] == 1)).astype(int)
    
    future_predictions = full_fitted.forecast(steps=forecast_steps, exog=future_exog)
    
    prediction_df = pd.DataFrame({
        'timestamp': future_timestamps,
        'predicted_power_usage': future_predictions.values
    })
    
    output_file = 'output/2_predictions_sarimax_intraday.csv'
    prediction_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Predictions saved to: {output_file}")
    
    print("\n[9] Analyzing intraday patterns...")
    prediction_df['hour'] = prediction_df['timestamp'].dt.hour
    
    hourly_avg = prediction_df.groupby('hour')['predicted_power_usage'].mean()
    
    print("\nHourly Average Predictions:")
    for hour, avg in hourly_avg.items():
        print(f"  {hour:02d}:00 - {avg:.2f} kW")
    
    print("\n[10] Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    ax1 = axes[0, 0]
    ax1.plot(prediction_df['timestamp'], prediction_df['predicted_power_usage'],
             'o-', color='blue', alpha=0.7, markersize=2)
    ax1.set_title('SARIMAX Intraday Predictions (Next 24 Hours)', fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Power Usage (kW)')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    ax2 = axes[0, 1]
    ax2.plot(hourly_avg.index, hourly_avg.values, 'o-', 
             color='blue', linewidth=2, markersize=8)
    ax2.axvspan(9, 18, alpha=0.1, color='green', label='Office Hours')
    ax2.set_title('Average Hourly Pattern', fontweight='bold')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Average Power Usage (kW)')
    ax2.set_xticks(range(0, 24, 2))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    last_24h = df_15min.tail(24 * 4)
    ax3.plot(last_24h.index, last_24h['power_usage'], 
             label='Historical (Last 24h)', alpha=0.7, linewidth=2, color='green')
    ax3.plot(prediction_df['timestamp'], prediction_df['predicted_power_usage'], 
             label='Predicted (Next 24h)', alpha=0.7, linewidth=2, color='red')
    ax3.axvline(x=df_15min.index[-1], color='black', linestyle='--', 
                linewidth=2, label='Prediction Start')
    ax3.set_title('Historical vs Predicted', fontweight='bold')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Power Usage (kW)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    ax4 = axes[1, 1]
    ax4.plot(val_data.index[-96:], val_data['power_usage'].values[-96:], 
             label='Actual', alpha=0.7, linewidth=2)
    ax4.plot(val_data.index[-96:], val_predictions.values[-96:], 
             label='Predicted', alpha=0.7, linewidth=2)
    ax4.set_title('Validation Set Performance (Last 24h)', fontweight='bold')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Power Usage (kW)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plot_file = 'output/2_sarimax_intraday_results.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {plot_file}")
    
    print("\n" + "=" * 80)
    print("SARIMAX INTRADAY PREDICTION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
