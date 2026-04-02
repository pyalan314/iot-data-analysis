import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# HK Public Holidays 2026
HK_HOLIDAYS_2026 = pd.DataFrame({
    'holiday': 'hk_holiday',
    'ds': pd.to_datetime([
        '2026-01-01',  # New Year's Day
        '2026-01-29',  # Lunar New Year Day 1
        '2026-01-30',  # Lunar New Year Day 2
        '2026-01-31',  # Lunar New Year Day 3
        '2026-02-02',  # Lunar New Year Day 4 (in lieu)
        '2026-04-03',  # Ching Ming Festival
        '2026-04-06',  # Easter Monday
    ]),
    'lower_window': 0,
    'upper_window': 0,
})

def load_data():
    """Load power usage data"""
    df = pd.read_csv('power_usage_data.csv', encoding='utf-8')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def prepare_for_prophet(df):
    """Prepare data in Prophet format (ds, y columns)"""
    prophet_df = df[['timestamp', 'power_usage']].copy()
    prophet_df.columns = ['ds', 'y']
    return prophet_df

def main():
    print("=" * 80)
    print("Weekly Power Usage Prediction using Prophet (Per-30-Minute Data)")
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
    
    # Filter data up to 31/3 11:59pm
    cutoff_time = pd.Timestamp('2026-03-31 23:30:00')
    train_data = df_30min[df_30min['timestamp'] <= cutoff_time].copy()
    
    print(f"\n[3] Using data up to {cutoff_time} (inclusive)...")
    print(f"Training data: {len(train_data)} records")
    print(f"Last training timestamp: {train_data['timestamp'].max()}")
    
    # Prepare data for Prophet
    prophet_df = prepare_for_prophet(train_data)
    
    # Build Prophet model
    print("\n[4] Building Prophet model...")
    print("Model configuration:")
    print("  - Daily seasonality: Enabled (captures hourly patterns)")
    print("  - Weekly seasonality: Enabled (captures weekday/weekend patterns)")
    print("  - Seasonality prior scale: 20.0 (stronger seasonality)")
    print("  - Holidays: HK public holidays included")
    print("  - Data: 30-minute intervals")
    
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        holidays=HK_HOLIDAYS_2026,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=20.0,
        seasonality_mode='additive',
        interval_width=0.95
    )
    
    # Train model
    print("\n[5] Training Prophet model...")
    model.fit(prophet_df)
    print("Model training complete!")
    
    # Create future dataframe for prediction (7 days = 336 30-minute intervals)
    print("\n[6] Predicting 1/4 00:00 to 7/4 23:30 (336 30-minute intervals = 7 days)...")
    future = model.make_future_dataframe(periods=336, freq='30min')
    
    # Make predictions
    forecast = model.predict(future)
    
    # Extract predictions for 1/4 to 7/4
    prediction_start = pd.Timestamp('2026-04-01 00:00:00')
    prediction_end = pd.Timestamp('2026-04-07 23:30:00')
    
    future_predictions = forecast[
        (forecast['ds'] >= prediction_start) & 
        (forecast['ds'] <= prediction_end)
    ].copy()
    
    print(f"Generated {len(future_predictions)} 30-minute interval predictions")
    print(f"Prediction period: {future_predictions['ds'].min()} to {future_predictions['ds'].max()}")
    
    # Create prediction DataFrame
    prediction_df = pd.DataFrame({
        'timestamp': future_predictions['ds'],
        'predicted_power_usage': future_predictions['yhat'],
        'lower_bound': future_predictions['yhat_lower'],
        'upper_bound': future_predictions['yhat_upper']
    })
    
    # Save predictions
    output_file = 'output/3_predictions_prophet_weekly.csv'
    prediction_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Predictions saved to: {output_file}")
    
    # Get historical data from 26/3 onwards
    history_start = pd.Timestamp('2026-03-26 00:00:00')
    historical_data = df_30min[
        (df_30min['timestamp'] >= history_start) & 
        (df_30min['timestamp'] <= cutoff_time)
    ].copy()
    
    print("\n[7] Historical data for plotting...")
    print(f"Period: {historical_data['timestamp'].min()} to {historical_data['timestamp'].max()}")
    print(f"Records: {len(historical_data)}")
    
    # Visualizations
    print("\n[8] Creating visualizations...")
    
    plt.figure(figsize=(18, 14))
    
    # Plot 1: Full week prediction
    ax1 = plt.subplot(4, 2, 1)
    hist_plot = historical_data.iloc[::2]  # Every hour
    pred_plot = prediction_df.iloc[::2]  # Every hour
    
    ax1.plot(hist_plot['timestamp'], hist_plot['power_usage'], 
             label='Historical (26/3 - 31/3)', color='blue', linewidth=1, alpha=0.7)
    ax1.plot(pred_plot['timestamp'], pred_plot['predicted_power_usage'], 
             label='Predicted (1/4 - 7/4)', color='red', linewidth=1, alpha=0.8)
    ax1.fill_between(pred_plot['timestamp'],
                     pred_plot['lower_bound'],
                     pred_plot['upper_bound'],
                     alpha=0.2, color='red', label='95% Confidence Interval')
    ax1.axvline(x=cutoff_time, color='green', linestyle='--', linewidth=2, label='Cutoff (31/3 23:30)')
    ax1.set_title('7-Day Weekly Forecast (1/4 - 7/4) - Prophet', fontsize=12, fontweight='bold')
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
    
    weekday_pred = prediction_df[prediction_df['day_of_week'] == 0].head(48)  # Monday
    weekend_pred = prediction_df[prediction_df['day_of_week'] == 5].head(48)  # Saturday
    
    if len(weekday_pred) > 0:
        weekday_pred['hour_min'] = weekday_pred['timestamp'].dt.hour + weekday_pred['timestamp'].dt.minute / 60
        ax3.plot(weekday_pred['hour_min'], weekday_pred['predicted_power_usage'], 
                 label='Weekday (Mon 1/4)', color='blue', linewidth=2, alpha=0.8, marker='o', markersize=4)
        ax3.fill_between(weekday_pred['hour_min'],
                         weekday_pred['lower_bound'],
                         weekday_pred['upper_bound'],
                         alpha=0.2, color='blue')
    
    if len(weekend_pred) > 0:
        weekend_pred['hour_min'] = weekend_pred['timestamp'].dt.hour + weekend_pred['timestamp'].dt.minute / 60
        ax3.plot(weekend_pred['hour_min'], weekend_pred['predicted_power_usage'], 
                 label='Weekend (Sat 6/4)', color='red', linewidth=2, alpha=0.8, marker='s', markersize=4)
        ax3.fill_between(weekend_pred['hour_min'],
                         weekend_pred['lower_bound'],
                         weekend_pred['upper_bound'],
                         alpha=0.2, color='red')
    
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
    
    # Plot 5: Daily seasonality component
    ax5 = plt.subplot(4, 2, 5)
    daily_data = forecast[['ds', 'daily']].copy()
    daily_data['hour'] = daily_data['ds'].dt.hour
    daily_avg_component = daily_data.groupby('hour')['daily'].mean()
    
    ax5.plot(range(24), daily_avg_component.values, marker='o', linewidth=2, color='orange', markersize=8)
    ax5.axvspan(9, 18, alpha=0.2, color='green', label='Office Hours')
    ax5.set_title('Daily Seasonality Component', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Hour of Day')
    ax5.set_ylabel('Seasonal Effect (kW)')
    ax5.set_xticks(range(0, 24, 2))
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    
    # Plot 6: Weekly seasonality component
    ax6 = plt.subplot(4, 2, 6)
    weekly_data = forecast[['ds', 'weekly']].copy()
    weekly_data['day_of_week'] = weekly_data['ds'].dt.dayofweek
    weekly_avg = weekly_data.groupby('day_of_week')['weekly'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    colors = ['blue', 'blue', 'blue', 'blue', 'blue', 'red', 'red']
    ax6.bar(range(7), weekly_avg.values, color=colors, alpha=0.7)
    ax6.set_title('Weekly Seasonality Component', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Day of Week')
    ax6.set_ylabel('Seasonal Effect (kW)')
    ax6.set_xticks(range(7))
    ax6.set_xticklabels(days)
    ax6.grid(True, axis='y', alpha=0.3)
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    
    # Plot 7: Day-by-day comparison
    ax7 = plt.subplot(4, 2, 7)
    for date in daily_avg.index:
        day_data = prediction_df[prediction_df['date'] == date]
        if len(day_data) > 0:
            day_data['hour_min'] = day_data['timestamp'].dt.hour + day_data['timestamp'].dt.minute / 60
            day_name = pd.Timestamp(date).strftime('%a %d')
            ax7.plot(day_data['hour_min'], day_data['predicted_power_usage'], 
                     label=day_name, linewidth=1.5, alpha=0.7, marker='o', markersize=2)
    
    ax7.set_title('Day-by-Day Comparison', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Hour of Day')
    ax7.set_ylabel('Power Usage (kW)')
    ax7.set_xticks(range(0, 25, 3))
    ax7.legend(fontsize=7, loc='best')
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Summary
    ax8 = plt.subplot(4, 2, 8)
    ax8.axis('off')
    
    summary_text = f"""
PROPHET WEEKLY FORECAST

Training Period:
  • Up to: 31/3 23:30
  • Records: {len(train_data):,} intervals
  • Frequency: 30-minute intervals

Prediction Period:
  • 1/4 00:00 - 7/4 23:30 (7 days)
  • Records: {len(prediction_df):,} intervals
  • Mean: {prediction_df['predicted_power_usage'].mean():.2f} kW
  • Std: {prediction_df['predicted_power_usage'].std():.2f} kW
  • Range: {prediction_df['predicted_power_usage'].min():.2f} - {prediction_df['predicted_power_usage'].max():.2f} kW

Model Features:
  • Daily seasonality: Enabled
  • Weekly seasonality: Enabled
  • Holidays: HK public holidays
  • Seasonality prior: 20.0
  • 95% Confidence intervals

Daily Averages:
  Mon: {daily_avg.iloc[0]:.2f} kW
  Tue: {daily_avg.iloc[1]:.2f} kW
  Wed: {daily_avg.iloc[2]:.2f} kW
  Thu: {daily_avg.iloc[3]:.2f} kW
  Fri: {daily_avg.iloc[4]:.2f} kW
  Sat: {daily_avg.iloc[5]:.2f} kW
  Sun: {daily_avg.iloc[6]:.2f} kW
    """
    
    ax8.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
             verticalalignment='center', transform=ax8.transAxes)
    ax8.set_title('Forecast Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('Weekly Power Usage Forecast (1/4 - 7/4) - Prophet Model', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plot_file = 'output/3_prophet_weekly_results.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {plot_file}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("PROPHET WEEKLY FORECAST SUMMARY")
    print("=" * 80)
    
    print("\nPrediction Period: 1/4 00:00 to 7/4 23:30 (7 full days)")
    print(f"Total predictions: {len(prediction_df):,} 30-minute intervals")
    print("Frequency: 48 intervals per day, 336 intervals per week")
    
    print(f"\nPredicted Values:")
    print(f"  Mean: {prediction_df['predicted_power_usage'].mean():.2f} kW")
    print(f"  Std:  {prediction_df['predicted_power_usage'].std():.2f} kW")
    print(f"  Min:  {prediction_df['predicted_power_usage'].min():.2f} kW")
    print(f"  Max:  {prediction_df['predicted_power_usage'].max():.2f} kW")
    
    print("\nDaily Averages:")
    for date, avg in daily_avg.items():
        day_name = pd.Timestamp(date).strftime('%a %d/%m')
        print(f"  {day_name}: {avg:.2f} kW")
    
    print("\nConfidence Interval Width:")
    prediction_df['ci_width'] = prediction_df['upper_bound'] - prediction_df['lower_bound']
    print(f"  Average: {prediction_df['ci_width'].mean():.2f} kW")
    print(f"  Min: {prediction_df['ci_width'].min():.2f} kW")
    print(f"  Max: {prediction_df['ci_width'].max():.2f} kW")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    
    plt.show()

if __name__ == "__main__":
    main()
