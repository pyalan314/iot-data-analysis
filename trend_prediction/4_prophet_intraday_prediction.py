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

def resample_to_15min(df):
    """Resample minute data to 15-minute intervals"""
    df_15min = df.set_index('timestamp')[['power_usage']].resample('15min').mean()
    df_15min.reset_index(inplace=True)
    return df_15min

def prepare_for_prophet(df):
    """Prepare data in Prophet format (ds, y columns)"""
    prophet_df = df[['timestamp', 'power_usage']].copy()
    prophet_df.columns = ['ds', 'y']
    return prophet_df

def main():
    print("=" * 80)
    print("Intraday Power Usage Prediction using Prophet (15-minute intervals)")
    print("Scenario: Predict 31/3 12pm to 11:59pm using data up to 31/3 12pm")
    print("=" * 80)
    
    # Load data
    print("\n[1] Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Resample to 15-minute intervals
    print("\n[2] Resampling to 15-minute intervals...")
    df_15min = resample_to_15min(df)
    print(f"Resampled to {len(df_15min)} 15-minute records")
    
    # Filter data up to 31/3 12pm (noon) - include 12pm in training data
    cutoff_time = pd.Timestamp('2026-03-31 12:00:00')
    train_data = df_15min[df_15min['timestamp'] <= cutoff_time].copy()
    
    print(f"\n[3] Using data up to {cutoff_time} (inclusive)...")
    print(f"Training data: {len(train_data)} records")
    print(f"Last training timestamp: {train_data['timestamp'].max()}")
    
    # Prepare data for Prophet
    prophet_df = prepare_for_prophet(train_data)
    
    # Build Prophet model with increased seasonality strength
    print("\n[4] Building Prophet model...")
    print("Model configuration:")
    print("  - Daily seasonality: Enabled (captures hourly patterns)")
    print("  - Weekly seasonality: Enabled (captures weekday/weekend patterns)")
    print("  - Seasonality prior scale: 20.0 (stronger seasonality)")
    print("  - Holidays: HK public holidays included")
    
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        holidays=HK_HOLIDAYS_2026,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=20.0,  # Increased to preserve weekday/weekend contrast
        seasonality_mode='additive',  # Additive for smoother transitions
        interval_width=0.95
    )
    
    # Train model
    print("\n[5] Training Prophet model...")
    model.fit(prophet_df)
    print("Model training complete!")
    
    # Create future dataframe for prediction (12pm to 11:59pm = 12 hours = 48 15-min intervals)
    print("\n[6] Predicting 31/3 12:15pm to 11:59pm (48 15-minute intervals)...")
    future = model.make_future_dataframe(periods=48, freq='15min')
    
    # Make predictions
    forecast = model.predict(future)
    
    # Extract predictions for 1pm to 11:59pm on 31/3 (start from next hour after cutoff)
    prediction_start = pd.Timestamp('2026-03-31 13:00:00')  # 1pm (next hour after 12pm)
    prediction_end = pd.Timestamp('2026-03-31 23:59:59')    # 11:59pm
    
    future_predictions = forecast[
        (forecast['ds'] >= prediction_start) & 
        (forecast['ds'] <= prediction_end)
    ].copy()
    
    # Also get the 12pm forecast value for smooth connection
    cutoff_forecast = forecast[forecast['ds'] == cutoff_time].copy()
    
    # Create prediction DataFrame - include 12pm point for smooth connection
    if len(cutoff_forecast) > 0:
        # Prepend the 12pm forecast to create smooth connection
        all_predictions = pd.concat([cutoff_forecast, future_predictions], ignore_index=True)
    else:
        all_predictions = future_predictions
    
    prediction_df = pd.DataFrame({
        'timestamp': all_predictions['ds'],
        'predicted_power_usage': all_predictions['yhat'],
        'lower_bound': all_predictions['yhat_lower'],
        'upper_bound': all_predictions['yhat_upper']
    })
    
    print(f"Generated {len(prediction_df)} 15-minute interval predictions (including 12pm connection point)")
    print(f"Prediction period: {prediction_df['timestamp'].min()} to {prediction_df['timestamp'].max()}")
    
    # Save predictions
    output_file = 'output/4_predictions_prophet_intraday.csv'
    prediction_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Predictions saved to: {output_file}")
    
    # Get historical data from 26/3 onwards
    history_start = pd.Timestamp('2026-03-26 00:00:00')
    historical_data = df_15min[
        (df_15min['timestamp'] >= history_start) & 
        (df_15min['timestamp'] <= cutoff_time)
    ].copy()
    
    print("\n[7] Historical data for plotting...")
    print(f"Period: {historical_data['timestamp'].min()} to {historical_data['timestamp'].max()}")
    print(f"Records: {len(historical_data)}")
    
    # Visualizations
    print("\n[8] Creating visualizations...")
    
    plt.figure(figsize=(20, 14))
    
    # Plot 1: Main timeline - Historical (26/3-31/3 12pm) + Predictions (31/3 12pm-7pm)
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(historical_data['timestamp'], historical_data['power_usage'], 
             label='Historical (26/3 - 31/3 12pm)', color='blue', linewidth=2, alpha=0.8)
    ax1.plot(prediction_df['timestamp'], prediction_df['predicted_power_usage'], 
             label='Predicted (31/3 1pm - 11:59pm)', color='red', linewidth=2, alpha=0.8, marker='o', markersize=4)
    ax1.fill_between(prediction_df['timestamp'],
                     prediction_df['lower_bound'],
                     prediction_df['upper_bound'],
                     alpha=0.2, color='red', label='95% Confidence Interval')
    ax1.axvline(x=cutoff_time, color='green', linestyle='--', 
                linewidth=2, label='Prediction Start (31/3 12pm)')
    ax1.set_title('Historical Data + Intraday Predictions', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date & Time')
    ax1.set_ylabel('Power Usage (kW)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Zoom into 31/3 only
    ax2 = plt.subplot(3, 2, 2)
    march31_hist = historical_data[historical_data['timestamp'].dt.date == pd.Timestamp('2026-03-31').date()]
    
    ax2.plot(march31_hist['timestamp'], march31_hist['power_usage'], 
             'o-', label='Historical (31/3 00:00-12:00)', color='blue', 
             linewidth=2, markersize=6, alpha=0.8)
    ax2.plot(prediction_df['timestamp'], prediction_df['predicted_power_usage'], 
             'o-', label='Predicted (31/3 12:00-23:59)', color='red', 
             linewidth=2, markersize=5, alpha=0.8)
    ax2.fill_between(prediction_df['timestamp'],
                     prediction_df['lower_bound'],
                     prediction_df['upper_bound'],
                     alpha=0.2, color='red')
    ax2.axvline(x=cutoff_time, color='green', linestyle='--', 
                linewidth=2, label='Cutoff (12pm)')
    ax2.set_title('Focus: March 31 (Full Day)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Power Usage (kW)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Daily seasonality component
    ax3 = plt.subplot(3, 2, 3)
    daily_data = forecast[['ds', 'daily']].copy()
    daily_data['hour'] = daily_data['ds'].dt.hour
    daily_avg = daily_data.groupby('hour')['daily'].mean()
    
    ax3.plot(range(24), daily_avg.values, marker='o', linewidth=2, color='orange', markersize=8)
    ax3.axvspan(9, 18, alpha=0.2, color='green', label='Office Hours')
    ax3.axvline(x=12, color='red', linestyle='--', linewidth=2, label='Prediction Start (12pm)')
    ax3.axvspan(13, 24, alpha=0.15, color='red', label='Prediction Window')
    ax3.set_title('Daily Seasonality Pattern (Hourly Effect)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Seasonal Effect (kW)')
    ax3.set_xticks(range(0, 24, 2))
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    
    # Plot 4: Weekly seasonality component
    ax4 = plt.subplot(3, 2, 4)
    weekly_data = forecast[['ds', 'weekly']].copy()
    weekly_data['day_of_week'] = weekly_data['ds'].dt.dayofweek
    weekly_avg = weekly_data.groupby('day_of_week')['weekly'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    colors = ['blue', 'blue', 'blue', 'blue', 'blue', 'red', 'red']
    ax4.bar(range(7), weekly_avg.values, color=colors, alpha=0.7)
    ax4.set_title('Weekly Seasonality Pattern (Day of Week Effect)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Day of Week')
    ax4.set_ylabel('Seasonal Effect (kW)')
    ax4.set_xticks(range(7))
    ax4.set_xticklabels(days)
    ax4.grid(True, axis='y', alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    
    # Add annotation for Monday (31/3)
    ax4.annotate('31/3 is Monday', xy=(0, weekly_avg.values[0]), 
                xytext=(1, weekly_avg.values[0] + 2),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, color='green', fontweight='bold')
    
    # Plot 5: Hourly comparison for last 3 days
    ax5 = plt.subplot(3, 2, 5)
    last_3_days = historical_data.tail(72)  # Last 3 days (72 hours)
    
    ax5.plot(last_3_days['timestamp'], last_3_days['power_usage'], 
             label='Historical (Last 3 days)', color='blue', linewidth=2, alpha=0.7)
    ax5.plot(prediction_df['timestamp'], prediction_df['predicted_power_usage'], 
             label='Predicted (31/3 12pm-midnight)', color='red', linewidth=2, alpha=0.7, marker='o', markersize=3)
    ax5.fill_between(prediction_df['timestamp'],
                     prediction_df['lower_bound'],
                     prediction_df['upper_bound'],
                     alpha=0.2, color='red')
    ax5.axvline(x=cutoff_time, color='green', linestyle='--', 
                linewidth=2, label='Cutoff')
    ax5.set_title('Last 3 Days + Predictions', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Date & Time')
    ax5.set_ylabel('Power Usage (kW)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis='x', rotation=45)
    
    # Plot 6: Prediction details table (show first 10 and last 10 predictions)
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    
    # Create table data - show first 10 and last 10 predictions
    table_data = []
    table_data.append(['Time', 'Predicted (kW)', 'Lower (kW)', 'Upper (kW)'])
    
    # First 10 predictions
    for _, row in prediction_df.head(10).iterrows():
        time_str = row['timestamp'].strftime('%H:%M')
        table_data.append([
            time_str,
            f"{row['predicted_power_usage']:.2f}",
            f"{row['lower_bound']:.2f}",
            f"{row['upper_bound']:.2f}"
        ])
    
    # Add separator row
    table_data.append(['...', '...', '...', '...'])
    
    # Last 10 predictions
    for _, row in prediction_df.tail(10).iterrows():
        time_str = row['timestamp'].strftime('%H:%M')
        table_data.append([
            time_str,
            f"{row['predicted_power_usage']:.2f}",
            f"{row['lower_bound']:.2f}",
            f"{row['upper_bound']:.2f}"
        ])
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.3, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style separator row
    for j in range(4):
        table[(11, j)].set_facecolor('#CCCCCC')
        table[(11, j)].set_text_props(style='italic')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        if i != 11:  # Skip separator row
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E7E6E6')
    
    ax6.set_title('Prediction Details (First & Last 10 of 48 intervals)', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('Intraday Power Usage Prediction - Prophet Model (15-min intervals)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98], h_pad=3.0, w_pad=2.0)
    
    plot_file = 'output/4_prophet_intraday_results.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {plot_file}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)
    
    print("\nScenario:")
    print(f"  - Training data: Up to {cutoff_time}")
    print(f"  - Prediction window: {prediction_df['timestamp'].min()} to {prediction_df['timestamp'].max()}")
    print("  - Day: Monday, March 31, 2026")
    print("  - Time: 1pm to 11:59pm (afternoon to midnight)")
    
    print("\nHistorical Context (26/3 - 31/3 12pm):")
    print(f"  Mean: {historical_data['power_usage'].mean():.2f} kW")
    print(f"  Std:  {historical_data['power_usage'].std():.2f} kW")
    print(f"  Min:  {historical_data['power_usage'].min():.2f} kW")
    print(f"  Max:  {historical_data['power_usage'].max():.2f} kW")
    
    print(f"\nPredicted Values (31/3 1pm-11:59pm):")
    print(f"  Mean: {prediction_df['predicted_power_usage'].mean():.2f} kW")
    print(f"  Std:  {prediction_df['predicted_power_usage'].std():.2f} kW")
    print(f"  Min:  {prediction_df['predicted_power_usage'].min():.2f} kW")
    print(f"  Max:  {prediction_df['predicted_power_usage'].max():.2f} kW")
    
    print("\n15-Minute Interval Predictions:")
    for _, row in prediction_df.iterrows():
        time_str = row['timestamp'].strftime('%Y-%m-%d %H:%M')
        print(f"  {time_str}: {row['predicted_power_usage']:.2f} kW "
              f"(95% CI: {row['lower_bound']:.2f} - {row['upper_bound']:.2f})")
    
    print("\nContext:")
    print("  - Monday is a weekday (higher usage expected)")
    print("  - 1pm-6pm covers office hours, 6pm-midnight covers evening/night")
    print("  - Expect high usage during office hours, then gradual decrease to night minimum")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    
    plt.show()

if __name__ == "__main__":
    main()
