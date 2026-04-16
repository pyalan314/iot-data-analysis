import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import DLinear
from neuralforecast.losses.pytorch import MAE

df = pd.read_csv('data/office_power_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

df_nf = df.rename(columns={
    'timestamp': 'ds',
    'power_kw': 'y'
})
df_nf['unique_id'] = 'office_tower'
df_nf = df_nf[['unique_id', 'ds', 'y']]

horizon = 28
input_size = 672

models = [
    DLinear(
        h=horizon,
        input_size=input_size,
        max_steps=1000,
        learning_rate=0.001,
        batch_size=32,
        windows_batch_size=128,
        loss=MAE(),
        random_seed=42,
        scaler_type='robust'
    )
]

nf = NeuralForecast(
    models=models,
    freq='15min'
)

print("Starting training...")
print(f"Training data shape: {df_nf.shape}")
print(f"Date range: {df_nf['ds'].min()} to {df_nf['ds'].max()}")
print(f"Horizon: {horizon} steps (7 hours)")
print(f"Input size: {input_size} steps (7 days)")

nf.fit(df=df_nf)

nf.save(
    path='model/',
    model_index=None,
    overwrite=True,
    save_dataset=False
)

print("\nTraining completed!")
print("Model saved to: model/")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

last_30_days = df_nf[df_nf['ds'] >= df_nf['ds'].max() - pd.Timedelta(days=30)]
ax1.plot(last_30_days['ds'], last_30_days['y'], linewidth=1, alpha=0.7)
ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('Power (kW)', fontsize=11)
ax1.set_title('Training Data - Last 30 Days', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

ax2.hist(df_nf['y'], bins=50, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Power (kW)', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Power Distribution - Full Dataset', fontsize=13, fontweight='bold')
ax2.axvline(df_nf['y'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df_nf["y"].mean():.2f} kW')
ax2.axvline(df_nf['y'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df_nf["y"].median():.2f} kW')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('output/training_overview.png', dpi=300, bbox_inches='tight')
print("Training visualization saved to: output/training_overview.png")
plt.show()
