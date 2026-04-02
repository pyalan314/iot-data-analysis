"""
Step 2: Load detected anomalies and generate LLM prompts
This script:
- Loads artifacts from Step 1 (anomalies, baseline statistics)
- Generates structured prompts for each anomaly with full context
- Saves prompts to JSON file for LLM processing in Step 3
"""

import pandas as pd
import json

# Import configuration
import step0_config as config

OUTPUT_DIR = config.OUTPUT_DIR

print("="*80)
print("STEP 2: GENERATE LLM PROMPTS FOR ANOMALY INTERPRETATION")
print("="*80)

# 1. LOAD ARTIFACTS FROM STEP 1
print("\n[1/3] Loading artifacts from Step 1...")

# Load baseline statistics
stats_path = OUTPUT_DIR / 'baseline_statistics.json'
if not stats_path.exists():
    print(f"ERROR: {stats_path} not found!")
    print("Please run step1_train_and_detect.py first.")
    exit(1)

with open(stats_path, 'r') as f:
    train_stats = json.load(f)
print(f"  Loaded baseline statistics from: {stats_path}")

# Load detected anomalies
anomalies_path = OUTPUT_DIR / 'detected_anomalies.csv'
if not anomalies_path.exists():
    print(f"ERROR: {anomalies_path} not found!")
    print("Please run step1_train_and_detect.py first.")
    exit(1)

anomalies = pd.read_csv(anomalies_path)
anomalies['timestamp'] = pd.to_datetime(anomalies['timestamp'])
print(f"  Loaded {len(anomalies)} anomalies from: {anomalies_path}")

# Load analysis summary
summary_path = OUTPUT_DIR / 'analysis_summary.json'
with open(summary_path, 'r') as f:
    summary = json.load(f)
current_cols = summary['current_channels']
print(f"  Current channels: {current_cols}")

# 2. GENERATE PROMPTS FOR EACH ANOMALY
print("\n[2/3] Generating LLM prompts...")

prompts = []

for idx, row in anomalies.iterrows():
    # Calculate expected values
    hour = int(row['hour'])
    expected_current = train_stats['hourly_mean'].get(str(hour), train_stats['total_current_mean'])
    expected_std = train_stats['hourly_std'].get(str(hour), train_stats['total_current_std'])
    
    # Build channel information
    channel_info = {}
    for col in current_cols:
        if col in row.index:
            channel_info[col] = {
                'current': round(float(row[col]), 2),
                'expected_mean': round(train_stats['channel_means'].get(col, 0), 2),
                'expected_std': round(train_stats['channel_stds'].get(col, 0), 2),
                'is_active': float(row[col]) > 0
            }
    
    # Create context dictionary
    context = {
        'anomaly_id': idx + 1,
        'timestamp': str(row['timestamp']),
        'hour': hour,
        'hour_fraction': round(float(row['hour_fraction']), 2) if 'hour_fraction' in row.index else hour,
        'day_of_week': int(row['day_of_week']),
        'is_weekend': bool(row['is_weekend']),
        'is_night': bool(row['is_night']),
        'is_business_hours': bool(row['is_business_hours']) if 'is_business_hours' in row.index else False,
        'total_current': {
            'actual': round(float(row['total_current']), 2),
            'expected': round(expected_current, 2),
            'std_dev': round(expected_std, 2),
            'deviation_percent': round(abs(row['total_current'] - expected_current) / expected_current * 100, 1) if expected_current > 0 else 0
        },
        'channels': channel_info,
        'baseline_stats': {
            'mean_current': round(train_stats['total_current_mean'], 2),
            'std_current': round(train_stats['total_current_std'], 2),
            'q25': round(train_stats['total_current_q25'], 2),
            'q75': round(train_stats['total_current_q75'], 2)
        }
    }
    
    # Add anomaly scores for enabled models
    if 'anomaly_score_iforest' in row.index:
        context['anomaly_score_iforest'] = round(float(row['anomaly_score_iforest']), 4)
    if 'anomaly_score_ecod' in row.index:
        context['anomaly_score_ecod'] = round(float(row['anomaly_score_ecod']), 4)
    if 'anomaly_score_lof' in row.index:
        context['anomaly_score_lof'] = round(float(row['anomaly_score_lof']), 4)
    
    # Add detection source if available
    if 'detection_source' in row.index:
        context['detection_source'] = str(row['detection_source'])
    
    # Create prompt for LLM
    prompt = f"""You are an expert electrical engineer analyzing power meter anomalies from IoT sensor data.

ANOMALY CONTEXT:
{json.dumps(context, indent=2)}

TASK:
Analyze this electrical current anomaly and provide a professional assessment:

1. SEVERITY: Classify as HIGH, MEDIUM, or LOW based on:
   - HIGH: Equipment failure, safety risk, channel dropout, or >50% deviation
   - MEDIUM: Load imbalance, unusual timing patterns, 25-50% deviation
   - LOW: Minor statistical outlier, <25% deviation

2. REASON: Explain in 2-3 clear sentences what is abnormal. Focus on:
   - Specific measurements that deviate from baseline
   - What the pattern suggests (circuit failure, overload, unauthorized usage, etc.)
   - Any channel-specific issues (a1A, a2A, a3A, a4A are individual circuits)

3. RECOMMENDATION: Provide one actionable recommendation for maintenance/operations team

RESPONSE FORMAT (JSON only):
{{
  "severity": "HIGH|MEDIUM|LOW",
  "reason": "Your technical explanation here",
  "recommendation": "Your actionable recommendation here"
}}

Be concise, technical but clear, and focus on actionable insights for facility managers."""

    prompts.append({
        'anomaly_id': idx + 1,
        'timestamp': str(row['timestamp']),
        'context': context,
        'prompt': prompt
    })
    
    if (idx + 1) % 10 == 0:
        print(f"  Generated {idx + 1}/{len(anomalies)} prompts...")

print(f"  Generated all {len(prompts)} prompts!")

# 3. SAVE PROMPTS TO FILE
print("\n[3/3] Saving prompts...")

prompts_path = OUTPUT_DIR / 'llm_prompts.json'
with open(prompts_path, 'w') as f:
    json.dump(prompts, f, indent=2)
print(f"  Prompts saved to: {prompts_path}")

# Save a sample prompt for inspection
sample_path = OUTPUT_DIR / 'sample_prompt.txt'
with open(sample_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("SAMPLE LLM PROMPT (Anomaly #1)\n")
    f.write("="*80 + "\n\n")
    f.write(prompts[0]['prompt'])
    f.write("\n\n" + "="*80 + "\n")
    f.write("CONTEXT DATA:\n")
    f.write("="*80 + "\n")
    f.write(json.dumps(prompts[0]['context'], indent=2))
print(f"  Sample prompt saved to: {sample_path}")

# Generate summary
print("\n" + "="*80)
print("STEP 2 COMPLETE!")
print("="*80)
print(f"\nGenerated {len(prompts)} LLM prompts for anomaly interpretation")
print(f"\nFiles created:")
print(f"  1. llm_prompts.json - All prompts ready for LLM processing")
print(f"  2. sample_prompt.txt - Sample prompt for inspection")
print(f"\nPrompt statistics:")
print(f"  Average prompt length: {sum(len(p['prompt']) for p in prompts) / len(prompts):.0f} characters")
print(f"  Total prompts: {len(prompts)}")
print(f"\nNext: Run step3_call_llm.py to generate alert messages")
print("="*80)
