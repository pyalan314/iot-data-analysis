"""
Step 3: Load prompts and call LLM to generate alert messages
This script:
- Loads prompts from Step 2
- Calls Grok API (or other LLM) for each anomaly
- Parses LLM responses and generates human-readable alerts
- Saves final alert messages with severity and recommendations
"""

import json
import time
from openai import OpenAI

# Import configuration
import step0_config as config

# Initialize LLM client (Grok API)
if not config.GROK_API_KEY:
    print("ERROR: XAI_API_KEY not found in .env file!")
    print("Please create a .env file with: XAI_API_KEY=your-api-key-here")
    exit(1)

client = OpenAI(
    api_key=config.GROK_API_KEY,
    base_url="https://api.x.ai/v1"
)

print("="*80)
print("STEP 3: CALL LLM TO GENERATE ALERT MESSAGES")
print("="*80)
print(f"Using model: {config.GROK_MODEL}")
print(f"Temperature: {config.LLM_TEMPERATURE}, Max tokens: {config.LLM_MAX_TOKENS}")

# 1. LOAD PROMPTS FROM STEP 2
print("\n[1/3] Loading prompts from Step 2...")

prompts_path = config.OUTPUT_DIR / 'llm_prompts.json'
if not prompts_path.exists():
    print(f"ERROR: {prompts_path} not found!")
    print("Please run step2_generate_prompts.py first.")
    exit(1)

with open(prompts_path, 'r') as f:
    prompts = json.load(f)
print(f"  Loaded {len(prompts)} prompts from: {prompts_path}")

# 2. CALL LLM FOR EACH PROMPT
print(f"\n[2/3] Calling LLM API to generate alerts...")
print(f"  Processing {len(prompts)} anomalies...")
print(f"  Note: This may take a few minutes.\n")
print(f"  Progress will be saved incrementally to: {config.OUTPUT_DIR / 'alerts_progress.json'}")
print(f"  You can resume from this file if interrupted.\n")

# Load existing progress if available
progress_file = config.OUTPUT_DIR / 'alerts_progress.json'
if progress_file.exists():
    print(f"  Found existing progress file. Loading...")
    with open(progress_file, 'r') as f:
        progress_data = json.load(f)
        alerts = progress_data.get('alerts', [])
        failed_count = progress_data.get('failed_count', 0)
        start_idx = len(alerts) + 1
    print(f"  Resuming from anomaly #{start_idx}\n")
else:
    alerts = []
    failed_count = 0
    start_idx = 1

for idx, prompt_data in enumerate(prompts, 1):
    # Skip already processed
    if idx < start_idx:
        continue
    try:
        # Call Grok API
        response = client.chat.completions.create(
            model=config.GROK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert electrical engineer specializing in IoT power monitoring and anomaly detection. Provide clear, actionable technical analysis."
                },
                {
                    "role": "user",
                    "content": prompt_data['prompt']
                }
            ],
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS
        )
        
        # Parse LLM response
        llm_output = response.choices[0].message.content.strip()
        
        # Extract JSON from response (handle markdown code blocks)
        if '```json' in llm_output:
            llm_output = llm_output.split('```json')[1].split('```')[0].strip()
        elif '```' in llm_output:
            llm_output = llm_output.split('```')[1].split('```')[0].strip()
        
        result = json.loads(llm_output)
        
        # Create alert record
        alert = {
            'anomaly_id': prompt_data['anomaly_id'],
            'timestamp': prompt_data['timestamp'],
            'severity': result.get('severity', 'LOW'),
            'reason': result.get('reason', 'Anomaly detected'),
            'recommendation': result.get('recommendation', 'Monitor situation'),
            'context': prompt_data['context'],
            'model_used': config.GROK_MODEL,
            'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else None
        }
        alerts.append(alert)
        
        # Display progress for every anomaly
        severity_emoji = {'HIGH': '[!!!]', 'MEDIUM': '[!!]', 'LOW': '[!]'}
        emoji = severity_emoji.get(alert['severity'], '[?]')
        print(f"  [{idx}/{len(prompts)}] {emoji} Anomaly #{alert['anomaly_id']} - {alert['severity']} - {alert['timestamp'][:16]}")
        
        # Save progress after each successful call
        with open(progress_file, 'w') as f:
            json.dump({
                'alerts': alerts,
                'failed_count': failed_count,
                'last_processed': idx,
                'total': len(prompts)
            }, f, indent=2)
        
        # Rate limiting - small delay between requests
        time.sleep(config.LLM_RATE_LIMIT_DELAY)
        
    except Exception as e:
        print(f"  ERROR processing anomaly #{prompt_data['anomaly_id']}: {e}")
        failed_count += 1
        
        # Create fallback alert
        context = prompt_data['context']
        total_current = context['total_current']
        alert = {
            'anomaly_id': prompt_data['anomaly_id'],
            'timestamp': prompt_data['timestamp'],
            'severity': 'MEDIUM',
            'reason': f"Anomaly detected: {total_current['actual']}A vs expected {total_current['expected']}A. Deviation: {total_current['deviation_percent']}%",
            'recommendation': 'Review electrical system and investigate cause of deviation',
            'context': context,
            'model_used': 'fallback',
            'tokens_used': None
        }
        alerts.append(alert)
        
        # Save progress even for failed calls
        with open(progress_file, 'w') as f:
            json.dump({
                'alerts': alerts,
                'failed_count': failed_count,
                'last_processed': idx,
                'total': len(prompts)
            }, f, indent=2)

print(f"\n  Completed! Processed {len(alerts)} anomalies")
if failed_count > 0:
    print(f"  Note: {failed_count} anomalies used fallback (LLM call failed)")

# 3. SAVE ALERT MESSAGES
print("\n[3/3] Saving alert messages...")

# Save full alerts with context
full_alerts_path = config.OUTPUT_DIR / 'final_alerts_full.json'
with open(full_alerts_path, 'w') as f:
    json.dump(alerts, f, indent=2)
print(f"  Full alerts saved to: {full_alerts_path}")

# Save simplified alerts for easy reading
simplified_alerts = []
for alert in alerts:
    simplified_alerts.append({
        'anomaly_id': alert['anomaly_id'],
        'timestamp': alert['timestamp'],
        'severity': alert['severity'],
        'total_current': alert['context']['total_current']['actual'],
        'expected_current': alert['context']['total_current']['expected'],
        'deviation_percent': alert['context']['total_current']['deviation_percent'],
        'reason': alert['reason'],
        'recommendation': alert['recommendation']
    })

simplified_path = config.OUTPUT_DIR / 'final_alerts_summary.csv'
import pandas as pd
pd.DataFrame(simplified_alerts).to_csv(simplified_path, index=False)
print(f"  Summary alerts saved to: {simplified_path}")

# Generate statistics
severity_counts = {}
for alert in alerts:
    severity = alert['severity']
    severity_counts[severity] = severity_counts.get(severity, 0) + 1

total_tokens = sum(a['tokens_used'] for a in alerts if a['tokens_used'] is not None)

# Save final report
report = {
    'total_anomalies': len(alerts),
    'severity_breakdown': severity_counts,
    'model_used': config.GROK_MODEL,
    'total_tokens_used': total_tokens,
    'failed_llm_calls': failed_count,
    'success_rate': f"{(len(alerts) - failed_count) / len(alerts) * 100:.1f}%"
}

report_path = config.OUTPUT_DIR / 'step3_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"  Report saved to: {report_path}")

# Display sample alerts
print("\n" + "="*80)
print("STEP 3 COMPLETE!")
print("="*80)
print(f"\nGenerated {len(alerts)} alert messages using {config.GROK_MODEL}")
print(f"\nSeverity breakdown:")
for severity, count in sorted(severity_counts.items()):
    print(f"  {severity}: {count} ({count/len(alerts)*100:.1f}%)")

print(f"\nAPI Usage:")
print(f"  Total tokens used: {total_tokens:,}")
print(f"  Success rate: {report['success_rate']}")

print(f"\nFiles created:")
print(f"  1. final_alerts_full.json - Complete alerts with all context")
print(f"  2. final_alerts_summary.csv - Simplified alert summary")
print(f"  3. step3_report.json - Processing statistics")

# Display top 3 high severity alerts
high_severity = [a for a in alerts if a['severity'] == 'HIGH']
if high_severity:
    print(f"\n" + "="*80)
    print(f"TOP 3 HIGH SEVERITY ALERTS")
    print("="*80)
    for i, alert in enumerate(high_severity[:3], 1):
        print(f"\n[{i}] Anomaly #{alert['anomaly_id']} - {alert['timestamp']}")
        print(f"    Current: {alert['context']['total_current']['actual']}A "
              f"(expected {alert['context']['total_current']['expected']}A)")
        print(f"    REASON: {alert['reason']}")
        print(f"    ACTION: {alert['recommendation']}")

print("\n" + "="*80)
print("ANALYSIS PIPELINE COMPLETE!")
print("="*80)
print("\nAll steps finished successfully:")
print("  Step 1: Trained models and detected anomalies")
print("  Step 2: Generated LLM prompts with context")
print("  Step 3: Called LLM and generated alert messages")
print(f"\nCheck the '{config.OUTPUT_DIR}/' directory for all results.")
print("="*80)
