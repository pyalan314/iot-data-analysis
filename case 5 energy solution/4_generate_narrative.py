"""
LLM Narrative Generation
Loads the forecast analysis prompt and uses Grok (via OpenAI API) to generate business narratives.
"""

import json
import os
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from config import DATA_DIR, OUTPUT_DIR

# Load environment variables from .env file
load_dotenv()

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("LLM NARRATIVE GENERATION")
print("="*60)

# ============================================================================
# Load LLM Prompt
# ============================================================================
print("\nLoading LLM prompt...")

try:
    with open(f'{DATA_DIR}/forecast_llm_prompt.json', 'r') as f:
        llm_prompt = json.load(f)
    print(f"✓ Prompt loaded: {DATA_DIR}/forecast_llm_prompt.json")
except FileNotFoundError:
    print(f"✗ Error: {DATA_DIR}/forecast_llm_prompt.json not found")
    print("  Please run 2_forecast_anomaly_detection.py first")
    exit(1)

# Display prompt summary
print(f"\nPrompt summary:")
print(f"  Task: {llm_prompt['task']}")
print(f"  Data insights available: {len(llm_prompt['data_insights']['top_contributors'])} contributors")
print(f"  Context available: {'Yes' if llm_prompt['context'] else 'No'}")
print(f"  Total excess consumption: {llm_prompt['data_insights']['total_excess_kwh']:.1f} kWh")

# ============================================================================
# Initialize OpenAI Client for Grok
# ============================================================================
print("\n" + "="*60)
print("INITIALIZING GROK API")
print("="*60)

# Get API key from environment
api_key = os.getenv('XAI_API_KEY')
if not api_key:
    print("\n✗ Error: XAI_API_KEY environment variable not set")
    print("\nTo set your Grok API key:")
    print("  Option 1 - Using .env file (recommended):")
    print("    1. Copy example.env to .env")
    print("    2. Edit .env and add your API key")
    print("    3. Run this script again")
    print("\n  Option 2 - Set environment variable:")
    print("    Windows (PowerShell): $env:XAI_API_KEY='your-api-key-here'")
    print("    Windows (CMD): set XAI_API_KEY=your-api-key-here")
    print("    Linux/Mac: export XAI_API_KEY='your-api-key-here'")
    exit(1)

# Get model configuration from environment (with defaults)
grok_model = os.getenv('GROK_MODEL', 'grok-beta')
grok_temperature = float(os.getenv('GROK_TEMPERATURE', '0.7'))
grok_max_tokens = int(os.getenv('GROK_MAX_TOKENS', '2000'))

# Initialize client
client = OpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1"
)

print("✓ Grok API client initialized")
print(f"  Model: {grok_model}")
print(f"  Temperature: {grok_temperature}")
print(f"  Max tokens: {grok_max_tokens}")

# ============================================================================
# Construct System and User Messages
# ============================================================================
print("\n" + "="*60)
print("CONSTRUCTING MESSAGES")
print("="*60)

# System message
system_message = """You are an expert energy analyst and technical writer specializing in building management systems. 
Your role is to analyze energy consumption data and generate clear, actionable business narratives for facility managers.

Focus on:
- Identifying patterns and anomalies
- Explaining technical findings in business terms
- Providing practical recommendations
- Highlighting cost and efficiency implications
"""

# User message - combine all prompt components
user_message = f"""# Task
{llm_prompt['task']}

# Data Insights
{json.dumps(llm_prompt['data_insights'], indent=2)}

# Project Context
{json.dumps(llm_prompt['context'], indent=2) if llm_prompt['context'] else 'No context provided'}

# Instructions
{chr(10).join(f"{i+1}. {instruction}" for i, instruction in enumerate(llm_prompt['instructions']))}

# Output Format
{llm_prompt['output_format']}

Please generate a comprehensive narrative based on the data insights and context provided.
"""

print("✓ Messages constructed")
print(f"  System message: {len(system_message)} characters")
print(f"  User message: {len(user_message)} characters")

# ============================================================================
# Call Grok API
# ============================================================================
print("\n" + "="*60)
print("CALLING GROK API")
print("="*60)

try:
    print("\nSending request to Grok...")
    
    response = client.chat.completions.create(
        model=grok_model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=grok_temperature,
        max_tokens=grok_max_tokens
    )
    
    # Extract narrative
    narrative = response.choices[0].message.content
    
    print("✓ Response received from Grok")
    print(f"  Model: {response.model}")
    print(f"  Tokens used: {response.usage.total_tokens}")
    print(f"  - Prompt tokens: {response.usage.prompt_tokens}")
    print(f"  - Completion tokens: {response.usage.completion_tokens}")
    
except Exception as e:
    print(f"\n✗ Error calling Grok API: {e}")
    exit(1)

# ============================================================================
# Display and Save Narrative
# ============================================================================
print("\n" + "="*60)
print("GENERATED NARRATIVE")
print("="*60)

print("\n" + narrative)

# Save narrative
output_data = {
    'narrative': narrative,
    'metadata': {
        'model': response.model,
        'tokens_used': response.usage.total_tokens,
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens,
        'temperature': grok_temperature,
        'max_tokens': grok_max_tokens,
        'generated_at': datetime.now().isoformat()
    },
    'data_insights': llm_prompt['data_insights']
}

# Save as JSON
json_path = f'{OUTPUT_DIR}/generated_narrative.json'
with open(json_path, 'w') as f:
    json.dump(output_data, f, indent=2)

print("\n" + "="*60)
print(f"✓ Narrative saved (JSON): {json_path}")

# Save as plain text
txt_path = f'{OUTPUT_DIR}/generated_narrative.txt'
with open(txt_path, 'w') as f:
    f.write("ENERGY CONSUMPTION ANOMALY ANALYSIS\n")
    f.write("="*60 + "\n\n")
    f.write(narrative)
    f.write("\n\n" + "="*60 + "\n")
    f.write(f"Generated by: {response.model}\n")
    f.write(f"Tokens used: {response.usage.total_tokens}\n")

print(f"✓ Narrative saved (TXT): {txt_path}")

# Save as Markdown
md_path = f'{OUTPUT_DIR}/generated_narrative.md'
with open(md_path, 'w') as f:
    f.write("# Energy Consumption Anomaly Analysis\n\n")
    f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Model:** {response.model}\n\n")
    f.write("---\n\n")
    f.write(narrative)
    f.write("\n\n---\n\n")
    f.write("## Generation Metadata\n\n")
    f.write(f"- **Tokens Used:** {response.usage.total_tokens}\n")
    f.write(f"  - Prompt: {response.usage.prompt_tokens}\n")
    f.write(f"  - Completion: {response.usage.completion_tokens}\n")
    f.write(f"- **Temperature:** {grok_temperature}\n")
    f.write(f"- **Max Tokens:** {grok_max_tokens}\n")

print(f"✓ Narrative saved (MD): {md_path}")

print("\n✓ Narrative generation complete!")
