"""
Step 0: Configuration file for anomaly detection pipeline
Centralized configuration for all pipeline steps.
Modify these settings to customize the analysis.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# DIRECTORY CONFIGURATION
# ============================================================================
DATA_DIR = Path('data')
OUTPUT_DIR = Path('output')

# ============================================================================
# TIMEZONE CONFIGURATION
# ============================================================================
# Timezone for data analysis (loaded from .env, default: Asia/Hong_Kong)
# Other examples: Asia/Shanghai, Asia/Singapore, America/New_York, Europe/London
TIMEZONE = os.getenv('TIMEZONE', 'Asia/Hong_Kong')

# ============================================================================
# DATE RANGE CONFIGURATION
# ============================================================================
# Training data cutoff (data before this date will be used for training)
# Format: 'YYYY-MM-DD' in UTC
TRAIN_CUTOFF_DATE = '2025-11-01'

# Test data range (data in this range will be analyzed for anomalies)
# Format: 'YYYY-MM-DD' in UTC
TEST_START_DATE = '2025-11-01'
TEST_END_DATE = '2026-03-01'

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Contamination rate: Expected percentage of anomalies in the data
# Range: 0.0 to 0.5 (e.g., 0.02 = 2% of data expected to be anomalies)
# Lower values = stricter anomaly detection (fewer anomalies)
# Higher values = more lenient anomaly detection (more anomalies)
CONTAMINATION = 0.02

# Model selection: Enable/disable specific algorithms
# Set to True to enable, False to disable
USE_IFOREST = True   # Isolation Forest (global anomaly detection)
USE_ECOD = False     # ECOD (inconsistent across test ranges - not recommended)
USE_LOF = False       # Local Outlier Factor (local density-based detection)

# Isolation Forest specific parameters
IFOREST_N_ESTIMATORS = 100  # Number of trees (higher = more accurate but slower)
IFOREST_RANDOM_STATE = 42   # Random seed for reproducibility

# LOF (Local Outlier Factor) specific parameters
LOF_N_NEIGHBORS = 20        # Number of neighbors to consider (higher = smoother, lower = more sensitive)

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================
# Time-based features to extract
USE_CYCLICAL_TIME = True      # Use sin/cos encoding for hour and day
USE_BUSINESS_HOURS = True     # Include is_business_hours flag
USE_WEEKEND_FLAG = True       # Include is_weekend flag
USE_NIGHT_FLAG = True         # Include is_night flag

# Business hours definition (local time, 24-hour format)
BUSINESS_HOURS_START = 9   # 09:00
BUSINESS_HOURS_END = 18    # 18:00 (exclusive, so 17:59 is last business hour)

# Night hours definition (local time, 24-hour format)
NIGHT_HOURS_START = 0   # 00:00
NIGHT_HOURS_END = 6     # 06:00 (exclusive, so 05:59 is last night hour)

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================
PLOT_DPI = 150              # Resolution for saved plots
PLOT_FIGSIZE = (14, 14)     # Figure size (width, height) in inches

# ============================================================================
# LLM CONFIGURATION
# ============================================================================
# LLM API settings (loaded from .env)
GROK_API_KEY = os.getenv('XAI_API_KEY')
GROK_MODEL = os.getenv('GROK_MODEL', 'grok-4-1-fast')
LLM_TEMPERATURE = 0.3       # Lower = more consistent, Higher = more creative
LLM_MAX_TOKENS = 400        # Maximum tokens in LLM response
LLM_RATE_LIMIT_DELAY = 0.1  # Delay between API calls (seconds)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_config_summary():
    """Return a dictionary of all configuration settings"""
    enabled_models = []
    if USE_IFOREST:
        enabled_models.append('IForest')
    if USE_ECOD:
        enabled_models.append('ECOD')
    if USE_LOF:
        enabled_models.append('LOF')
    
    return {
        'timezone': TIMEZONE,
        'train_cutoff': TRAIN_CUTOFF_DATE,
        'test_start': TEST_START_DATE,
        'test_end': TEST_END_DATE,
        'contamination': CONTAMINATION,
        'enabled_models': enabled_models,
        'iforest_n_estimators': IFOREST_N_ESTIMATORS,
        'lof_n_neighbors': LOF_N_NEIGHBORS,
        'business_hours': f"{BUSINESS_HOURS_START:02d}:00-{BUSINESS_HOURS_END:02d}:00",
        'night_hours': f"{NIGHT_HOURS_START:02d}:00-{NIGHT_HOURS_END:02d}:00",
        'grok_model': GROK_MODEL,
        'llm_temperature': LLM_TEMPERATURE
    }

def print_config():
    """Print current configuration settings"""
    print("="*80)
    print("PIPELINE CONFIGURATION")
    print("="*80)
    print(f"Timezone: {TIMEZONE}")
    print(f"Training data: Before {TRAIN_CUTOFF_DATE}")
    print(f"Test data: {TEST_START_DATE} to {TEST_END_DATE}")
    print(f"Contamination rate: {CONTAMINATION*100:.1f}%")
    print(f"\nEnabled Models:")
    if USE_IFOREST:
        print(f"  ✓ Isolation Forest (n_estimators={IFOREST_N_ESTIMATORS})")
    if USE_ECOD:
        print(f"  ✓ ECOD (WARNING: Inconsistent across test ranges)")
    if USE_LOF:
        print(f"  ✓ LOF (n_neighbors={LOF_N_NEIGHBORS})")
    if not (USE_IFOREST or USE_ECOD or USE_LOF):
        print(f"  ✗ No models enabled!")
    print(f"\nBusiness hours: {BUSINESS_HOURS_START:02d}:00-{BUSINESS_HOURS_END:02d}:00 (local time)")
    print(f"Night hours: {NIGHT_HOURS_START:02d}:00-{NIGHT_HOURS_END:02d}:00 (local time)")
    print(f"LLM Model: {GROK_MODEL}")
    print("="*80)

# ============================================================================
# VALIDATION
# ============================================================================
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    if not 0 < CONTAMINATION < 0.5:
        errors.append(f"CONTAMINATION must be between 0 and 0.5, got {CONTAMINATION}")
    
    if not (USE_IFOREST or USE_ECOD or USE_LOF):
        errors.append("At least one anomaly detection model must be enabled")
    
    if not 0 <= BUSINESS_HOURS_START < 24:
        errors.append(f"BUSINESS_HOURS_START must be 0-23, got {BUSINESS_HOURS_START}")
    
    if not 0 <= BUSINESS_HOURS_END <= 24:
        errors.append(f"BUSINESS_HOURS_END must be 0-24, got {BUSINESS_HOURS_END}")
    
    if BUSINESS_HOURS_START >= BUSINESS_HOURS_END:
        errors.append("BUSINESS_HOURS_START must be less than BUSINESS_HOURS_END")
    
    if not DATA_DIR.exists():
        errors.append(f"Data directory not found: {DATA_DIR}")
    
    if errors:
        print("Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True

if __name__ == "__main__":
    # Print configuration when run directly
    print_config()
    print()
    if validate_config():
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration has errors")
