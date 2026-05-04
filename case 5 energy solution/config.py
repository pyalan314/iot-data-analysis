"""
Central Configuration File
All scripts load configuration from here to ensure consistency
"""

from datetime import datetime

# ============================================================================
# Time Period Configuration
# ============================================================================
START_DATE = datetime(2025, 12, 1)
END_DATE = datetime(2026, 3, 31)
INTERVAL_MINUTES = 15
TEST_DATA_START = datetime(2026, 3, 1)

# ============================================================================
# Building Structure
# ============================================================================
# Floor names (can be strings like "G", "LG", "1F", etc.)
FLOORS = ["1", "2", "3"]

# Floors that will have anomalies injected in test period
ANOMALY_FLOORS = ["3"]

# ============================================================================
# Energy Categories
# ============================================================================
CATEGORY_KEYS = ['air_con', 'computer_appliance', 'lobby_corridor_lighting']
CATEGORY_NAMES = ['Air Con', 'Computer Appliance', 'Lobby/Corridor Lighting']

# Category display names mapping
CATEGORY_DISPLAY_MAP = {
    'air_con': 'Air Conditioning',
    'computer_appliance': 'Computer Appliances',
    'lobby_corridor_lighting': 'Lobby/Corridor Lighting'
}

# Short names for compact display
CATEGORY_SHORT_MAP = {
    'air_con': 'Air Con',
    'computer_appliance': 'Computer',
    'lobby_corridor_lighting': 'Lighting'
}

# ============================================================================
# Analysis Parameters
# ============================================================================
# Forecast anomaly detection threshold (30% deviation)
DEVIATION_THRESHOLD = 0.30

# Spike detection parameters
SPIKE_ZSCORE_THRESHOLD = 2.0
SPIKE_CHANGE_THRESHOLD = 50.0  # Percentage
SPIKE_PATTERN_THRESHOLD = 50.0  # Percentage
ISOLATION_FOREST_CONTAMINATION = 0.05

# ============================================================================
# Data Paths
# ============================================================================
DATA_DIR = 'data'
OUTPUT_DIR = 'output'
MODEL_DIR = 'model'

# Data files
ENERGY_DATA_ALL = f'{DATA_DIR}/energy_consumption_all.csv'
ENERGY_DATA_TRAIN = f'{DATA_DIR}/energy_consumption_train.csv'
ENERGY_DATA_TEST = f'{DATA_DIR}/energy_consumption_test.csv'

# ============================================================================
# Visualization Settings
# ============================================================================
FIGURE_DPI = 300
FIGURE_SIZE_LARGE = (20, 12)
FIGURE_SIZE_MEDIUM = (16, 10)
FIGURE_SIZE_SMALL = (12, 8)

# Color scheme
COLOR_NORMAL = '#2ecc71'
COLOR_ANOMALY = '#e74c3c'
COLOR_FORECAST = '#3498db'
COLOR_ACTUAL = '#95a5a6'

# ============================================================================
# Random Seed for Reproducibility
# ============================================================================
RANDOM_SEED = 42
