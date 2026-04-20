# Anomaly Detection for IoT Energy Monitoring Using Isolation Forest

## 1. Problem Statement: The Need for Anomaly Detection in IoT Sensor Devices

In modern IoT deployments, sensor devices continuously generate massive amounts of time-series data. Traditional monitoring approaches that rely on fixed thresholds face several critical limitations:

### Challenges with IoT Sensor Monitoring:

- **Dynamic Patterns**: IoT sensor data exhibits complex temporal patterns (hourly, daily, weekly cycles) that fixed thresholds cannot capture
- **Context-Dependent Behavior**: Normal behavior varies significantly based on time of day, day of week, and seasonal factors
- **Alert Fatigue**: Fixed thresholds generate excessive false positives, leading to ignored alerts
- **Missed Anomalies**: Subtle deviations that indicate real issues may fall within threshold ranges
- **Manual Tuning**: Constant threshold adjustments are required as system behavior evolves

### The Need for Intelligent Anomaly Detection:

IoT systems require automated, adaptive anomaly detection that can:
- Learn normal patterns from historical data
- Detect deviations in context (e.g., high usage at unusual times)
- Reduce false positives while catching genuine anomalies
- Adapt to changing system behavior without manual intervention
- Provide severity classification for prioritized response

---

## 2. Solution: Isolation Forest for Anomaly Detection

### What is Isolation Forest?

Isolation Forest is an unsupervised machine learning algorithm specifically designed for anomaly detection. It operates on a fundamentally different principle than traditional methods:

**Core Principle**: Anomalies are rare and different, making them easier to isolate than normal points.

### How Isolation Forest Works:

1. **Random Partitioning**: Builds multiple decision trees by randomly selecting features and split values
2. **Isolation**: Anomalies require fewer splits to isolate compared to normal points
3. **Path Length**: Measures the average path length to isolate each point across all trees
4. **Anomaly Score**: Shorter paths → higher anomaly scores

### Why Isolation Forest for IoT?

**Advantages:**
- **Unsupervised**: No labeled anomaly data required
- **Efficient**: Linear time complexity O(n), suitable for large IoT datasets
- **Multi-dimensional**: Handles multiple features simultaneously (energy, time, context)
- **Robust**: Works well with high-dimensional data and mixed feature types
- **Interpretable**: Anomaly scores provide clear severity rankings

**Application to Energy Monitoring:**
- Learns normal energy consumption patterns across different times and days
- Detects unusual consumption events (e.g., equipment left on overnight, system malfunctions)
- Adapts to seasonal changes and evolving usage patterns
- Provides actionable insights with severity classification

---

## 3. Example Scenario: Office Energy Monitoring

### Scenario Description

We demonstrate anomaly detection on a **3-month office energy monitoring dataset** with the following characteristics:

**Data Specifications:**
- **Time Period**: January 1 - March 31, 2024 (3 months)
- **Sampling Interval**: 15 minutes
- **Total Records**: ~8,640 data points

**Normal Energy Usage Patterns:**
- **Weekday Office Hours (9 AM - 6 PM)**: Average 80 kWh
- **Weekday Non-Office Hours**: Average 10 kWh
- **Weekends**: Average 10 kWh

**Injected Anomalies (Last Week):**
1. **Tuesday Night (Mar 26, 11 PM - 12 AM)**: Energy rises to 50 kWh (equipment left on)
2. **Friday Afternoon (Mar 29, 2 PM - 4 PM)**: Energy spikes to 100 kWh (HVAC malfunction)
3. **Saturday Evening (Mar 30, 7 PM - 10 PM)**: Energy rises to 30 kWh (unauthorized access)

### Generated Data Visualization

![Generated Office Energy Data](output/1_generated_data.png)

**Figure 1**: Three-month energy usage dataset showing:
- **Top Panel**: Complete 3-month timeline with injected anomalies marked in red
- **Middle Panel**: Zoomed view of the last week (detection period) highlighting anomaly events
- **Bottom Panel**: Average hourly patterns comparing weekday vs weekend usage, with office hours shaded

The data clearly shows distinct patterns:
- High energy consumption during weekday office hours
- Low baseline consumption during nights and weekends
- Three anomalous events in the final week that deviate from normal patterns

---

## 4. Methodology: Applying Isolation Forest

### Step 1: Data Generation

Generate synthetic office energy data with realistic patterns:

```bash
python 1_generate_data.py
```

**Output**: `data/office_energy_data.csv`

### Step 2: Feature Engineering

Transform raw timestamps into meaningful features:

**Temporal Features:**
- `hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0=Monday, 6=Sunday)
- `is_weekday`: Binary flag (1=weekday, 0=weekend)
- `is_office_hours`: Binary flag (1=office hours, 0=otherwise)

**Target Variable:**
- `energy_usage`: Energy consumption in kWh

### Step 3: Model Training

Train Isolation Forest on normal data (Jan 1 - Mar 24):

```bash
python 2_training.py
```

**Model Configuration:**
- **Algorithm**: Isolation Forest (PyOD implementation)
- **Number of Trees**: 100
- **Contamination Rate**: 1% (expected anomaly proportion)
- **Training Samples**: ~8,256 records (3 months minus last week)

![Training Results](output/2_training_results.png)

**Figure 2**: Training phase analysis showing:
- **Top Left**: Training data with detected anomalies during training
- **Top Right**: Distribution of anomaly scores with threshold line
- **Bottom Left**: Hourly patterns of energy usage vs anomaly scores
- **Bottom Right**: Anomaly score distribution comparing weekdays vs weekends

The model learns that:
- High energy during office hours is normal
- Low energy at night/weekends is normal
- Deviations from these patterns receive higher anomaly scores

### Step 4: Anomaly Detection

Apply trained model to detect anomalies in test period (Mar 24-31):

```bash
python 3_detect.py
```

**Severity Classification:**
- **Normal**: Anomaly score ≤ threshold
- **Moderate Anomaly**: Threshold < score ≤ 75th percentile of anomalies
- **Severe Anomaly**: Score > 75th percentile of anomalies

![Detection Results](output/3_detection_results.png)

**Figure 3**: Anomaly detection results showing:
- **Top Panel**: Energy usage timeline with flagged anomalies (M=Moderate, S=Severe)
- **Middle Panel**: Anomaly scores over time with threshold lines
- **Bottom Panel**: Distribution of severity classifications

![Anomaly Details](output/3_anomaly_details.png)

**Figure 4**: Detailed anomaly analysis:
- **Top Left**: Moderate anomalies colored by severity score
- **Top Right**: Severe anomalies with highest risk scores
- **Bottom Left**: Hourly distribution of anomaly types
- **Bottom Right**: Daily breakdown of detected anomalies

---

## 5. Results Evaluation

### Detection Performance

The Isolation Forest model successfully identified all three injected anomalies with appropriate severity levels:

**Detected Anomalies:**

1. **Tuesday Night Anomaly (Mar 26, 11 PM - 12 AM)**
   - **Energy Level**: ~50 kWh (5x normal night usage)
   - **Classification**: Severe
   - **Interpretation**: Equipment left running overnight

2. **Friday Afternoon Anomaly (Mar 29, 2 PM - 4 PM)**
   - **Energy Level**: ~100 kWh (1.25x normal office hours)
   - **Classification**: Severe
   - **Interpretation**: HVAC system malfunction or excessive load

3. **Saturday Evening Anomaly (Mar 30, 7 PM - 10 PM)**
   - **Energy Level**: ~30 kWh (3x normal weekend usage)
   - **Classification**: Moderate to Severe
   - **Interpretation**: Unauthorized building access or security breach

### Model Strengths

✅ **Contextual Understanding**: The model correctly identifies that 50 kWh at night is anomalous, while 80 kWh during office hours is normal

✅ **Severity Ranking**: Provides actionable prioritization - severe anomalies require immediate attention

✅ **Low False Positives**: With 3 months of training data, the model accurately learned normal patterns without flagging regular Monday/Friday variations

✅ **Temporal Awareness**: Successfully captures weekly and daily cycles through feature engineering

### Limitations and Considerations

⚠️ **Training Data Quality**: Requires sufficient historical data (recommended: 2-3 months minimum) to learn patterns

⚠️ **Contamination Parameter**: Must be tuned based on expected anomaly rate in your environment

⚠️ **Concept Drift**: Model should be retrained periodically as normal patterns evolve (e.g., seasonal changes)

⚠️ **Feature Selection**: Performance depends on choosing relevant features for your specific use case

---

## 6. Anomaly Detection vs. Fixed Threshold Alerts

### Visual Comparison

![Threshold Comparison](output/3_threshold_comparison.png)

**Figure 5**: Three-panel comparison of detection approaches:

- **Panel A (Top)**: Fixed threshold approach using 90 kWh threshold
  - Shows alerts triggered when energy exceeds 90 kWh
  - Simple binary detection (above/below threshold)

- **Panel B (Middle)**: ML-based anomaly detection approach
  - Context-aware detection with severity classification
  - Identifies anomalies based on learned patterns, not absolute values

- **Panel C (Bottom)**: Direct comparison highlighting missed anomalies
  - **Green circles**: Anomalies detected by both approaches
  - **Red X markers with "MISSED!" labels**: Anomalies completely missed by fixed threshold
  - Clearly shows which critical events would go undetected with traditional monitoring

### Key Findings from Visualization:

**Fixed Threshold Failures:**
1. **Tuesday Night Anomaly (50 kWh)**: MISSED - Equipment left running overnight goes undetected because 50 kWh < 90 kWh threshold
2. **Saturday Evening Anomaly (30 kWh)**: MISSED - Unauthorized building access unnoticed because 30 kWh < 90 kWh threshold
3. **Context Blindness**: Cannot distinguish between normal office hour peaks and unusual nighttime/weekend usage

**Anomaly Detection Success:**
- Detects all anomalies regardless of absolute energy value
- Understands that 50 kWh at 11 PM is highly unusual (5x normal night usage)
- Recognizes that 30 kWh on Saturday evening is abnormal (3x normal weekend usage)
- Provides severity classification for prioritized response

### Comparison Analysis

| Aspect | Fixed Threshold | Anomaly Detection (Isolation Forest) |
|--------|----------------|-------------------------------------|
| **Context Awareness** | ❌ No - same threshold 24/7 | ✅ Yes - considers time, day, patterns |
| **Adaptability** | ❌ Manual updates required | ✅ Learns from data automatically |
| **False Positives** | ❌ High - normal peaks trigger alerts | ✅ Low - understands normal variations |
| **Missed Anomalies** | ❌ Subtle issues go undetected | ✅ Detects context-dependent deviations |
| **Severity Ranking** | ❌ Binary (alert/no alert) | ✅ Continuous scores with classification |
| **Setup Complexity** | ✅ Simple - set one value | ⚠️ Moderate - requires training data |
| **Maintenance** | ❌ Constant tuning needed | ✅ Periodic retraining only |
| **Interpretability** | ✅ Very clear | ✅ Clear with anomaly scores |

### Real-World Example: Tuesday Night Anomaly

**Scenario**: Energy usage of 50 kWh at 11 PM

**Fixed Threshold Approach (e.g., threshold = 90 kWh):**
- ❌ **Missed Detection**: 50 kWh < 90 kWh → No alert
- ❌ **Problem**: Equipment running overnight goes unnoticed
- ❌ **Cost**: Wasted energy, potential equipment damage

**Anomaly Detection Approach:**
- ✅ **Detected**: 50 kWh is 5x normal night usage → Severe anomaly
- ✅ **Context**: Model knows night usage should be ~10 kWh
- ✅ **Action**: Immediate alert for investigation
- ✅ **Outcome**: Problem identified and resolved quickly

### Advantages of Anomaly Detection

1. **Reduced Alert Fatigue**
   - Fewer false alarms → higher trust in alerts
   - Operations team can focus on genuine issues

2. **Earlier Problem Detection**
   - Catches subtle deviations before they become critical
   - Identifies unusual patterns that fixed thresholds miss

3. **Lower Operational Costs**
   - Automated learning reduces manual threshold tuning
   - Prevents energy waste and equipment damage

4. **Better Resource Allocation**
   - Severity classification enables prioritized response
   - Critical issues get immediate attention

5. **Scalability**
   - Single model handles complex multi-dimensional patterns
   - Easy to deploy across multiple buildings/devices

6. **Continuous Improvement**
   - Model adapts as system behavior evolves
   - Historical anomalies inform future detection

---

## 7. IoT Applications for Anomaly Detection

### Industrial IoT Applications

#### 1. **Smart Building Management**
- **HVAC Systems**: Detect inefficient operation, refrigerant leaks, filter clogs
- **Lighting**: Identify lights left on, fixture failures, occupancy anomalies
- **Water Usage**: Detect leaks, unusual consumption, pipe bursts
- **Access Control**: Flag unauthorized entry, tailgating, unusual access patterns

#### 2. **Manufacturing & Industry 4.0**
- **Equipment Monitoring**: Predict machine failures, detect abnormal vibrations/temperatures
- **Quality Control**: Identify defective products, process deviations
- **Supply Chain**: Detect delays, inventory anomalies, logistics issues
- **Energy Management**: Optimize consumption, identify waste, prevent overloads

#### 3. **Smart Cities**
- **Traffic Management**: Detect accidents, congestion, signal malfunctions
- **Waste Management**: Identify overflow, collection inefficiencies
- **Environmental Monitoring**: Detect pollution spikes, air quality issues
- **Street Lighting**: Identify outages, energy waste

#### 4. **Healthcare IoT**
- **Patient Monitoring**: Detect vital sign anomalies, fall detection
- **Medical Equipment**: Predict failures, ensure proper operation
- **Cold Chain**: Monitor vaccine/medication storage conditions
- **Hospital Operations**: Optimize resource utilization

#### 5. **Agriculture (AgriTech)**
- **Irrigation Systems**: Detect leaks, pump failures, soil moisture anomalies
- **Greenhouse Monitoring**: Temperature/humidity deviations, ventilation issues
- **Livestock Monitoring**: Detect sick animals, unusual behavior
- **Crop Health**: Identify pest infestations, disease outbreaks

#### 6. **Utilities & Energy**
- **Smart Grid**: Detect power theft, grid instability, equipment failures
- **Solar/Wind Farms**: Identify underperforming panels/turbines
- **Oil & Gas**: Pipeline leak detection, pressure anomalies
- **Water Distribution**: Detect leaks, contamination, pressure issues

#### 7. **Transportation & Logistics**
- **Fleet Management**: Detect vehicle malfunctions, driver behavior anomalies
- **Predictive Maintenance**: Identify component wear, prevent breakdowns
- **Route Optimization**: Detect delays, traffic anomalies
- **Cold Chain Logistics**: Monitor temperature-sensitive cargo

#### 8. **Retail & Commercial**
- **Refrigeration**: Detect temperature deviations, door seal failures
- **Foot Traffic**: Identify unusual patterns, security concerns
- **Inventory**: Detect shrinkage, stock anomalies
- **Point of Sale**: Fraud detection, transaction anomalies

### Implementation Benefits Across Domains

**Cost Savings:**
- Reduce energy waste by 15-30%
- Prevent equipment failures (predictive maintenance)
- Minimize downtime and repair costs

**Operational Efficiency:**
- Automated monitoring reduces manual inspection
- Faster incident response and resolution
- Optimized resource allocation

**Safety & Compliance:**
- Early detection of hazardous conditions
- Ensure regulatory compliance
- Protect personnel and assets

**Sustainability:**
- Reduce carbon footprint through energy optimization
- Minimize waste and resource consumption
- Support ESG (Environmental, Social, Governance) goals

---

## Conclusion

Anomaly detection using Isolation Forest provides a powerful, scalable solution for IoT sensor monitoring. By learning normal patterns from historical data, it delivers context-aware alerts with significantly fewer false positives than traditional threshold-based approaches.

The office energy monitoring example demonstrates how this technique can identify subtle anomalies (equipment left on overnight) that fixed thresholds would miss, while avoiding false alarms during normal peak usage periods.

As IoT deployments continue to grow across industries, intelligent anomaly detection will become essential for managing the complexity and scale of sensor data, enabling proactive maintenance, cost savings, and improved operational efficiency.

### Key Takeaways

1. ✅ Isolation Forest excels at unsupervised anomaly detection in IoT time-series data
2. ✅ Context-aware detection reduces false positives and catches subtle issues
3. ✅ Severity classification enables prioritized incident response
4. ✅ Applicable across diverse IoT domains from smart buildings to industrial monitoring
5. ✅ Significant advantages over fixed threshold approaches in dynamic environments

---

## Repository Structure

```
case 3.1 anomaly detection by ml/
├── 1_generate_data.py          # Generate synthetic office energy data
├── 2_training.py                # Train Isolation Forest model
├── 3_detect.py                  # Detect anomalies with severity classification
├── requirements.txt             # Python dependencies
├── REPORT.md                    # This report
├── data/
│   ├── office_energy_data.csv   # Generated energy dataset
│   └── detection_results.csv    # Anomaly detection results
├── model/
│   ├── iforest_model.pkl        # Trained Isolation Forest model
│   └── feature_columns.csv      # Feature configuration
└── output/
    ├── 1_generated_data.png     # Data generation visualizations
    ├── 2_training_results.png   # Training phase analysis
    ├── 3_detection_results.png  # Detection results overview
    └── 3_anomaly_details.png    # Detailed anomaly analysis
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Generate data
python 1_generate_data.py

# Train model
python 2_training.py

# Detect anomalies
python 3_detect.py
```

---

**Author**: IoT Data Analysis Project  
**Date**: April 2026  
**Technology Stack**: Python, PyOD, Isolation Forest, Pandas, Matplotlib
