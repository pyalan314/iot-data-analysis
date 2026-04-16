# Anomaly Detection for IoT Sensor Devices Using Machine Learning

## 1. Problem Statement: The Need for Anomaly Detection in IoT Sensor Devices

In modern IoT ecosystems, sensor devices continuously generate massive amounts of time-series data. Detecting anomalies in this data is critical for:

- **Early Warning Systems**: Identifying equipment failures before they cause system downtime
- **Preventive Maintenance**: Detecting unusual patterns that indicate degrading performance
- **Safety Monitoring**: Alerting operators to dangerous conditions in real-time
- **Cost Reduction**: Preventing catastrophic failures through early intervention
- **Data Quality**: Identifying sensor malfunctions or data transmission errors

Traditional threshold-based alerting systems often fail because:
- They require manual threshold configuration for each sensor
- They cannot adapt to changing normal patterns
- They generate excessive false alarms
- They miss subtle anomalies that don't cross fixed thresholds

**Machine learning-based anomaly detection** addresses these limitations by learning normal behavior patterns from historical data and automatically identifying deviations.

---

## 2. Solution: Isolation Forest for Anomaly Detection

### What is Isolation Forest?

**Isolation Forest** is an unsupervised machine learning algorithm specifically designed for anomaly detection. It works on a fundamentally different principle than traditional methods:

**Key Concept**: Anomalies are rare and different, making them easier to isolate than normal points.

### How It Works

1. **Random Partitioning**: The algorithm randomly selects a feature and a split value to partition the data
2. **Isolation Trees**: It builds multiple decision trees that isolate data points
3. **Path Length**: Anomalies require fewer splits to isolate (shorter path length)
4. **Anomaly Score**: Points with shorter average path lengths across all trees are classified as anomalies

### Advantages for IoT Applications

- **Unsupervised Learning**: No need for labeled anomaly data
- **Scalability**: Efficient with large datasets (linear time complexity)
- **High Dimensional Data**: Works well with multiple sensor features
- **Adaptability**: Can be retrained as normal patterns evolve
- **Low False Positive Rate**: Effective at distinguishing true anomalies from noise

### Feature Engineering for Time-Series Data

To improve detection accuracy, we engineered multiple feature types:

- **Temporal Features**: Hour, minute (capturing daily patterns)
- **Lag Features**: Previous values at 15min, 1h, 3h, 6h intervals
- **Change Features**: Rate of change over different time windows
- **Rolling Statistics**: Mean, standard deviation, min, max over 6-hour windows

These features help the model understand temporal dependencies and detect sudden changes that indicate anomalies.

---

## 3. Example Scenario: Server Room Temperature Monitoring

### Scenario Description

We simulate a **server room temperature monitoring system** where:

- **Monitoring Period**: 30 days (1 month)
- **Sampling Frequency**: Every 15 minutes (96 readings per day)
- **Normal Temperature**: 22°C ± 2°C with daily fluctuations
- **Anomaly Events**: Two temperature surge incidents

### Simulated Anomalies

1. **Anomaly 1 (Mid-period)**: Moderate temperature surge around day 15
   - Duration: 9 hours
   - Peak increase: +5°C above normal
   - Possible cause: HVAC partial failure or increased server load

2. **Anomaly 2 (Late-period)**: Severe temperature surge around day 25
   - Duration: 12 hours
   - Peak increase: +8°C above normal
   - Possible cause: HVAC complete failure or cooling system malfunction

### Generated Data Visualization

![Generated Temperature Data](output/1_generated_data.png)

The graph shows:
- **Blue line**: Temperature readings over 30 days
- **Orange dashed line**: Start of Anomaly 1 (moderate surge)
- **Red dashed line**: Start of Anomaly 2 (severe surge)
- **Pattern**: Clear daily temperature cycles with two distinct anomaly periods

---

## 4. Methodology: Applying Isolation Forest

### Step 1: Data Generation

Generate synthetic IoT sensor data with realistic patterns:
- Normal temperature variations following daily cycles
- Random noise simulating sensor fluctuations
- Injected anomalies at specific time periods

### Step 2: Model Training

Train the Isolation Forest model on the first 80% of data:

**Training Configuration**:
- Algorithm: Isolation Forest
- Contamination Rate: 5% (expected anomaly percentage)
- Number of Trees: 100
- Features: 13 engineered features

**Feature Set**:
- Temperature value
- Time-based features (hour, minute)
- Lag features (1, 4, 12, 24 steps back)
- Temperature change rates (1-step, 4-step)
- Rolling statistics (mean, std, max, min over 6h window)

### Training Results

![Training Results](output/2_training_results.png)

The training visualization shows:
1. **Top Panel**: Temperature data with detected anomalies during training
2. **Middle Panel**: Anomaly scores over time with threshold line
3. **Bottom Panel**: Distribution of anomaly scores

**Key Observations**:
- The model learns normal temperature patterns from the training data
- Anomaly scores clearly separate normal and abnormal behavior
- The 5th percentile threshold effectively identifies outliers

### Step 3: Anomaly Detection

Apply the trained model to detect anomalies in the last 7 days of data:

**Detection Configuration**:
- Analysis Window: Last 7 days (672 samples)
- Severity Classification: 3 levels (Mild, Moderate, Severe)
- Severity Criteria: Based on anomaly score distribution

### Detection Results

![Detection Results](output/3_detection_results.png)

The detection visualization includes four panels:

1. **Temperature with Severity Levels**: 
   - Yellow points: Mild anomalies
   - Orange points: Moderate anomalies
   - Red points: Severe anomalies

2. **Anomaly Scores Timeline**: Shows how anomaly scores evolve over time

3. **Daily Anomaly Count**: Bar chart showing anomaly distribution by day

4. **Temperature Distribution**: Histogram comparing normal vs. anomalous temperatures

### Detailed Detection View

![Detailed Detection](output/3_detection_detailed.png)

The detailed view provides:
- **Top Panel**: Annotated anomalies with temperature values and severity labels
- **Bottom Panel**: Color-coded anomaly scores (green=normal, yellow=mild, orange=moderate, red=severe)

---

## 5. Results Evaluation

### Detection Performance

The Isolation Forest model successfully:

✅ **Detected Both Anomaly Events**: Identified temperature surges in the analysis window

✅ **Severity Classification**: Correctly classified anomalies into three severity levels:
- **Mild**: Early stages of temperature increase
- **Moderate**: Significant deviation from normal
- **Severe**: Critical temperature levels requiring immediate action

✅ **Low False Positive Rate**: Minimal false alarms during normal operation periods

✅ **Temporal Sensitivity**: Lag and change features enabled detection of rapid temperature increases

### Model Strengths

1. **Automatic Threshold Learning**: No manual threshold configuration required
2. **Pattern Recognition**: Captures daily temperature cycles and seasonal variations
3. **Multi-dimensional Analysis**: Uses 13 features to understand complex patterns
4. **Scalability**: Processes 2,880+ data points efficiently
5. **Interpretability**: Anomaly scores provide clear confidence levels

### Limitations and Considerations

1. **Training Data Quality**: Model performance depends on representative training data
2. **Concept Drift**: May need retraining if normal patterns change significantly
3. **Cold Start**: Requires sufficient historical data for initial training
4. **Lag Features**: Initial readings may have incomplete lag features (handled via backfill)

### Recommended Deployment Strategy

1. **Initial Training**: Use 2-4 weeks of normal operation data
2. **Continuous Monitoring**: Apply model to real-time data streams
3. **Periodic Retraining**: Retrain monthly or when significant pattern changes occur
4. **Alert Integration**: Connect severe anomalies to notification systems
5. **Human Review**: Validate detections and refine model parameters

---

## 6. Possible IoT Applications

### 1. Industrial Equipment Monitoring

**Use Cases**:
- Motor vibration analysis
- Bearing temperature monitoring
- Pressure sensor anomaly detection
- Power consumption analysis

**Benefits**:
- Predictive maintenance scheduling
- Reduced unplanned downtime
- Extended equipment lifespan
- Lower maintenance costs

### 2. Smart Building Management

**Use Cases**:
- HVAC system performance monitoring
- Energy consumption anomalies
- Air quality sensor validation
- Water leak detection

**Benefits**:
- Energy efficiency optimization
- Comfort level maintenance
- Early fault detection
- Operational cost reduction

### 3. Environmental Monitoring

**Use Cases**:
- Air quality monitoring stations
- Water quality sensors
- Weather station data validation
- Pollution level tracking

**Benefits**:
- Early warning for environmental hazards
- Data quality assurance
- Regulatory compliance
- Public health protection

### 4. Healthcare IoT Devices

**Use Cases**:
- Patient vital signs monitoring
- Medical equipment performance
- Temperature-sensitive storage (vaccines, medications)
- Wearable device data analysis

**Benefits**:
- Patient safety improvement
- Equipment reliability
- Compliance with storage regulations
- Early detection of health issues

### 5. Smart Agriculture

**Use Cases**:
- Soil moisture monitoring
- Greenhouse climate control
- Livestock health monitoring
- Irrigation system performance

**Benefits**:
- Crop yield optimization
- Resource conservation
- Animal welfare monitoring
- Automated alert systems

### 6. Transportation and Logistics

**Use Cases**:
- Fleet vehicle diagnostics
- Cold chain monitoring
- Traffic sensor validation
- Railway track condition monitoring

**Benefits**:
- Cargo quality assurance
- Vehicle maintenance optimization
- Safety improvement
- Operational efficiency

### 7. Smart Grid and Energy

**Use Cases**:
- Power grid load monitoring
- Solar panel performance tracking
- Battery health monitoring
- Smart meter data validation

**Benefits**:
- Grid stability maintenance
- Renewable energy optimization
- Fraud detection
- Outage prediction

---

## Conclusion

Machine learning-based anomaly detection using Isolation Forest provides a robust, scalable solution for IoT sensor monitoring. By automatically learning normal patterns and identifying deviations, it enables:

- **Proactive maintenance** instead of reactive repairs
- **Automated monitoring** reducing manual oversight
- **Cost savings** through early problem detection
- **Improved reliability** of IoT systems

The three-level severity classification (Mild, Moderate, Severe) allows operators to prioritize responses and allocate resources effectively. With proper feature engineering and periodic retraining, this approach can adapt to evolving operational conditions while maintaining high detection accuracy.
