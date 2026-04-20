Office Energy Usage Anomaly Detection using PyOD Isolation Forest
===================================================================

Project Structure:
------------------
1_generate_data.py  - Generate synthetic office energy usage data with anomalies
2_training.py       - Train Isolation Forest model on normal data
3_detect.py         - Detect anomalies with severity classification (moderate/severe)

Data Scenario:
--------------
- Time period: 1 month (March 2024)
- Interval: 15 minutes
- Office hours: 9am - 6pm
- Weekday office hours: Average 80 kWh
- Weekday non-office hours: Average 10 kWh
- Weekend: Average 10 kWh

Injected Anomalies (Last Week):
--------------------------------
1. Tuesday night 11pm-12am: Usage rises to average 50 kWh
2. Friday 2pm-4pm: Usage rises to average 100 kWh
3. Saturday 7pm-10pm: Usage rises to average 30 kWh

How to Run:
-----------
1. Install dependencies:
   pip install -r requirements.txt

2. Generate data:
   python 1_generate_data.py
   Output: data/office_energy_data.csv, output/1_generated_data.png

3. Train model:
   python 2_training.py
   Output: model/iforest_model.pkl, model/feature_columns.csv, output/2_training_results.png

4. Detect anomalies:
   python 3_detect.py
   Output: data/detection_results.csv, output/3_detection_results.png, output/3_anomaly_details.png

Directory Structure:
--------------------
/data   - CSV data files
/model  - Trained model files
/output - Visualization PNG files

Anomaly Classification:
-----------------------
- Normal: Anomaly score <= threshold
- Moderate: Anomaly score > threshold and <= 75th percentile of anomalies
- Severe: Anomaly score > 75th percentile of anomalies
