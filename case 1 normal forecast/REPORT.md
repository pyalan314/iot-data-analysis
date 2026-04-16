# Time Series Forecasting for IoT Applications: A Comparative Study

## 1. Problem Statement

### The Need for Forecasting in IoT Systems

In modern IoT ecosystems, the ability to **predict future states** is as critical as recording historical data. Traditional IoT systems focus primarily on data collection and monitoring, but this reactive approach has significant limitations:

- **Reactive vs. Proactive Decision Making**: Historical data alone only tells us what happened, not what will happen
- **Resource Optimization**: Forecasting enables preemptive resource allocation (e.g., energy management, inventory planning)
- **Anomaly Detection**: Predicted baselines help identify deviations before they become critical issues
- **Cost Reduction**: Anticipating demand patterns allows for optimized operational scheduling
- **Maintenance Planning**: Predictive insights enable preventive maintenance rather than reactive repairs

For example, in building energy management systems, forecasting power consumption allows facility managers to:
- Optimize HVAC scheduling based on predicted occupancy
- Participate in demand response programs
- Identify unusual consumption patterns early
- Plan maintenance during low-demand periods

**The challenge**: IoT time series data exhibits complex patterns including seasonality, trends, and irregular fluctuations that require sophisticated forecasting methods.

---

## 2. Forecasting Methods and Solution Approach

### Overview of Time Series Forecasting Methods

Time series forecasting has evolved through three major paradigms:

**1. Statistical Methods:**
- ARIMA (AutoRegressive Integrated Moving Average)
- Exponential Smoothing (Holt-Winters)
- Prophet (Facebook's forecasting tool)
- Strengths: Interpretable, well-understood theory, fast
- Limitations: Struggle with complex non-linear patterns

**2. Machine Learning Methods:**
- Random Forests, Gradient Boosting (XGBoost, LightGBM)
- Support Vector Regression
- Strengths: Handle non-linearity, feature engineering flexibility
- Limitations: Require manual feature engineering, limited temporal modeling

**3. Deep Learning Methods:**
- LSTM (Long Short-Term Memory), GRU (Gated Recurrent Units)
- Transformer-based models (Informer, Autoformer)
- Specialized architectures (N-BEATS, DLinear)
- Strengths: Automatic feature learning, capture complex patterns
- Limitations: Require large datasets, computationally expensive

**Revolutionary Approach: Foundation Models**

Similar to how ChatGPT revolutionized natural language processing with a single model for all text tasks, **Google TimesFM** represents a paradigm shift: **one pre-trained model for all time series forecasting tasks**. This eliminates the need for task-specific model development and training.

---

### 2.1 DLinear: Fast and Reliable Forecasting

#### Why DLinear?

DLinear was chosen for its **exceptional balance of simplicity, speed, and accuracy**. In production IoT environments, we need models that:
- Train quickly on new data
- Provide reliable predictions
- Are easy to interpret and debug
- Require minimal computational resources

#### Research Foundation

The seminal paper **"Are Transformers Effective for Time Series Forecasting?"** (Zeng et al., 2023, AAAI) challenged conventional wisdom:

**Key Findings:**
- **Simple beats complex**: DLinear outperformed sophisticated transformer models on multiple benchmarks
- **Temporal pattern learning**: Transformers were learning position embeddings rather than true temporal patterns
- **Efficiency matters**: DLinear achieves better results with 10-100x faster training
- **Practical impact**: Simpler models are easier to deploy, maintain, and understand

This research fundamentally shifted the field toward **efficient, interpretable models** without sacrificing accuracy.

#### How DLinear Works

**Architecture:**
1. **Decomposition**: Separates time series into trend and seasonal components using moving average
2. **Dual Linear Layers**: Applies separate linear transformations to each component
3. **Recombination**: Sums the forecasts from both components

```
Input → [Decomposition] → Trend → [Linear Layer] → Trend Forecast
                       ↓                                    ↓
                    Seasonal → [Linear Layer] → Seasonal Forecast
                                                            ↓
                                                    [Sum] → Final Forecast
```

#### Implementation: NeuralForecast

We use the **NeuralForecast** library (by Nixtla), which provides:
- Production-ready implementation of DLinear
- Unified API for multiple forecasting models
- Automatic hyperparameter optimization
- GPU acceleration support
- Integration with pandas DataFrames

**Configuration:**
- Input window: 672 steps (7 days of 15-minute data)
- Forecast horizon: 28 steps (7 hours)
- Loss function: Mean Absolute Error (MAE)
- Scaler: Robust scaling (handles outliers)

#### Pros and Cons

**Advantages:**
✓ **Speed**: Trains in minutes, inference in milliseconds  
✓ **Interpretability**: Clear decomposition into trend and seasonal components  
✓ **Reliability**: Consistent performance across different datasets  
✓ **Low resource requirements**: Runs on CPU, minimal memory  
✓ **Easy deployment**: Simple architecture, few dependencies  
✓ **Proven performance**: State-of-the-art results on standard benchmarks  

**Limitations:**
✗ **Requires training**: Needs historical data and retraining for each new time series  
✗ **Linear assumptions**: May struggle with highly non-linear patterns  
✗ **No built-in uncertainty**: Confidence intervals require additional estimation  
✗ **Fixed horizon**: Trained for specific forecast length  
✗ **Limited external features**: Primarily univariate, limited multivariate support  

---

### 2.2 Google TimesFM: The Foundation Model Revolution

#### Why TimesFM is Revolutionary

TimesFM represents a **paradigm shift** in time series forecasting, analogous to the GPT revolution in NLP:

**Traditional Approach:**
- Train a separate model for each time series
- Requires sufficient historical data
- Model tuning for each use case
- Retraining when patterns change

**TimesFM Approach:**
- **One model, all time series**: Pre-trained on 100 billion real-world data points
- **Zero-shot forecasting**: No training required for new time series
- **Universal patterns**: Learns generalizable temporal patterns across domains
- **Instant deployment**: Ready to use out-of-the-box

This is the **"ChatGPT moment"** for time series forecasting.

#### Solving Traditional Difficulties

TimesFM addresses fundamental challenges in time series forecasting:

**1. Cold Start Problem**
- **Traditional difficulty**: New time series lack training data
- **TimesFM solution**: Pre-trained knowledge transfers to new series immediately

**2. Distribution Shift**
- **Traditional difficulty**: Models fail when patterns change
- **TimesFM solution**: Robust to different data distributions through diverse pre-training

**3. Uncertainty Quantification**
- **Traditional difficulty**: Point forecasts without confidence measures
- **TimesFM solution**: Built-in quantile predictions (10th to 90th percentiles)

**4. Scalability**
- **Traditional difficulty**: Training hundreds of models for different time series
- **TimesFM solution**: Single model handles all series

**5. Domain Expertise**
- **Traditional difficulty**: Requires time series expertise for model selection and tuning
- **TimesFM solution**: Pre-configured, works across domains without tuning

#### How TimesFM Works

**Architecture:**
- **Patch-based processing**: Divides time series into patches (similar to vision transformers)
- **Decoder-only transformer**: 20 layers, 1280 dimensions (200M parameters)
- **Residual connections**: Enables learning from very long contexts
- **Quantile head**: Predicts multiple quantiles simultaneously

**Pre-training:**
- Trained on Google's massive time series corpus
- Diverse domains: retail, energy, finance, web traffic, sensors
- 100+ billion time points across millions of series
- Self-supervised learning on forecasting tasks

**Inference:**
- Context window: Up to 1024 historical points
- Forecast horizon: Up to 256 steps ahead
- Frequency-agnostic: Works with any time granularity
- Batch processing: Efficient for multiple series

#### Implementation

We use **TimesFM 2.5** (200M parameter PyTorch model):

```python
- Model: google/timesfm-2.5-200m-pytorch
- Context: 672 historical observations
- Horizon: 28 future steps
- Quantiles: 10th, 20th, ..., 90th percentiles
- Backend: PyTorch with CPU/GPU support
```

**Configuration Options:**
- `normalize_inputs`: Automatic input normalization
- `use_continuous_quantile_head`: Smooth quantile predictions
- `force_flip_invariance`: Handles trend reversals
- `infer_is_positive`: Optimizes for non-negative values (e.g., power consumption)
- `fix_quantile_crossing`: Ensures quantile ordering

#### Pros and Cons

**Advantages:**
✓ **Zero-shot capability**: No training required, instant deployment  
✓ **Uncertainty quantification**: Built-in probabilistic forecasts  
✓ **Robust generalization**: Works across diverse domains and patterns  
✓ **Scalability**: Single model for unlimited time series  
✓ **State-of-the-art**: Competitive with or better than specialized models  
✓ **Continuous improvement**: Benefits from Google's ongoing research  
✓ **Handles missing data**: Robust to gaps in historical data  

**Limitations:**
✗ **Computational cost**: Requires more resources than simple models (200M parameters)  
✗ **Black box**: Less interpretable than decomposition-based models  
✗ **Model size**: ~800MB download, larger memory footprint  
✗ **Inference speed**: Slower than lightweight models like DLinear  
✗ **Limited customization**: Pre-trained, cannot easily adapt to specific domain knowledge  
✗ **Dependency**: Relies on Google's model availability and updates

---

## 3. Example Scenario: Office Tower Power Consumption

### Scenario Description

We simulate a **commercial office tower** with realistic power consumption patterns:

**Data Characteristics:**
- **Duration**: 1 year of historical data
- **Frequency**: 15-minute intervals (96 readings per day)
- **Total records**: 35,136 data points

**Consumption Patterns:**

1. **Weekday Pattern (Monday-Friday)**:
   - Base load: ~100 kW (overnight)
   - Ramp-up: 7:00-9:00 AM (gradual increase)
   - Peak hours: 9:00 AM - 6:00 PM (300-450 kW)
   - Ramp-down: 6:00-8:00 PM (gradual decrease)

2. **Weekend Pattern (Saturday-Sunday)**:
   - Minimal operations: 50-120 kW
   - Slight midday increase (10:00 AM - 4:00 PM)

3. **Additional Factors**:
   - Seasonal variation: ±20% based on time of year
   - Random noise: Simulates real-world fluctuations

### Data Visualization

![Data Generation Overview](output/data_generation_overview.png)

**Figure 1**: Sample week showing distinct weekday vs. weekend patterns. The top panel shows temporal progression, while the bottom panel illustrates the clear separation between weekday (blue) and weekend (orange) consumption by hour of day.

**Key Observations:**
- Weekday office hours show consistent high consumption (350-450 kW)
- Weekend consumption remains low and stable (50-100 kW)
- Clear daily cycles with morning ramp-up and evening ramp-down
- Realistic noise and variability in the data

---

## 4. Methodology and Application

### 4.1 Data Preparation

**Training Dataset:**
- Full year of historical data (365 days)
- Preprocessed into NeuralForecast format with unique_id, timestamp (ds), and value (y)
- No missing values or outliers requiring treatment

![Training Overview](output/training_overview.png)

**Figure 2**: Training data characteristics. Top panel shows the last 30 days of consumption data, demonstrating consistent patterns. Bottom panel shows the power distribution across the full dataset, with clear bimodal distribution reflecting weekday vs. weekend consumption.

### 4.2 Model Training

**DLinear Training:**
```python
- Input window: 672 steps (7 days)
- Forecast horizon: 28 steps (7 hours)
- Training steps: 1,000 epochs
- Batch size: 32
- Learning rate: 0.001
- Loss function: Mean Absolute Error (MAE)
```

**TimesFM Application:**
```python
- Pre-trained model: google/timesfm-2.5-200m-pytorch
- Zero-shot forecasting (no training required)
- Context: Last 672 observations
- Quantile predictions: 10th, 50th, 90th percentiles
```

### 4.3 Forecasting Task

**Objective**: Predict office tower power consumption for a weekday afternoon

**Forecast Specification:**
- Start time: 11:00 AM (last complete weekday)
- End time: 6:00 PM
- Duration: 7 hours (28 intervals of 15 minutes)
- Context: Previous 7 days of data

**Visualization Context:**
- Historical data: 7 days prior to forecast start
- Forecast period: 11:00 AM - 6:00 PM
- Confidence/uncertainty intervals for both models

---

## 5. Results and Evaluation

### 5.1 Forecast Comparison

![Forecast Visualization](output/forecast_visualization.png)

**Figure 3**: Comparative forecast results. Top panel shows DLinear predictions with 90% confidence interval (red shaded area). Bottom panel shows TimesFM predictions with 10th-90th percentile range (purple shaded area). Both models show the 7-day historical context (blue line) and forecast starting point (green vertical line).

### 5.2 Model Performance Analysis

**DLinear Model:**
- **Point Forecast**: Smooth, trend-following predictions
- **Confidence Interval**: Symmetric ±90% CI based on historical variance
- **Pattern Recognition**: Captures the afternoon decline in office consumption
- **Strengths**: Fast inference, interpretable results, consistent predictions
- **Limitations**: Assumes constant uncertainty across forecast horizon

**TimesFM Model:**
- **Point Forecast**: More conservative, closer to mean values
- **Uncertainty Quantification**: Asymmetric quantile predictions, wider intervals
- **Pattern Recognition**: Captures general trend but less aggressive in following recent patterns
- **Strengths**: Built-in uncertainty, no training required, robust to distribution shifts
- **Limitations**: May be overly conservative for well-behaved patterns

### 5.3 Quantitative Comparison

Based on the forecast results:

| Metric | DLinear | TimesFM |
|--------|---------|---------|
| Average Prediction | Follows recent trend | More conservative |
| Uncertainty Width | Narrower (constant variance) | Wider (data-driven quantiles) |
| Computation Time | ~2-3 seconds (after training) | ~5-8 seconds (zero-shot) |
| Training Required | Yes (~2-5 minutes) | No (pre-trained) |
| Interpretability | High (decomposition) | Medium (foundation model) |

**Mean Absolute Difference**: The average difference between DLinear and TimesFM predictions provides insight into model agreement. Smaller differences suggest both models capture similar underlying patterns.

### 5.4 Practical Implications

**When to use DLinear:**
- Stable, well-understood patterns
- Need for fast, frequent retraining
- Interpretability is important
- Limited computational resources

**When to use TimesFM:**
- New time series with limited history
- Need uncertainty quantification
- Diverse data sources
- Zero-shot capability valued

---

## 6. Practical Applications of Power Forecasting

Power consumption forecasting enables **proactive building management** that goes beyond simple monitoring. By predicting future energy demand, facility managers can optimize operations, reduce costs, and improve occupant comfort.

### 6.1 Smart HVAC Control

**Predictive Climate Control:**

**Morning Ramp-Up Optimization:**
- **Forecast insight**: Power consumption increases at 7:00-9:00 AM indicate office occupancy rising
- **Action**: Pre-cool or pre-heat the building 30-60 minutes before peak occupancy
- **Benefit**: Achieve target temperature when employees arrive, avoiding uncomfortable conditions
- **Energy savings**: Gradual temperature adjustment is more efficient than rapid changes

**Peak Hour Management:**
- **Forecast insight**: High power demand during 9:00 AM - 6:00 PM signals full occupancy
- **Action**: Increase central air conditioning output in advance to maintain comfort
- **Benefit**: Prevent temperature spikes during peak occupancy periods
- **Cost optimization**: Avoid emergency cooling which consumes more energy

**Evening Ramp-Down:**
- **Forecast insight**: Declining power consumption after 6:00 PM indicates people leaving
- **Action**: Gradually reduce HVAC output as occupancy decreases
- **Benefit**: Significant energy savings without compromising comfort for remaining occupants
- **Smart zones**: Shut down HVAC in empty zones while maintaining service in occupied areas

**Weekend Mode:**
- **Forecast insight**: Low weekend power consumption indicates minimal occupancy
- **Action**: Switch to maintenance-only HVAC mode (minimal heating/cooling)
- **Benefit**: 60-80% reduction in weekend HVAC energy costs
- **Exception handling**: Quick response if unexpected occupancy detected

### 6.2 Demand Response and Grid Integration

**Peak Demand Reduction:**
- **Forecast insight**: Predict when building will hit peak power consumption
- **Action**: Pre-cool building before peak hours, then reduce HVAC during peak pricing periods
- **Benefit**: Lower electricity bills through time-of-use optimization
- **Grid support**: Reduce strain on electrical grid during high-demand periods

**Load Shifting:**
- **Forecast insight**: Identify low-demand periods (e.g., early morning, weekends)
- **Action**: Schedule energy-intensive tasks (water heating, equipment charging) during off-peak hours
- **Benefit**: Take advantage of lower electricity rates
- **Example**: Charge battery storage systems at night, discharge during peak hours

**Demand Response Programs:**
- **Forecast insight**: Predict baseline consumption to calculate curtailment capacity
- **Action**: Participate in utility demand response programs for financial incentives
- **Benefit**: Revenue generation by reducing consumption during grid emergencies
- **Automation**: Automatically respond to utility signals based on forecasted flexibility

### 6.3 Predictive Maintenance

**Equipment Health Monitoring:**
- **Forecast insight**: Deviation between predicted and actual consumption indicates equipment issues
- **Action**: Schedule maintenance before complete failure occurs
- **Benefit**: Prevent costly emergency repairs and downtime
- **Example**: HVAC system consuming 20% more than forecast suggests filter replacement or refrigerant leak

**Optimal Maintenance Scheduling:**
- **Forecast insight**: Identify low-occupancy periods (weekends, holidays)
- **Action**: Schedule maintenance during predicted low-demand windows
- **Benefit**: Minimize disruption to building operations
- **Planning**: Coordinate multiple maintenance tasks during same low-demand period

### 6.4 Energy Cost Optimization

**Real-Time Pricing Response:**
- **Forecast insight**: Predict consumption during different pricing periods
- **Action**: Adjust operations to minimize costs under time-of-use tariffs
- **Benefit**: 15-30% reduction in electricity costs
- **Strategy**: Shift flexible loads to low-price periods

**Budget Planning:**
- **Forecast insight**: Long-term consumption predictions for upcoming months
- **Action**: Accurate energy budget allocation and cost forecasting
- **Benefit**: Better financial planning and cost control
- **Reporting**: Provide stakeholders with reliable energy cost projections

**Renewable Energy Integration:**
- **Forecast insight**: Predict building consumption alongside solar generation forecast
- **Action**: Optimize battery storage charging/discharging schedules
- **Benefit**: Maximize self-consumption of renewable energy
- **Grid independence**: Reduce reliance on grid power during peak pricing

### 6.5 Occupancy and Space Management

**Dynamic Space Allocation:**
- **Forecast insight**: Power patterns reveal actual space utilization
- **Action**: Identify underutilized areas for repurposing or consolidation
- **Benefit**: Optimize real estate usage and reduce unnecessary HVAC costs
- **Example**: Consistently low power in certain floors suggests opportunity for hot-desking

**Meeting Room Optimization:**
- **Forecast insight**: Predict meeting room usage based on power consumption patterns
- **Action**: Implement smart booking systems and HVAC control per room
- **Benefit**: Condition only occupied meeting rooms
- **Savings**: 30-40% reduction in meeting room HVAC costs

**Workforce Planning:**
- **Forecast insight**: Power consumption trends indicate office attendance patterns
- **Action**: Inform HR decisions on remote work policies and office capacity
- **Benefit**: Align facility operations with actual occupancy needs
- **Flexibility**: Adapt to hybrid work models efficiently

### 6.6 Anomaly Detection and Security

**Abnormal Consumption Alerts:**
- **Forecast insight**: Actual consumption significantly different from forecast
- **Action**: Trigger automated alerts for investigation
- **Benefit**: Early detection of equipment failures, leaks, or unauthorized usage
- **Security**: Identify unusual after-hours activity that may indicate security issues

**Energy Theft Detection:**
- **Forecast insight**: Unexplained consumption patterns inconsistent with occupancy
- **Action**: Investigate potential meter tampering or unauthorized connections
- **Benefit**: Prevent revenue loss and safety hazards
- **Compliance**: Ensure accurate billing and regulatory compliance

### 6.7 Sustainability and ESG Reporting

**Carbon Footprint Reduction:**
- **Forecast insight**: Predict high-consumption periods to target reduction efforts
- **Action**: Implement energy-saving measures during forecasted peak periods
- **Benefit**: Measurable reduction in carbon emissions
- **Reporting**: Accurate data for ESG (Environmental, Social, Governance) reports

**Renewable Energy Procurement:**
- **Forecast insight**: Long-term consumption predictions guide renewable energy contracts
- **Action**: Size solar installations or wind power agreements appropriately
- **Benefit**: Optimal investment in renewable energy infrastructure
- **ROI**: Maximize return on green energy investments

**Sustainability Goals Tracking:**
- **Forecast insight**: Project future consumption against sustainability targets
- **Action**: Identify gaps and implement corrective measures proactively
- **Benefit**: Stay on track to meet corporate sustainability commitments
- **Transparency**: Provide stakeholders with credible progress reports

---

## Conclusion

This study demonstrates that both simple (DLinear) and sophisticated (TimesFM) approaches can effectively forecast IoT time series data. The choice between models depends on specific requirements:

- **DLinear** excels in speed, interpretability, and performance on stable patterns
- **TimesFM** provides zero-shot capability and robust uncertainty quantification

For production IoT systems, a **hybrid approach** combining multiple models with proper uncertainty quantification offers the best balance of accuracy, reliability, and operational flexibility. The key is not just predicting the future, but providing actionable insights that enable proactive decision-making in IoT ecosystems.

---

## References

1. Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). "Are Transformers Effective for Time Series Forecasting?" *AAAI Conference on Artificial Intelligence*.

2. Das, A., Kong, W., Leach, A., Mathur, S., Sen, R., & Yu, R. (2024). "A decoder-only foundation model for time-series forecasting." *Google Research*.

3. Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J., & Montero-Manso, P. (2021). "Monash Time Series Forecasting Archive."

4. NeuralForecast Documentation: https://nixtla.github.io/neuralforecast/

5. TimesFM Repository: https://github.com/google-research/timesfm
