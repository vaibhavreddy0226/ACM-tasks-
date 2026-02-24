## Multi-Variable Weather Forecasting with LSTM   
## Problem Statement  
Predict temperature, humidity, and atmospheric pressure for the next 7 days (168 hours) using 30 days (720 hours) of historical meteorological data from the MPI weather station. This time series forecasting task requires learning temporal dependencies, inter-variable relationships, and patterns.  

--- 
## Dataset Source:  
Max Planck Institute Weather Station, Jena, Germany   
[Link](https://www.bgc-jena.mpg.de/wetter/)   
**Aspect Details**:   
**Period**: January 1 - February 13, 2026 (44 days)   
**Frequency**: 10 minutes (6,291 timesteps)   
**Features**: 21    

---
### Data Preparation & Preprocessing
- **Input sequence length**: 720 hours (≈ 30 days)
- **Output sequence length**: 168 hours (7 days)
- **Target variable**: Air temperature `T (degC)`
- **Features used** (14 total after selection):
  - Atmospheric pressure, temperature, potential temperature, dew point, relative humidity,
  - Vapor pressure, 
  - wind speed (mean & max), wind direction,
  - Solar radiation (SWDR), PAR
  - Cyclic time encodings: hour_sin, hour_cos, day_sin, day_cos

**Preprocessing steps**:
- Resampled raw 10-minute data to hourly using appropriate aggregations (mean, max, sum)
- Standardized all features using `StandardScaler` (saved as `scaler.pkl`)
- Created sliding window sequences explicitly via custom function
- Train / Validation / Test split: **70% / 15% / 15%** (chronological, no shuffling)
  
--- 
## Why This Dataset?  
Professional Quality - Research-grade meteorological station   
Optimal Resolution - Hourly data captures patterns without noise   
Multi-Variate - 14 features enable inter-variable learning   
Complete Data - Zero missing values, continuous time series  
Realistic Task - 7 day forecasting aligns with operational needs

---
## Model architecture diagram
[Baseline model image](https://github.com/vaibhavreddy0226/ACM-tasks-/blob/main/LSTM/Baseline_LSTM_Diagram.jpeg)  
[custom model image](https://github.com/vaibhavreddy0226/ACM-tasks-/blob/main/LSTM/custom_LSTM_Diagram.jpeg)

---
## Training strategy and setup  
**Baseline model**  
- The Base line model uses `Vanilla LSTM` architecture and contain about `230K` paramaters.
- Layers: 2 LSTM layers
- Hidden size: 64
- Dropout: None
- Output: Linear layer mapping last hidden state → 168 future values

**Custom model**  
- The Custom line model uses `deep LSTM with dropouts` architecture and contain about `650K` paramaters.
- Layers: 3 LSTM layers
- Hidden size: 128
- Dropout: 0.2
- Output: Linear layer mapping last hidden state → 168 future values

Both models use the **last hidden state** of the LSTM to predict the entire future sequence (encoder-only style, no explicit decoder).

## Training Setup
- **Framework**: PyTorch
- **Loss function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (learning rate = 0.001, default betas and weight decay)
- **Batch size**: 32
- **Number of epochs**: 20 
- **Data loaders**: `torch.utils.data.DataLoader` with shuffling only on training set
- **Evaluation metric**: MSE on validation and test sets

### Results Snapshot (from training logs)
| Model       | Final Train MSE | Final Val MSE | Test MSE |
|-------------|------------------|---------------|----------|
| Model 1     | ~0.03           | ~0.75         | ~0.8655   |
| Model 2     | ~0.03           | ~0.50         | ~0.8650   |
