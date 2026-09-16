---
title: NYC Taxi Fare Predictor
emoji: 🚕
colorFrom: yellow
colorTo: gray
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
---

# NYC Taxi Fare Predictor

A deep feedforward neural network (DNN/MLP) that predicts New York City taxi fares
from pickup/drop-off coordinates, trip time, and passenger count.

Built for Lab Assignment 02 — Deep Learning.

## Model

- Architecture: Dense(128) → BatchNorm → Dropout → Dense(64) → BatchNorm → Dropout →
  Dense(32) → BatchNorm → Dropout → Dense(1, linear)
- Loss: Huber (delta=5.0), chosen for robustness to fare outliers
- Optimizer: Adam
- Features: 20 engineered features including Haversine trip distance, temporal features
  (hour, day of week, month, year, weekend and rush-hour flags), coordinate differences,
  and distances to JFK/LaGuardia

## Data

[NYC Taxi Fare Prediction](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction)
(Kaggle), fares from 2009–2015.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface + feature engineering pipeline |
| `final_taxi_fare_model.keras` | Trained Keras model |
| `scaler.pkl` | Fitted StandardScaler from training |
| `requirements.txt` | Python dependencies |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
