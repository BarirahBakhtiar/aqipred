import hopsworks
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta

# Connect to Hopsworks
project = hopsworks.login(
    project="barirahb",
    api_key_value="fiEMd5rPImQfpLA8.8qWqBnYd7YjnDY0oNriJquZFzcSHVAsBIkdiDGL8bAkRgBIWa8pye4cUjfVFHUAe"
)
fs = project.get_feature_store()
fg = fs.get_feature_group("historical_air_quality", version=1)
df = fg.read()

# Preprocessing
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime')

# Feature engineering
df['pm10_lag1'] = df['pm10'].shift(1)
df['pm10_lag2'] = df['pm10'].shift(2)
df['pm10_avg3'] = df['pm10'].rolling(window=3).mean()
df = df.dropna()

features = ['pm10', 'pm2_5', 'pm10_lag1', 'pm10_lag2', 'pm10_avg3']
pred_input = df[features].iloc[-1]

# Load model
with open('RandomForest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict next 3 days
forecast = []
last_date = df['datetime'].max()

for i in range(1, 4):
    next_date = last_date + timedelta(days=i)
    pred_input['pm10'] += np.random.uniform(-5, 5)
    pred_input['pm2_5'] += np.random.uniform(-2, 2)
    pred_input['pm10_lag2'] = pred_input['pm10_lag1']
    pred_input['pm10_lag1'] = pred_input['pm10']
    pred_input['pm10_avg3'] = (pred_input['pm10'] + pred_input['pm10_lag1'] + pred_input['pm10_lag2']) / 3

    input_array = pred_input.to_numpy().reshape(1, -1)
    pred_aqi = model.predict(input_array)[0]
    forecast.append((next_date.strftime('%Y-%m-%d'), round(pred_aqi)))

forecast_df = pd.DataFrame(forecast, columns=['Date', 'Predicted AQI'])
forecast_df.to_csv('forecast.csv', index=False)
