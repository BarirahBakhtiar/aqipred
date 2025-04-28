import hopsworks
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

# Feature Engineering
df['pm10_lag1'] = df['pm10'].shift(1)
df['pm10_lag2'] = df['pm10'].shift(2)
df['pm10_avg3'] = df['pm10'].rolling(window=3).mean()
df = df.dropna()

# Features and Target
features = ['pm10', 'pm2_5', 'pm10_lag1', 'pm10_lag2', 'pm10_avg3']
target = 'aqi'

X = df[features]
y = df[target]

# Train models
models = {
    'RandomForest': RandomForestRegressor(),
    'XGBoost': XGBRegressor(),
    'Ridge': Ridge()
}

model_results = {}
for name, model in models.items():
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    model_results[name] = {'model': model, 'MSE': mse, 'MAE': mae, 'R2': r2}

# Save models
for name, result in model_results.items():
    with open(f'{name}_model.pkl', 'wb') as f:
        pickle.dump(result['model'], f)

# Save model metrics
results_df = pd.DataFrame.from_dict(model_results, orient='index').drop(columns='model')
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Model'}, inplace=True)
results_df.to_csv('model_results.csv', index=False)
