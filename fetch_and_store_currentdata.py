# fetch_and_store_currentdata.py (updated)
import requests
import pandas as pd
from datetime import datetime
import hopsworks

def fetch_current_air_pollution(lat, lon, api_key):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(url)
    return response.json()

def process_current_data(response_json):
    record = response_json.get("list", [])[0]
    dt_obj = datetime.utcfromtimestamp(record["dt"])
    data = {
        "datetime": dt_obj.isoformat(),
        "aqi": record["main"]["aqi"],
        **record["components"]
    }
    df = pd.DataFrame([data])

    # Fixing schema issues for Hopsworks compatibility
    df["nh3"] = df["nh3"].astype('int64')  # Ensuring nh3 is bigint
    pollutant_cols = ["co", "no", "no2", "o3", "pm2_5", "pm10", "so2"]
    for col in pollutant_cols:
        df[col] = df[col].astype('float64')  # Ensuring pollutants are float

    return df

def save_to_hopsworks(df):
    if df.empty:
        print("❌ DataFrame is empty. Nothing to insert.")
        return

    df["datetime"] = df["datetime"].astype(str)

    project = hopsworks.login(
        project="barirahb",
        api_key_value="fiEMd5rPImQfpLA8.8qWqBnYd7YjnDY0oNriJquZFzcSHVAsBIkdiDGL8bAkRgBIWa8pye4cUjfVFHUAe"
    )
    fs = project.get_feature_store()

    fg = fs.get_feature_group(
        name="historical_air_quality",
        version=1
    )

    fg.insert(df, write_options={"wait_for_job": True})
    print("✅ Current data inserted into feature store.")

if __name__ == '__main__':
    api_key = '38138b1c1a295cef06f2d6918a10e562'
    latitude = 24.8607
    longitude = 67.0011

    current_data = fetch_current_air_pollution(latitude, longitude, api_key)
    df_current = process_current_data(current_data)
    save_to_hopsworks(df_current)
