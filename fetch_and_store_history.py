import requests
import pandas as pd
from datetime import datetime, timedelta
import hopsworks

def fetch_historical_air_pollution(lat, lon, start, end, api_key):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start}&end={end}&appid={api_key}"
    response = requests.get(url)
    return response.json()

def process_historical_data(response_json):
    records = []
    for entry in response_json.get("list", []):
        dt_obj = datetime.utcfromtimestamp(entry["dt"])
        data = {
            "datetime": dt_obj.isoformat(),
            "aqi": entry["main"]["aqi"],
            **entry["components"]
        }
        records.append(data)
    return pd.DataFrame(records)

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

    fg = fs.get_or_create_feature_group(
        name="historical_air_quality",
        version=1,
        primary_key=["datetime"],
        description="Historical air pollution data from OpenWeather",
        online_enabled=False
    )
    fg.insert(df)
    print("✅ Historical data inserted into feature store.")

if __name__ == '__main__':
    api_key = '38138b1c1a295cef06f2d6918a10e562'
    latitude = 24.8607
    longitude = 67.0011

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=3)
    start_unix = int(start_date.timestamp())
    end_unix = int(end_date.timestamp())

    historical_data = fetch_historical_air_pollution(latitude, longitude, start_unix, end_unix, api_key)
    df_historical = process_historical_data(historical_data)
    save_to_hopsworks(df_historical)
