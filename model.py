#streamlitrun.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import hopsworks
import pickle
import numpy as np
from datetime import timedelta

st.set_page_config(page_title="AQI Prediction Dashboard", layout="wide")

st.title("Air Quality Forecast Dashboard")

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

# Load model results
model_results = pd.read_csv('model_results.csv')
model_results['MSE'] = pd.to_numeric(model_results['MSE'], errors='coerce')
model_results['MAE'] = pd.to_numeric(model_results['MAE'], errors='coerce')
model_results['R2'] = pd.to_numeric(model_results['R2'], errors='coerce')
model_results.set_index('Model', inplace=True)

# Layout: 1 Single Page Dashboard
st.sidebar.title("Dashboard Sections")
section = st.sidebar.radio("Jump to", ["Overview", "Model Performance", "Forecasting"])

# --- Section 1: Overview ---
if section == "Overview":
    st.header("Latest Air Quality Data")
    st.dataframe(df.tail(10))

    st.header("AQI Trends")
    col1, col2 = st.columns([3, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(df['datetime'], df['aqi'], label='AQI', color='blue')
        ax.set_xlabel("Date")
        ax.set_ylabel("AQI")
        ax.set_title("Air Quality Index Over Time")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.info("This graph shows the AQI trend over time.")

    st.header("AQI Distribution")
    col1, col2 = st.columns([3, 1])

    with col1:
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.hist(df['aqi'], bins=30, color='skyblue', edgecolor='black')
        ax2.set_title('Distribution of AQI Values')
        ax2.set_xlabel('AQI')
        ax2.set_ylabel('Frequency')
        st.pyplot(fig2)

    st.header("PM10 vs PM2.5 Levels")
    col1, col2 = st.columns([3, 1])

    with col1:
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        ax3.scatter(df['pm10'], df['pm2_5'], alpha=0.7)
        ax3.set_xlabel("PM10")
        ax3.set_ylabel("PM2.5")
        ax3.set_title("PM10 vs PM2.5 Scatter Plot")
        st.pyplot(fig3)

    st.header("Pollutants Correlation Heatmap")
    col1, col2 = st.columns([3, 1])

    with col1:
        fig4, ax4 = plt.subplots(figsize=(6, 3))
        corr = df[['aqi', 'pm10', 'pm2_5', 'co', 'no', 'no2', 'o3', 'so2']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax4)
        ax4.set_title("Correlation Heatmap")
        st.pyplot(fig4)

# --- Section 2: Model Performance ---
elif section == "Model Performance":
    st.header("Model Performance Metrics")

    st.dataframe(model_results)

    st.subheader("Detailed Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Best R² Model", value=model_results['R2'].idxmax(), delta=f"{model_results['R2'].max():.2f}")

    with col2:
        st.metric(label="Lowest MSE Model", value=model_results['MSE'].idxmin(), delta=f"{model_results['MSE'].min():.4f}")

    with col3:
        st.metric(label="Lowest MAE Model", value=model_results['MAE'].idxmin(), delta=f"{model_results['MAE'].min():.4f}")

# --- Section 3: Forecast ---
elif section == "Forecasting":
    st.header("3-Day AQI Forecast from All Models")

    latest = df.copy()
    latest['datetime'] = pd.to_datetime(latest['datetime'])
    latest = latest.sort_values('datetime')

    latest['pm10_lag1'] = latest['pm10'].shift(1)
    latest['pm10_lag2'] = latest['pm10'].shift(2)
    latest['pm10_avg3'] = latest['pm10'].rolling(window=3).mean()
    latest = latest.dropna()

    features = ['pm10', 'pm2_5', 'pm10_lag1', 'pm10_lag2', 'pm10_avg3']
    pred_input = latest[features].iloc[-1]

    forecast_results = []
    hazardous_detected = False

    models_to_forecast = ['RandomForest_model.pkl', 'XGBoost_model.pkl', 'Ridge_model.pkl']

    model_forecasts = {}

    for model_file in models_to_forecast:
        model_name = model_file.split('_')[0]
        with open(model_file, 'rb') as f:
            model = pickle.load(f)

        predictions = []
        last_date = latest['datetime'].max()
        temp_input = pred_input.copy()

        for i in range(1, 4):
            next_date = last_date + timedelta(days=i)

            temp_input['pm10'] += np.random.uniform(-8, 8)
            temp_input['pm2_5'] += np.random.uniform(-4, 4)
            temp_input['pm10_lag2'] = temp_input['pm10_lag1']
            temp_input['pm10_lag1'] = temp_input['pm10']
            temp_input['pm10_avg3'] = (temp_input['pm10'] + temp_input['pm10_lag1'] + temp_input['pm10_lag2']) / 3

            pred_input_arr = temp_input.to_numpy().reshape(1, -1)
            pred_aqi = model.predict(pred_input_arr)[0]
            pred_aqi_rounded = round(pred_aqi)

            if pred_aqi_rounded > 300:
                hazardous_detected = True

            predictions.append((next_date.strftime('%Y-%m-%d'), model_name, pred_aqi_rounded))

        forecast_results.extend(predictions)
        model_forecasts[model_name] = predictions

    forecast_df = pd.DataFrame(forecast_results, columns=['Date', 'Model', 'Predicted AQI'])

    st.dataframe(forecast_df)

    st.subheader("Forecast Comparison Across Models")

    forecast_chart = alt.Chart(forecast_df).mark_line(point=True).encode(
        x='Date:T',
        y='Predicted AQI:Q',
        color='Model:N'
    ).properties(
        width=800,
        height=400
    )

    st.altair_chart(forecast_chart)

    st.subheader("Individual Model Forecasts")
    for model_name, preds in model_forecasts.items():
        st.markdown(f"### {model_name} Forecast")
        model_df = pd.DataFrame(preds, columns=['Date', 'Model', 'Predicted AQI'])

        col1, col2 = st.columns([3, 1])

        with col1:
            fig_model, ax_model = plt.subplots(figsize=(6, 3))
            ax_model.plot(model_df['Date'], model_df['Predicted AQI'], marker='o')
            ax_model.set_xlabel('Date')
            ax_model.set_ylabel('Predicted AQI')
            ax_model.set_title(f'{model_name} Forecasted AQI')
            plt.xticks(rotation=45)
            st.pyplot(fig_model)

            st.info(f"Forecast generated using {model_name}.")

    if hazardous_detected:
        st.error("Hazardous AQI Level forecasted! Please take precautions.")
    else:
        st.success("No hazardous AQI levels predicted in the next 3 days.")
