import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and scaler
try:
    dt_model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    model_columns = joblib.load("weather_features_encoded_cols.joblib")
except FileNotFoundError:
    st.error("Model or scaler files not found. Please ensure all required files are in the same directory.")
    st.stop()

# Reverse engineer weather conditions
unique_weather_conditions = [col.replace('Weather_Condition_', '') for col in model_columns if col.startswith('Weather_Condition_')]
unique_weather_conditions.insert(0, 'Fair')  # base case if drop_first=True

st.title('Accident Severity Prediction')

# Select feature type
feature_type = st.selectbox(
    "Enter the type of features you want to predict with:",
    ["Weather", "Road Type"]
)

if feature_type == "Weather":
    st.write("Enter the current weather conditions to predict accident severity.")

    temperature = st.slider("Temperature (F)", min_value=-45.0, max_value=196.0, step=0.1)
    wind_chill = st.slider("Wind_Chill (F)", min_value=-63.0, max_value=196.0, step=0.1)
    humidity = st.slider("Humidity (%)", min_value=1.0, max_value=100.0, step=0.1)
    pressure = st.slider("Pressure (in)", min_value=0.0, max_value=58.63, step=0.1)
    visibility = st.slider("Visibility (mi)", min_value=0.0, max_value=100.0, step=0.1)
    wind_speed = st.slider("Wind_Speed (mph)", min_value=0.0, max_value=1087.0, step=0.1)
    precipitation = st.slider("Precipitation (in)", min_value=0.0, max_value=24.0, step=0.1)
    weather_condition = st.selectbox('Weather Condition', unique_weather_conditions)

    if st.button('Predict Severity'):
        input_data = pd.DataFrame([[
            temperature, wind_chill, humidity, pressure, visibility, wind_speed, precipitation, weather_condition
        ]], columns=['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
                     'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition'])

        input_data_encoded = pd.get_dummies(input_data, columns=['Weather_Condition'], drop_first=True)

        for col in model_columns:
            if col not in input_data_encoded.columns:
                input_data_encoded[col] = 0

        input_data_encoded = input_data_encoded[model_columns]

        numerical_features = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
                              'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']
        input_data_encoded[numerical_features] = scaler.transform(input_data_encoded[numerical_features])

        prediction = dt_model.predict(input_data_encoded)
        st.write(f"Predicted Accident Severity: {prediction[0]}")

elif feature_type == "Road Type":
    st.write("Enter the road type to predict accident severity.")

    road_type = st.selectbox("Road Type", ['Crossing', 'Junction', 'Roundabout', 'Station'])

    if st.button('Predict Severity'):
        input_data = pd.DataFrame([[road_type]], columns=['Road_Type'])

        # One-hot encode Road_Type
        input_data_encoded = pd.get_dummies(input_data, columns=['Road_Type'], drop_first=True)

        # Align with training columns (you must have trained Road_Type features included in model_columns)
        for col in model_columns:
            if col not in input_data_encoded.columns:
                input_data_encoded[col] = 0

        input_data_encoded = input_data_encoded[model_columns]

        prediction = dt_model.predict(input_data_encoded)
        st.write(f"Predicted Accident Severity: {prediction[0]}")
