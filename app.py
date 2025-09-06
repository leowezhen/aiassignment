import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title('Accident Severity Prediction')

# Feature type selector
feature_type = st.selectbox(
    "Enter the type of features you want to predict with.",
    ["Weather", "Road Type"],
    index=0
)

# Weather-based prediction
if feature_type == "Weather":
    try:
        dt_model = joblib.load('model.joblib')
        scaler = joblib.load('scaler.joblib')
        model_columns = joblib.load("weather_features_encoded_cols.joblib")
    except FileNotFoundError:
        st.error("Weather model files not found.")
        st.stop()

    unique_weather_conditions = [col.replace('Weather_Condition_', '') for col in model_columns if col.startswith('Weather_Condition_')]
    unique_weather_conditions.insert(0, 'Fair')

    temperature = st.slider("Temperature (F)", min_value=-45.0, max_value=196.0, step=0.1)
    wind_chill = st.slider("Wind_Chill (F)", min_value=-63.0, max_value=196.0, step=0.1)
    humidity = st.slider("Humidity (%)", min_value=1.0, max_value=100.0, step=0.1)
    pressure = st.slider("Pressure (in)", min_value=0.0, max_value=58.63, step=0.1)
    visibility = st.slider("Visibility (mi)", min_value=0.0, max_value=100.0, step=0.1)
    wind_speed = st.slider("Wind_Speed (mph)", min_value=0.0, max_value=1087.0, step=0.1)
    precipitation = st.slider("Precipitation (in)", min_value=0.0, max_value=24.0, step=0.1)

    weather_condition = st.selectbox('Weather Condition', unique_weather_conditions)

    if st.button('Predict Severity (Weather)'):
        input_data = pd.DataFrame([{
            'Temperature(F)': temperature,
            'Wind_Chill(F)': wind_chill,
            'Humidity(%)': humidity,
            'Pressure(in)': pressure,
            'Visibility(mi)': visibility,
            'Wind_Speed(mph)': wind_speed,
            'Precipitation(in)': precipitation,
            'Weather_Condition': weather_condition
        }])

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

# Road-type prediction
elif feature_type == "Road Type":
    try:
        road_model = joblib.load("roadType.joblib")
    except FileNotFoundError:
        st.error("Road type model file 'roadType.joblib' not found.")
        st.stop()

    st.write("Select the road characteristics to predict accident severity:")
    crossing = st.checkbox("Crossing")
    junction = st.checkbox("Junction")
    roundabout = st.checkbox("Roundabout")
    station = st.checkbox("Station")

    if st.button("Predict Severity (Road Type)"):
        input_data = pd.DataFrame([{
            'Crossing': int(crossing),
            'Junction': int(junction),
            'Roundabout': int(roundabout),
            'Station': int(station)
        }])

        prediction = road_model.predict(input_data)
        st.write(f"Predicted Accident Severity: {prediction[0]}")
