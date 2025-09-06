import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ================================
# Load Models and Preprocessing
# ================================
try:
    # Weather model
    dt_model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    model_columns = joblib.load("weather_features_encoded_cols.joblib")

    # Road Type model
    road_model = joblib.load("roadType.joblib")

except FileNotFoundError:
    st.error("❌ Model or scaler files not found. Ensure all required joblib files are in the same directory.")
    st.stop()

# Extract unique weather conditions from model columns
unique_weather_conditions = [col.replace('Weather_Condition_', '') for col in model_columns if col.startswith('Weather_Condition_')]
unique_weather_conditions.insert(0, 'Fair')  # add the base case if dropped during encoding

# Define numerical features
numerical_features = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
                      'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']

# ================================
# Streamlit App
# ================================
st.title('Accident Severity Prediction')

feature_type = st.selectbox(
    "Select the type of features you want to predict with:",
    ["Weather", "Road Type"]
)

# ================================
# Weather Prediction
# ================================
if feature_type == "Weather":
    st.subheader("Weather-based Features")

    temperature = st.slider("Temperature (F)", min_value=-45.0, max_value=196.0, step=0.1)
    wind_chill = st.slider("Wind_Chill (F)", min_value=-63.0, max_value=196.0, step=0.1)
    humidity = st.slider("Humidity (%)", min_value=1.0, max_value=100.0, step=0.1)
    pressure = st.slider("Pressure (in)", min_value=0.0, max_value=58.63, step=0.1)
    visibility = st.slider("Visibility (mi)", min_value=0.0, max_value=100.0, step=0.1)
    wind_speed = st.slider("Wind_Speed (mph)", min_value=0.0, max_value=1087.0, step=0.1)
    precipitation = st.slider("Precipitation (in)", min_value=0.0, max_value=24.0, step=0.1)
    weather_condition = st.selectbox('Weather Condition', unique_weather_conditions)

    if st.button('Predict Weather Severity'):
        # Create input DataFrame
        input_data = pd.DataFrame([[temperature, wind_chill, humidity, pressure, visibility,
                                    wind_speed, precipitation, weather_condition]],
                                   columns=['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
                                            'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
                                            'Precipitation(in)', 'Weather_Condition'])

        # One-hot encode weather condition
        input_data_encoded = pd.get_dummies(input_data, columns=['Weather_Condition'], drop_first=True)

        # Ensure training columns match
        for col in model_columns:
            if col not in input_data_encoded.columns:
                input_data_encoded[col] = 0
        input_data_encoded = input_data_encoded[model_columns]

        # Scale numeric features
        input_data_encoded[numerical_features] = scaler.transform(input_data_encoded[numerical_features])

        # Predict
        prediction = dt_model.predict(input_data_encoded)
        st.success(f"🌦️ Predicted Accident Severity: {prediction[0]}")

# ================================
# Road Type Prediction
# ================================
elif feature_type == "Road Type":
    st.subheader("Road Type-based Features")

    crossing = st.checkbox("Crossing")
    junction = st.checkbox("Junction")
    roundabout = st.checkbox("Roundabout")
    station = st.checkbox("Station")

    if st.button("Predict Road Type Severity"):
        if not (crossing or junction or roundabout or station):
            st.warning("⚠️ Please select at least one road type before predicting.")
        else:
            # Build input DataFrame with binary values
            input_data = pd.DataFrame([{
                'Crossing': int(crossing),
                'Junction': int(junction),
                'Roundabout': int(roundabout),
                'Station': int(station)
            }])

            # Predict
            prediction = road_model.predict(input_data)
            st.success(f"🚦 Predicted Accident Severity: {prediction[0]}")
