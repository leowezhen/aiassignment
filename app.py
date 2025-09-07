import streamlit as st
import pandas as pd
import joblib
import numpy as np

from sklearn.tree import DecisionTreeClassifier

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

    # Full feature severity model
    severity_model_clf = joblib.load("severity_model.joblib")
    severity_columns = joblib.load("severity_columns.joblib")  # store trained feature columns

except FileNotFoundError:
    st.error("❌ One or more joblib files are missing. Ensure they are in the same directory.")
    st.stop()

# Extract unique weather conditions
unique_weather_conditions = [col.replace('Weather_Condition_', '') for col in model_columns if col.startswith('Weather_Condition_')]
unique_weather_conditions.insert(0, 'Fair')  # add dropped base case

# Define numerical features for Weather
numerical_features = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
                      'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']

# ================================
# Streamlit App
# ================================
st.title('Accident Severity Prediction')

feature_type = st.selectbox(
    "Select the type of features you want to predict with:",
    ["Weather", "Road Type", "Full Features"]
)

# ================================
# Weather Prediction
# ================================
if feature_type == "Weather":
    st.subheader("🌦️ Weather-based Prediction")

    temperature = st.slider("Temperature (F)", -45.0, 196.0, 60.0)
    wind_chill = st.slider("Wind_Chill (F)", -63.0, 196.0, 60.0)
    humidity = st.slider("Humidity (%)", 1.0, 100.0, 50.0)
    pressure = st.slider("Pressure (in)", 0.0, 58.63, 29.0)
    visibility = st.slider("Visibility (mi)", 0.0, 100.0, 10.0)
    wind_speed = st.slider("Wind_Speed (mph)", 0.0, 1087.0, 5.0)
    precipitation = st.slider("Precipitation (in)", 0.0, 24.0, 0.0)
    weather_condition = st.selectbox('Weather Condition', unique_weather_conditions)

    if st.button('Predict Weather Severity'):
        input_data = pd.DataFrame([[temperature, wind_chill, humidity, pressure, visibility,
                                    wind_speed, precipitation, weather_condition]],
                                   columns=['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
                                            'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
                                            'Precipitation(in)', 'Weather_Condition'])
        input_data_encoded = pd.get_dummies(input_data, columns=['Weather_Condition'], drop_first=True)
        for col in model_columns:
            if col not in input_data_encoded.columns:
                input_data_encoded[col] = 0
        input_data_encoded = input_data_encoded[model_columns]
        input_data_encoded[numerical_features] = scaler.transform(input_data_encoded[numerical_features])

        prediction = dt_model.predict(input_data_encoded)
        st.success(f"🌦️ Predicted Accident Severity: {prediction[0]}")

# ================================
# Road Type Prediction
# ================================
elif feature_type == "Road Type":
    st.subheader("🚦 Road Type-based Prediction")

    crossing = st.checkbox("Crossing")
    junction = st.checkbox("Junction")
    roundabout = st.checkbox("Roundabout")
    station = st.checkbox("Station")

    if st.button("Predict Road Type Severity"):
        if not (crossing or junction or roundabout or station):
            st.warning("⚠️ Please select at least one road type before predicting.")
        else:
            input_data = pd.DataFrame([{
                'Crossing': int(crossing),
                'Junction': int(junction),
                'Roundabout': int(roundabout),
                'Station': int(station)
            }])
            prediction = road_model.predict(input_data)
            st.success(f"🚦 Predicted Accident Severity: {prediction[0]}")

# ================================
# Full Features Prediction
# ================================
elif feature_type == "Full Features":
    st.subheader("🛣️ Full Features Prediction")

    # Sliders for key numeric inputs
    distance = st.slider("Distance (mi)", 0.0, 152.543, 1.0)
    temperature = st.slider("Temperature (F)", -35.0, 162.0, 60.0)
    wind_chill = st.slider("Wind_Chill (F)", -63.0, 162.0, 60.0)
    humidity = st.slider("Humidity (%)", 1.0, 100.0, 50.0)
    pressure = st.slider("Pressure (in)", 19.52, 39.45, 29.0)
    visibility = st.slider("Visibility (mi)", 0.0, 100.0, 10.0)
    wind_speed = st.slider("Wind_Speed (mph)", 0.0, 232.0, 5.0)
    precipitation = st.slider("Precipitation (in)", 0.0, 9.99, 0.0)

    weather_condition = st.selectbox("Weather Condition", unique_weather_conditions)

    # Example boolean features
    amenity = st.checkbox("Amenity")
    bump = st.checkbox("Bump")
    crossing = st.checkbox("Crossing")
    junction = st.checkbox("Junction")

    if st.button("Predict Severity with Full Features"):
        input_data = pd.DataFrame([{
            'Distance(mi)': distance,
            'Temperature(F)': temperature,
            'Wind_Chill(F)': wind_chill,
            'Humidity(%)': humidity,
            'Pressure(in)': pressure,
            'Visibility(mi)': visibility,
            'Wind_Speed(mph)': wind_speed,
            'Precipitation(in)': precipitation,
            'Weather_Condition': weather_condition,
            'Amenity': int(amenity),
            'Bump': int(bump),
            'Crossing': int(crossing),
            'Junction': int(junction),
        }])

        # One-hot encode categoricals
        input_data_encoded = pd.get_dummies(input_data)
        for col in severity_columns:
            if col not in input_data_encoded.columns:
                input_data_encoded[col] = 0
        input_data_encoded = input_data_encoded[severity_columns]

        prediction = severity_model_clf.predict(input_data_encoded)
        st.success(f"🛣️ Predicted Severity (Full Features): {prediction[0]}")
