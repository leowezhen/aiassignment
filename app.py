import streamlit as st
import pandas as pd
import joblib
import numpy as np

try:
    # Weather model
    dt_model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    model_columns = joblib.load("weather_features_encoded_cols.joblib")

    # Road Type model
    road_model = joblib.load("roadType.joblib")

    # Injury model
    injury_model = joblib.load("dt_injury_model.joblib")

except FileNotFoundError as e:
    st.error(f"❌ Missing model file: {e}")
    st.stop()


injury_columns = [
    'Distance(mi)', 'Street', 'City', 'County', 'State', 'Zipcode', 'Timezone', 'Airport_Code',
    'Weather_Timestamp', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
    'Wind_Direction', 'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition',
    'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
    'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop',
    'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight'
]


unique_weather_conditions = [col.replace('Weather_Condition_', '') for col in model_columns if col.startswith('Weather_Condition_')]
unique_weather_conditions.insert(0, 'Fair')  # Add base case if dropped
numerical_features = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
                      'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']


st.title('Accident Severity Prediction (Multi-Model)')

feature_type = st.selectbox(
    "Select the type of prediction:",
    ["Weather", "Road Type", "Injury Severity (Full Features)"]
)

if feature_type == "Weather":
    st.subheader("🌦️ Weather-based Prediction")

    temperature = st.slider("Temperature (F)", min_value=-45.0, max_value=196.0, step=0.1)
    wind_chill = st.slider("Wind_Chill (F)", min_value=-63.0, max_value=196.0, step=0.1)
    humidity = st.slider("Humidity (%)", min_value=1.0, max_value=100.0, step=0.1)
    pressure = st.slider("Pressure (in)", min_value=0.0, max_value=58.63, step=0.1)
    visibility = st.slider("Visibility (mi)", min_value=0.0, max_value=100.0, step=0.1)
    wind_speed = st.slider("Wind_Speed (mph)", min_value=0.0, max_value=1087.0, step=0.1)
    precipitation = st.slider("Precipitation (in)", min_value=0.0, max_value=24.0, step=0.1)
    weather_condition = st.selectbox('Weather Condition', unique_weather_conditions)

    if st.button('Predict Weather Severity'):
        input_data = pd.DataFrame([[temperature, wind_chill, humidity, pressure, visibility,
                                    wind_speed, precipitation, weather_condition]],
                                   columns=['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
                                            'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
                                            'Precipitation(in)', 'Weather_Condition'])

        # Encode categorical
        input_data_encoded = pd.get_dummies(input_data, columns=['Weather_Condition'], drop_first=True)
        for col in model_columns:
            if col not in input_data_encoded.columns:
                input_data_encoded[col] = 0
        input_data_encoded = input_data_encoded[model_columns]

        # Scale numerics
        input_data_encoded[numerical_features] = scaler.transform(input_data_encoded[numerical_features])

        prediction = dt_model.predict(input_data_encoded)
        st.success(f"Predicted Accident Severity: {prediction[0]}")


elif feature_type == "Road Type":
    st.subheader("🚦 Road Type-based Prediction")

    crossing = st.checkbox("Crossing")
    junction = st.checkbox("Junction")
    roundabout = st.checkbox("Roundabout")
    station = st.checkbox("Station")

    if st.button("Predict Road Type Severity"):
        if not (crossing or junction or roundabout or station):
            st.warning("Please select at least one road type before predicting.")
        else:
            input_data = pd.DataFrame([{
                'Crossing': int(crossing),
                'Junction': int(junction),
                'Roundabout': int(roundabout),
                'Station': int(station)
            }])

            prediction = road_model.predict(input_data)
            st.success(f"Predicted Accident Severity: {prediction[0]}")

elif feature_type == "Injury Severity (Full Features)":
    st.subheader("Injury Severity Prediction")

    # Minimal inputs for demo (you can expand to all injury_columns)
    distance = st.number_input("Distance (mi)", min_value=0.0, max_value=50.0, step=0.1)
    temperature = st.slider("Temperature (F)", min_value=-45.0, max_value=196.0, step=0.1)
    humidity = st.slider("Humidity (%)", min_value=1.0, max_value=100.0, step=0.1)
    pressure = st.slider("Pressure (in)", min_value=0.0, max_value=58.63, step=0.1)
    visibility = st.slider("Visibility (mi)", min_value=0.0, max_value=100.0, step=0.1)
    wind_speed = st.slider("Wind_Speed (mph)", min_value=0.0, max_value=1087.0, step=0.1)
    precipitation = st.slider("Precipitation (in)", min_value=0.0, max_value=24.0, step=0.1)
    weather_condition = st.selectbox("Weather Condition", unique_weather_conditions)

    crossing = st.checkbox("Crossing")
    junction = st.checkbox("Junction")
    roundabout = st.checkbox("Roundabout")
    station = st.checkbox("Station")
    stop = st.checkbox("Stop")
    traffic_signal = st.checkbox("Traffic Signal")

    if st.button("Predict Injury Severity"):
        # Build input (basic version, categorical placeholders kept simple)
        input_data = pd.DataFrame([{
            'Distance(mi)': distance,
            'Temperature(F)': temperature,
            'Humidity(%)': humidity,
            'Pressure(in)': pressure,
            'Visibility(mi)': visibility,
            'Wind_Speed(mph)': wind_speed,
            'Precipitation(in)': precipitation,
            'Weather_Condition': weather_condition,
            'Crossing': int(crossing),
            'Junction': int(junction),
            'Roundabout': int(roundabout),
            'Station': int(station),
            'Stop': int(stop),
            'Traffic_Signal': int(traffic_signal)
        }])
        input_data_encoded = pd.get_dummies(input_data)
        for col in injury_columns:
            if col not in input_data_encoded.columns:
                input_data_encoded[col] = 0
        input_data_encoded = input_data_encoded[injury_columns]

        prediction = injury_model.predict(input_data_encoded)
        st.success(f"Predicted Injury Severity: {prediction[0]}")
