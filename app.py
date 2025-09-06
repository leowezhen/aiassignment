import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ================================
# Load Models
# ================================
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

# ================================
# Columns for Injury Model
# ================================
injury_columns = [
    'Distance(mi)', 'Street', 'City', 'County', 'State', 'Zipcode', 'Timezone', 'Airport_Code',
    'Weather_Timestamp', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
    'Wind_Direction', 'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition',
    'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
    'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop',
    'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight'
]

# Weather categories from training columns
unique_weather_conditions = [col.replace('Weather_Condition_', '') for col in model_columns if col.startswith('Weather_Condition_')]
unique_weather_conditions.insert(0, 'Fair')  # Add base case if dropped
numerical_features = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
                      'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']

# ================================
# Streamlit App
# ================================
st.title('Accident Severity Prediction (Multi-Model)')

feature_type = st.selectbox(
    "Select the type of prediction:",
    ["Weather", "Road Type", "Injury Severity (Full Features)"]
)

# ================================
# Weather Prediction
# ================================
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
            st.success(f"Predicted Accident Severity: {prediction[0]}")

# ================================
# Injury Severity Prediction
# ================================
elif feature_type == "Injury Severity (Full Features)":
    st.subheader("🏥 Injury Severity Prediction")

    # Numerical sliders (dataset min/max)
    distance = st.slider("Distance (mi)", min_value=0.0, max_value=152.543, step=0.001)
    temperature = st.slider("Temperature (F)", min_value=-35.0, max_value=162.0, step=0.1)
    wind_chill = st.slider("Wind_Chill (F)", min_value=-63.0, max_value=162.0, step=0.1)
    humidity = st.slider("Humidity (%)", min_value=1.0, max_value=100.0, step=0.1)
    pressure = st.slider("Pressure (in)", min_value=19.52, max_value=39.45, step=0.01)
    visibility = st.slider("Visibility (mi)", min_value=0.0, max_value=100.0, step=0.1)
    wind_speed = st.slider("Wind_Speed (mph)", min_value=0.0, max_value=232.0, step=0.1)
    precipitation = st.slider("Precipitation (in)", min_value=0.0, max_value=9.99, step=0.01)

    # Weather condition
    weather_condition = st.selectbox("Weather Condition", unique_weather_conditions)

    # Binary checkboxes
    amenity = st.checkbox("Amenity")
    bump = st.checkbox("Bump")
    crossing = st.checkbox("Crossing")
    give_way = st.checkbox("Give Way")
    junction = st.checkbox("Junction")
    no_exit = st.checkbox("No Exit")
    railway = st.checkbox("Railway")
    roundabout = st.checkbox("Roundabout")
    station = st.checkbox("Station")
    stop = st.checkbox("Stop")
    traffic_calming = st.checkbox("Traffic Calming")
    traffic_signal = st.checkbox("Traffic Signal")
    turning_loop = st.checkbox("Turning Loop")

    # Twilight
    sunrise_sunset = st.selectbox("Sunrise_Sunset", ["Day", "Night"])
    civil_twilight = st.selectbox("Civil_Twilight", ["Day", "Night"])
    nautical_twilight = st.selectbox("Nautical_Twilight", ["Day", "Night"])
    astronomical_twilight = st.selectbox("Astronomical_Twilight", ["Day", "Night"])

    if st.button("Predict Injury Severity"):
        # Build the input row
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
            'Give_Way': int(give_way),
            'Junction': int(junction),
            'No_Exit': int(no_exit),
            'Railway': int(railway),
            'Roundabout': int(roundabout),
            'Station': int(station),
            'Stop': int(stop),
            'Traffic_Calming': int(traffic_calming),
            'Traffic_Signal': int(traffic_signal),
            'Turning_Loop': int(turning_loop),
            'Sunrise_Sunset': sunrise_sunset,
            'Civil_Twilight': civil_twilight,
            'Nautical_Twilight': nautical_twilight,
            'Astronomical_Twilight': astronomical_twilight
        }])

        # Encode categoricals
        input_data_encoded = pd.get_dummies(input_data)

        # Ensure model columns align
        for col in injury_columns:
            if col not in input_data_encoded.columns:
                input_data_encoded[col] = 0
        input_data_encoded = input_data_encoded[injury_columns]

        # Predict
        prediction = injury_model.predict(input_data_encoded)
        st.success(f"🏥 Predicted Injury Severity: {prediction[0]}")

