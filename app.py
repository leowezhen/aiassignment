import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================
# Load models and encoders
# ==========================
try:
    dt_weather_model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    weather_columns = joblib.load("weather_features_encoded_cols.joblib")

    dt_road_model = joblib.load("roadType.joblib")
    road_columns = joblib.load("road_features_encoded_cols.joblib")

    dt_injury_model = joblib.load("dt_injury_model.joblib")
    injury_columns = joblib.load("injury_features_encoded_cols.joblib")
except FileNotFoundError:
    st.error("Some model or column files are missing. Please check your directory.")
    st.stop()

# ==========================
# Weather conditions list
# ==========================
unique_weather_conditions = [
    col.replace("Weather_Condition_", "")
    for col in weather_columns
    if col.startswith("Weather_Condition_")
]
unique_weather_conditions.insert(0, "Fair")

# ==========================
# Streamlit UI
# ==========================
st.title("🚦 Accident Severity & Injury Prediction")

feature_type = st.selectbox(
    "Choose the feature set for prediction:",
    ["Weather Severity", "Road Type", "Injury Severity (Full Features)"],
    index=0,
)

# ==========================
# WEATHER SEVERITY
# ==========================
if feature_type == "Weather Severity":
    st.subheader("🌦️ Accident Severity Prediction (Weather Features)")

    temperature = st.slider("Temperature (F)", -45.0, 196.0, step=0.1)
    wind_chill = st.slider("Wind_Chill (F)", -63.0, 196.0, step=0.1)
    humidity = st.slider("Humidity (%)", 1.0, 100.0, step=0.1)
    pressure = st.slider("Pressure (in)", 0.0, 58.63, step=0.1)
    visibility = st.slider("Visibility (mi)", 0.0, 100.0, step=0.1)
    wind_speed = st.slider("Wind_Speed (mph)", 0.0, 1087.0, step=0.1)
    precipitation = st.slider("Precipitation (in)", 0.0, 24.0, step=0.1)

    weather_condition = st.selectbox("Weather Condition", unique_weather_conditions)

    if st.button("Predict Weather Severity"):
        input_data = pd.DataFrame([{
            "Temperature(F)": temperature,
            "Wind_Chill(F)": wind_chill,
            "Humidity(%)": humidity,
            "Pressure(in)": pressure,
            "Visibility(mi)": visibility,
            "Wind_Speed(mph)": wind_speed,
            "Precipitation(in)": precipitation,
            "Weather_Condition": weather_condition,
        }])

        encoded = pd.get_dummies(input_data, columns=["Weather_Condition"], drop_first=True)
        for col in weather_columns:
            if col not in encoded.columns:
                encoded[col] = 0
        encoded = encoded[weather_columns]
        encoded[numerical_features] = scaler.transform(encoded[numerical_features])

        prediction = dt_weather_model.predict(encoded)
        st.success(f"🌦️ Predicted Severity: {prediction[0]}")

# ==========================
# ROAD TYPE
# ==========================
elif feature_type == "Road Type":
    st.subheader("🛣️ Road Type Prediction")

    crossing = st.checkbox("Crossing")
    junction = st.checkbox("Junction")
    roundabout = st.checkbox("Roundabout")
    station = st.checkbox("Station")

    if st.button("Predict Road Type"):
        input_data = pd.DataFrame([{
            "Crossing": int(crossing),
            "Junction": int(junction),
            "Roundabout": int(roundabout),
            "Station": int(station),
        }])

        encoded = pd.get_dummies(input_data)
        for col in road_columns:
            if col not in encoded.columns:
                encoded[col] = 0
        encoded = encoded[road_columns]

        prediction = dt_road_model.predict(encoded)
        st.success(f"🛣️ Predicted Road Type: {prediction[0]}")

# ==========================
# INJURY SEVERITY (FULL FEATURES)
# ==========================
elif feature_type == "Injury Severity (Full Features)":
    st.subheader("🏥 Injury Severity Prediction")

    # Numerical sliders
    distance = st.slider("Distance (mi)", 0.0, 152.543, step=0.001)
    temperature = st.slider("Temperature (F)", -35.0, 162.0, step=0.1)
    wind_chill = st.slider("Wind_Chill (F)", -63.0, 162.0, step=0.1)
    humidity = st.slider("Humidity (%)", 1.0, 100.0, step=0.1)
    pressure = st.slider("Pressure (in)", 19.52, 39.45, step=0.01)
    visibility = st.slider("Visibility (mi)", 0.0, 100.0, step=0.1)
    wind_speed = st.slider("Wind_Speed (mph)", 0.0, 232.0, step=0.1)
    precipitation = st.slider("Precipitation (in)", 0.0, 9.99, step=0.01)

    weather_condition = st.selectbox("Weather Condition", unique_weather_conditions)

    # Binary features
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

    # Twilight categories
    sunrise_sunset = st.selectbox("Sunrise_Sunset", ["Day", "Night"])
    civil_twilight = st.selectbox("Civil_Twilight", ["Day", "Night"])
    nautical_twilight = st.selectbox("Nautical_Twilight", ["Day", "Night"])
    astronomical_twilight = st.selectbox("Astronomical_Twilight", ["Day", "Night"])

    if st.button("Predict Injury Severity"):
        input_data = pd.DataFrame([{
            "Distance(mi)": distance,
            "Temperature(F)": temperature,
            "Wind_Chill(F)": wind_chill,
            "Humidity(%)": humidity,
            "Pressure(in)": pressure,
            "Visibility(mi)": visibility,
            "Wind_Speed(mph)": wind_speed,
            "Precipitation(in)": precipitation,
            "Weather_Condition": weather_condition,
            "Amenity": int(amenity),
            "Bump": int(bump),
            "Crossing": int(crossing),
            "Give_Way": int(give_way),
            "Junction": int(junction),
            "No_Exit": int(no_exit),
            "Railway": int(railway),
            "Roundabout": int(roundabout),
            "Station": int(station),
            "Stop": int(stop),
            "Traffic_Calming": int(traffic_calming),
            "Traffic_Signal": int(traffic_signal),
            "Turning_Loop": int(turning_loop),
            "Sunrise_Sunset": sunrise_sunset,
            "Civil_Twilight": civil_twilight,
            "Nautical_Twilight": nautical_twilight,
            "Astronomical_Twilight": astronomical_twilight,
        }])

        encoded = pd.get_dummies(input_data)
        for col in injury_columns:
            if col not in encoded.columns:
                encoded[col] = 0
        encoded = encoded[injury_columns]

        prediction = dt_injury_model.predict(encoded)
        st.success(f"🏥 Predicted Injury Severity: {prediction[0]}")
