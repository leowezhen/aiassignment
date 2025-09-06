import streamlit as st
import pandas as pd
import joblib

# =========================
# Load models
# =========================
try:
    dt_weather_model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    weather_columns = joblib.load("weather_features_encoded_cols.joblib")

    road_type_model = joblib.load("roadType.joblib")
    road_columns = joblib.load("road_features_encoded_cols.joblib")

    injury_model = joblib.load("dt_injury_model.joblib")
    injury_columns = joblib.load("injury_features_encoded_cols.joblib")

except FileNotFoundError as e:
    st.error(f"Missing file: {e}")
    st.stop()

# Extract unique weather conditions
unique_weather_conditions = [col.replace("Weather_Condition_", "") 
                             for col in weather_columns if col.startswith("Weather_Condition_")]
unique_weather_conditions.insert(0, "Fair")

# =========================
# Page setup
# =========================
st.title("🚦 Accident Prediction System")

menu = st.radio("Choose a prediction task:", 
                ["Weather Severity", "Road Type", "Injury Severity"])

# =========================
# WEATHER SEVERITY
# =========================
if menu == "Weather Severity":
    st.header("🌦 Predict Accident Severity Based on Weather")

    temperature = st.slider("Temperature (F)", -45.0, 196.0, 60.0, 0.1)
    wind_chill = st.slider("Wind_Chill (F)", -63.0, 196.0, 50.0, 0.1)
    humidity = st.slider("Humidity (%)", 1.0, 100.0, 50.0, 0.1)
    pressure = st.slider("Pressure (in)", 0.0, 58.63, 29.0, 0.1)
    visibility = st.slider("Visibility (mi)", 0.0, 100.0, 10.0, 0.1)
    wind_speed = st.slider("Wind_Speed (mph)", 0.0, 1087.0, 5.0, 0.1)
    precipitation = st.slider("Precipitation (in)", 0.0, 24.0, 0.0, 0.1)

    weather_condition = st.selectbox("Weather Condition", unique_weather_conditions)

    if st.button("Predict Weather Severity"):
        input_df = pd.DataFrame([[temperature, wind_chill, humidity, pressure, visibility,
                                  wind_speed, precipitation, weather_condition]],
                                columns=['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
                                         'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
                                         'Precipitation(in)', 'Weather_Condition'])

        input_encoded = pd.get_dummies(input_df, columns=["Weather_Condition"], drop_first=True)
        for col in weather_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[weather_columns]

        input_encoded[['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
                       'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']] = scaler.transform(
            input_encoded[['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
                           'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']]
        )

        pred = dt_weather_model.predict(input_encoded)
        st.success(f"🌦 Predicted Weather Severity: {pred[0]}")

# =========================
# ROAD TYPE PREDICTION
# =========================
elif menu == "Road Type":
    st.header("🛣 Predict Accident Risk by Road Type")

    crossing = st.checkbox("Crossing")
    junction = st.checkbox("Junction")
    roundabout = st.checkbox("Roundabout")
    station = st.checkbox("Station")

    if st.button("Predict Road Type"):
        input_df = pd.DataFrame([{
            "Crossing": int(crossing),
            "Junction": int(junction),
            "Roundabout": int(roundabout),
            "Station": int(station)
        }])

        # Align columns
        for col in road_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[road_columns]

        pred = road_type_model.predict(input_df)
        st.success(f"🛣 Predicted Road Type Risk: {pred[0]}")

# =========================
# INJURY SEVERITY
# =========================
elif menu == "Injury Severity":
    st.header("🏥 Predict Injury Severity")

    # Numerical sliders (with real min/max from dataset stats)
    distance = st.slider("Distance (mi)", 0.0, 152.543, 0.5, 0.001)
    temperature = st.slider("Temperature (F)", -35.0, 162.0, 60.0, 0.1)
    wind_chill = st.slider("Wind_Chill (F)", -63.0, 162.0, 50.0, 0.1)
    humidity = st.slider("Humidity (%)", 1.0, 100.0, 50.0, 0.1)
    pressure = st.slider("Pressure (in)", 19.52, 39.45, 29.0, 0.01)
    visibility = st.slider("Visibility (mi)", 0.0, 100.0, 10.0, 0.1)
    wind_speed = st.slider("Wind_Speed (mph)", 0.0, 232.0, 5.0, 0.1)
    precipitation = st.slider("Precipitation (in)", 0.0, 9.99, 0.0, 0.01)

    weather_condition = st.selectbox("Weather Condition", unique_weather_conditions)

    # Booleans
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
        input_df = pd.DataFrame([{
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

        # Encode
        input_encoded = pd.get_dummies(input_df)
        for col in injury_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[injury_columns]

        pred = injury_model.predict(input_encoded)
        st.success(f"🏥 Predicted Injury Severity: {pred[0]}")
