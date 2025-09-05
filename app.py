import streamlit as st
import pickle
import numpy as np
import joblib
def load_model():
    return joblib.load('model.joblib')
try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}. Ensure you have trained and saved a 'model.joblib' file.")
st.title("Road Accident Severity Prediction")



option = st.selectbox(
    "Enter the type of features you want to predict with.",
    ("Weather"),
    index=None,
    placeholder="Select feature method...",
)

st.write("You selected:", option)
Weather_Condition = st.selectbox(
    "Enter the type of weather you want to predict with.",
    'Light Rain', 'Snow', 'Light Snow', 'Mostly Cloudy', 'Cloudy',
       'Partly Cloudy', 'Overcast', 'Scattered Clouds',
       'Light Freezing Drizzle', 'Light Drizzle', 'Rain', 'Fair', 'Fog',
       'Haze', 'Light Freezing Rain', 'Clear', 'Heavy Snow', 'Drizzle',
       'Heavy Rain', 'Light Ice Pellets', 'Thunder',
       'Thunder in the Vicinity', 'Fair / Windy',
       'Light Rain with Thunder', 'Heavy Thunderstorms and Snow',
       'Blowing Snow', 'Cloudy / Windy', 'Ice Pellets',
       'N/A Precipitation', 'Light Thunderstorms and Rain',
       'Thunderstorms and Rain', 'Light Thunderstorms and Snow', 'Mist',
       'T-Storm', 'Rain / Windy', 'Wintry Mix',
       'Heavy Thunderstorms and Rain', 'Partly Cloudy / Windy',
       'Heavy T-Storm', 'Shallow Fog', 'Light Rain / Windy',
       'Blowing Dust / Windy', 'Blowing Dust', 'Freezing Rain / Windy',
       'Light Freezing Fog', 'Mostly Cloudy / Windy', 'Smoke',
       'Light Snow / Windy', 'Heavy Ice Pellets', 'Thunderstorm',
       'Heavy Snow / Windy', 'Heavy Rain / Windy', 'Small Hail',
       'Heavy Drizzle', 'Heavy T-Storm / Windy', 'Fog / Windy',
       'Showers in the Vicinity', 'Thunder / Wintry Mix',
       'Light Snow and Sleet', 'Thunder / Wintry Mix / Windy',
       'Snow and Sleet', 'Haze / Windy', 'Freezing Rain',
       'T-Storm / Windy', 'Wintry Mix / Windy', 'Snow / Windy',
       'Light Drizzle / Windy', 'Drizzle and Fog', 'Light Rain Shower',
       'Light Freezing Rain / Windy', 'Snow and Sleet / Windy',
       'Drizzle / Windy', 'Hail', 'Light Snow with Thunder',
       'Widespread Dust', 'Light Snow Shower', 'Patches of Fog',
       'Heavy Snow with Thunder', 'Blowing Snow / Windy',
       'Thunder / Windy', 'Sleet and Thunder', 'Squalls', 'Light Sleet',
       'Smoke / Windy', 'Sleet', 'Heavy Freezing Drizzle',
       'Widespread Dust / Windy', 'Heavy Sleet and Thunder',
       'Drifting Snow / Windy', 'Freezing Drizzle', 'Snow and Thunder',
       'Light Sleet / Windy', 'Sand / Dust Whirls Nearby',
       'Thunder and Hail', 'Sleet / Windy', 'Funnel Cloud', 'Heavy Sleet',
       'Light Snow and Sleet / Windy', 'Shallow Fog / Windy',
       'Squalls / Windy', 'Light Rain Shower / Windy', 'Tornado',
       'Thunder and Hail / Windy', 'Light Snow Shower / Windy',
       'Sand / Dust Whirlwinds', 'Heavy Sleet / Windy', 'Sand / Windy',
       'Heavy Rain Shower / Windy', 'Snow and Thunder / Windy',
       'Rain Shower', 'Sand / Dust Whirlwinds / Windy',
       'Blowing Snow Nearby', 'Snow Grains', 'Heavy Freezing Rain',
       'Blowing Sand', 'Partial Fog', 'Patches of Fog / Windy',
       'Heavy Rain Shower', 'Drifting Snow', 'Light Blowing Snow',
       'Light Rain Showers', 'Heavy Thunderstorms with Small Hail'),
    index=None,
    placeholder="Select weather condition...",
)
Temperature = st.slider("Temperature (F)", min_value=0.0, max_value=8.0, step=0.1)
Wind_Chill = st.slider("Wind_Chill (F)", min_value=0.0, max_value=8.0, step=0.1)
Humidity = st.slider("Humidity (F)", min_value=0.0, max_value=8.0, step=0.1)
Pressure = st.slider("Pressure (in)", min_value=0.0, max_value=8.0, step=0.1)
Visibility = st.slider("Visibility (mi)", min_value=0.0, max_value=8.0, step=0.1)
Wind_Speed = st.slider("Wind_Speed (mph)", min_value=0.0, max_value=8.0, step=0.1)
Precipitation = st.slider("Precipitation (in)", min_value=0.0, max_value=8.0, step=0.1)


if st.button("Predict"):
    features = np.array([['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
                   'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
                   'Weather_Condition']])
    prediction = model.predict(features)
    st.write(f"Predicted Severity: {prediction[0]}")

