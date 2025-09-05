import streamlit as st
import pickle
import numpy as np
import joblib
import pandas as pd
import streamlit as st
def load_artifacts():
    model = joblib.load("model.joblib")
    return model

model = load_artifacts()


option = st.selectbox(
    "Enter the type of features you want to predict with.",
    ("Weather"),
    index=None,
    placeholder="Select feature method...",
)

st.write("You selected:", option)


Temperature = st.slider("Temperature (F)", min_value=0.0, max_value=8.0, step=0.1)
Wind_Chill = st.slider("Wind_Chill (F)", min_value=0.0, max_value=8.0, step=0.1)
Humidity = st.slider("Humidity (F)", min_value=0.0, max_value=8.0, step=0.1)
Pressure = st.slider("Pressure (in)", min_value=0.0, max_value=8.0, step=0.1)
Visibility = st.slider("Visibility (mi)", min_value=0.0, max_value=8.0, step=0.1)
Wind_Speed = st.slider("Wind_Speed (mph)", min_value=0.0, max_value=8.0, step=0.1)
Precipitation = st.slider("Precipitation (in)", min_value=0.0, max_value=8.0, step=0.1)


input_df = pd.DataFrame([{
    'Temperature(F)': Temperature,
    'Wind_Chill(F)': Wind_Chill,
    'Humidity(%)': Humidity,
    'Pressure(in)': Pressure,
    'Visibility(mi)': Visibility,
    'Wind_Speed(mph)': Wind_Speed,
    'Precipitation(in)': Precipitation
}])


if st.button("Predict"):
    features = np.array([[
        float(Temperature),
        float(Wind_Chill),
        float(Humidity),
        float(Pressure),
        float(Visibility),
        float(Wind_Speed),
        float(Precipitation),

    ]], dtype=object)
    prediction = model.predict(encoded)
    st.write(f"Predicted Severity: {prediction[0]}")

