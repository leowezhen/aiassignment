import streamlit as st
import pickle
import numpy as np

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

