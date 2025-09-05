import streamlit as st
import joblib

@st.cache_data(show_spinner=True)  # Caches the model loading
def load_model():
    return joblib.load('model.joblib')

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}. Ensure you have trained and saved a 'model.joblib' file.")
