import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and scaler
try:
    dt_model = joblib.load('weather.joblib')
    scaler = joblib.load('scaler.joblib')
    # Load the list of columns the model was trained on
    model_columns = joblib.load("weather_features_encoded_cols.joblib")
except FileNotFoundError:
    st.error("Model or scaler files not found. Please ensure 'weather.joblib', 'scaler.joblib', and 'weather_features_encoded_cols.joblib' are in the same directory.")
    st.stop()

# Define the unique weather conditions your model was trained on
# This should be extracted from your data preprocessing step
# For example:
# unique_weather_conditions = data['Weather_Condition'].unique().tolist() # Assuming 'data' was your original DataFrame

# For now, we'll use the columns from the loaded model and reverse engineer the weather conditions
# A better approach would be to save the unique weather conditions during preprocessing
unique_weather_conditions = [col.replace('Weather_Condition_', '') for col in model_columns if col.startswith('Weather_Condition_')]
unique_weather_conditions.insert(0, 'Fair') # Add the base case if it was dropped by drop_first=True

st.title('Accident Severity Prediction based on Weather')

st.write("Enter the current weather conditions to predict the accident severity.")

# Get user input for numerical features
temperature = st.number_input('Temperature (F)', value=50.0)
wind_chill = st.number_input('Wind Chill (F)', value=50.0)
humidity = st.number_input('Humidity (%)', value=60.0)
pressure = st.number_input('Pressure (in)', value=29.92)
visibility = st.number_input('Visibility (mi)', value=10.0)
wind_speed = st.number_input('Wind Speed (mph)', value=10.0)
precipitation = st.number_input('Precipitation (in)', value=0.0)

# Get user input for categorical feature (Weather Condition)
weather_condition = st.selectbox('Weather Condition', unique_weather_conditions)

if st.button('Predict Severity'):
    # Create a DataFrame from user input
    input_data = pd.DataFrame([[temperature, wind_chill, humidity, pressure, visibility, wind_speed, precipitation, weather_condition]],
                               columns=['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
                                        'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
                                        'Weather_Condition'])

    # Preprocess the input data
    # Apply one-hot encoding to 'Weather_Condition'
    input_data_encoded = pd.get_dummies(input_data, columns=['Weather_Condition'], drop_first=True)

    # Ensure all columns from the training data are present in the input data, filling missing ones with 0
    for col in model_columns:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0

    # Reorder columns to match the training data
    input_data_encoded = input_data_encoded[model_columns]


    # Scale numerical features
    numerical_features = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
                          'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']

    input_data_encoded[numerical_features] = scaler.transform(input_data_encoded[numerical_features])


    # Make prediction
    prediction = dt_model.predict(input_data_encoded)

    # Display the prediction
    st.write(f"Predicted Accident Severity: {prediction[0]}")
