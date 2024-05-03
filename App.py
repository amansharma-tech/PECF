import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Function to load the saved model
def load_saved_model(model_path):
    model = load_model(model_path, custom_objects={'MeanSquaredError': MeanSquaredError})
    return model


# Function to preprocess data for prediction
def preprocess_data(data):
    # Preprocessing steps (normalize the data as done before)
    processed_data = (data - train_mean) / train_std
    return processed_data

# Function to make predictions
def make_prediction(model, input_data):
    predictions = model.predict(input_data)
    return predictions

# Load the saved model
model_path = 'linear_model.h5'
model = load_saved_model(model_path)

# Streamlit app
st.title('Power Consumption Prediction')

# Sidebar for user input
st.sidebar.title('Input Parameters')

# Input fields for user to input data
input_data = []
for i in range(6):
    input_data.append(st.sidebar.number_input(f'Input Data {i+1}', value=0.0))

# Preprocess the input data
processed_input = preprocess_data(np.array([input_data]))

# Make prediction
if st.button('Predict'):
    prediction = make_prediction(model, processed_input)
    st.write(f'Predicted Power Consumption: {prediction[0][0]}')
