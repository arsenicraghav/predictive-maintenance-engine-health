import streamlit as st
import pandas as pd
from model_loader import load_model

st.set_page_config(page_title="Predictive Maintenance App")

st.title("Predictive Maintenance – Engine Fault Prediction")

st.markdown(
    "This app uses a Gradient Boosting model to predict whether an engine is **Normal (0)** "
    "or **Faulty (1)** based on sensor readings."
)

# Load model once
model = load_model()

# Input fields
st.sidebar.header("Input Engine Sensor Values")

engine_rpm = st.sidebar.number_input("Engine RPM", min_value=1, max_value=5000, value=800)
lub_oil_pressure = st.sidebar.number_input("Lub Oil Pressure (bar)", min_value=0.0, max_value=10.0, value=3.0)
fuel_pressure = st.sidebar.number_input("Fuel Pressure (bar)", min_value=0.0, max_value=25.0, value=6.0)
coolant_pressure = st.sidebar.number_input("Coolant Pressure (bar)", min_value=0.0, max_value=10.0, value=2.0)
lub_oil_temp = st.sidebar.number_input("Lub Oil Temperature (°C)", min_value=0.0, max_value=150.0, value=78.0)
coolant_temp = st.sidebar.number_input("Coolant Temperature (°C)", min_value=0.0, max_value=200.0, value=78.0)

# Convert input to dataframe (as required by model)
input_df = pd.DataFrame([{
    "Engine_RPM": engine_rpm,
    "Lub_Oil_Pressure": lub_oil_pressure,
    "Fuel_Pressure": fuel_pressure,
    "Coolant_Pressure": coolant_pressure,
    "Lub_Oil_Temperature": lub_oil_temp,
    "Coolant_Temperature": coolant_temp
}])

st.subheader("Input Data")
st.write(input_df)

if st.button("Predict Engine Condition"):
    pred = model.predict(input_df)[0]
    label = "Faulty Engine (Maintenance Required)" if pred == 1 else "Normal Engine"
    st.subheader("Prediction")
    st.success(f"Engine Condition: {label}")
