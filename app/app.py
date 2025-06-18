# app/app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter transaction details to predict if it's fraud or not.")

# Input fields
with st.form("prediction_form"):
    V_inputs = [st.number_input(f"V{i}", value=0.0, format="%.4f") for i in range(1, 29)]
    Amount = st.number_input("Amount", value=0.0, format="%.2f")
    Time = st.number_input("Time (seconds since first transaction)", value=0.0, format="%.0f")
    
    submit = st.form_submit_button("Predict")

if submit:
    input_data = pd.DataFrame([[Time] + V_inputs + [Amount]], columns=[f"Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"🚨 Prediction: Fraudulent Transaction (Confidence: {prob:.2%})")
    else:
        st.success(f"✅ Prediction: Legitimate Transaction (Confidence: {1 - prob:.2%})")
