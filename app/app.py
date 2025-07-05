import sys
import os

# Add the project root directory to sys.path so we can import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from utils.predictor import predict_transaction

# Streamlit page settings
st.set_page_config(page_title="SmartBank Fraud Detection", layout="centered")
st.title("💳 SmartBank Fraud Detection")

with st.form("fraud_form"):
    st.write("### 📥 Enter Transaction Details")

    # Inputs
    step = st.number_input("Step", min_value=1, value=1)

    txn_type_str = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "DEBIT", "PAYMENT", "CASH_IN"])
    txn_type_map = {
        "TRANSFER": 0,
        "CASH_OUT": 1,
        "DEBIT": 2,
        "PAYMENT": 3,
        "CASH_IN": 4
    }
    txn_type = txn_type_map[txn_type_str]

    amount = st.number_input("Amount", min_value=0.0, value=5000.0)
    oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0, value=8000.0)
    newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0, value=3000.0)
    oldbalanceDest = st.number_input("Old Balance Dest", min_value=0.0)
    newbalanceDest = st.number_input("New Balance Dest", min_value=0.0)
    isFlaggedFraud = st.selectbox("Is Flagged Fraud?", [0, 1])

    # Predict button
    submitted = st.form_submit_button("🚀 Predict Fraud")

    if submitted:
        input_data = {
            "step": step,
            "type": txn_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
            "isFlaggedFraud": isFlaggedFraud
        }

        pred, prob = predict_transaction(input_data)

        st.write(f"🔎 **Prediction:** {'FRAUD' if pred == 1 else 'LEGITIMATE'}")
        st.write(f"💰 **Transaction Amount:** ₹{amount:,.2f}")

        if pred == 1:
            st.error(f"⚠️ **FRAUD DETECTED!**\nThis ₹{amount:,.2f} transaction is likely **fraudulent** with confidence **{prob:.2f}**.")
        else:
            st.success(f"✅ **LEGITIMATE TRANSACTION**\nThis ₹{amount:,.2f} transaction is predicted **safe** with confidence **{(1 - prob):.2f}**.")
