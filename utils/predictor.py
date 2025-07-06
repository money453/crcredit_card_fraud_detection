import joblib
import pandas as pd
import os
from datetime import datetime

# Create logs directory if not exists
os.makedirs("logs", exist_ok=True)

# Load the compressed trained model
MODEL_PATH = os.path.join("assets", "model_compressed.pkl")
model = joblib.load(MODEL_PATH)

# Log file path
LOG_FILE = os.path.join("logs", "predictions_log.csv")

def log_prediction(input_dict, prediction, probability):
    # Add timestamp, prediction, and probability to input
    log_data = input_dict.copy()
    log_data["prediction"] = prediction
    log_data["confidence"] = round(probability, 4)
    log_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert to DataFrame and append
    df = pd.DataFrame([log_data])

    # If file doesn't exist, write headers
    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode='a', index=False, header=False)

# Function to make prediction
def predict_transaction(input_dict):
    input_df = pd.DataFrame([input_dict])
    expected_columns = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                        'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
    input_df = input_df[expected_columns]

    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    # Log the result
    log_prediction(input_dict, prediction, prediction_proba)

    return prediction, prediction_proba
