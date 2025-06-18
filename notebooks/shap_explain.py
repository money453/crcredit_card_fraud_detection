import joblib
import shap
import pandas as pd

# Load model and scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load dataset
data = pd.read_csv("data/creditcard.csv")

# Separate features and labels
X = data.drop(columns=["Class"])  # Keep "Time" this time

# Scale using saved scaler
X_scaled = scaler.transform(X)

# Use SHAP for explanation (only sample first 100 to save time)
explainer = shap.Explainer(model.predict, X_scaled)
shap_values = explainer(X_scaled[:100])

# Plot summary
shap.summary_plot(shap_values, X.iloc[:100], show=True)
