# train.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import os

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Separate features and labels
X = df.drop(['Class'], axis=1)
y = df['Class']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# ✅ Reduce size for faster training (30,000 samples)
X_res = X_res[:30000]
y_res = y_res[:30000]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Define models (lightweight versions)
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
    "XGBoost": XGBClassifier(
        use_label_encoder=False, eval_metric='logloss',
        max_depth=4, n_estimators=50, verbosity=0, random_state=42
    )
}

best_model = None
best_auc = 0

print("=== Training Models ===\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)

    print(f"Model: {name}")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {auc:.4f}")
    print("-" * 50)

    if auc > best_auc:
        best_model = model
        best_auc = auc
        best_model_name = name

# Save best model and scaler
if not os.path.exists("models"):
    os.makedirs("models")

joblib.dump(best_model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print(f"\n✅ Best Model: {best_model_name} (ROC-AUC: {best_auc:.4f}) saved to models/model.pkl")
import shap
import matplotlib.pyplot as plt

# Load model
import joblib
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Sample data
X_sample = X_test[:100]  # explain only first 100 points

# Use SHAP's TreeExplainer for tree-based models
explainer = shap.Explainer(model)
shap_values = explainer(X_sample)

# Plot SHAP summary (feature importance)
shap.summary_plot(shap_values, X_sample)
