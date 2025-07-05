import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle
import os

# Load dataset
df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

# Clean data
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Map transaction type to numeric (use same in app.py)
type_mapping = {'TRANSFER': 1, 'CASH_OUT': 2, 'DEBIT': 3, 'PAYMENT': 4, 'CASH_IN': 5}
df['type'] = df['type'].map(type_mapping)

# Features and target
X = df[['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']]
y = df['isFraud']

# Balance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print(confusion_matrix(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test)))

# Save model
os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved to models/model.pkl")
