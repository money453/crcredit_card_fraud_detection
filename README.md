# 💳 SmartBank Fraud Detection

A real-time Streamlit web app to detect fraudulent banking transactions using a trained machine learning model.

---

## 🚀 Project Overview

Banking fraud is a serious concern in digital transactions. This project demonstrates a machine learning pipeline trained to identify fraudulent transactions using key features of a typical banking log. 

It includes a full-stack ML system:
- Data preprocessing
- Model training (with performance optimization)
- A Streamlit UI for interactive predictions
- Deployment on Streamlit Cloud

---

## 💡 Features

- ✅ Predicts whether a transaction is **Fraudulent** or **Legitimate**
- 🔍 Takes transaction details as input
- 🧠 Uses a compressed and optimized `RandomForestClassifier`
- 📊 Logs every prediction to a CSV file
- 🌐 Deployed on the web using **Streamlit Cloud**

---

## 🧠 Model Details

- **Algorithm**: Random Forest
- **ROC AUC Score**: 0.9995
- **Custom Features Added**:
  - `errorOrig`: Difference in origin account
  - `errorDest`: Difference in destination account
  - `isZeroChange`: Flag for transactions that didn't change balances

---

## 🖥️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/money453/crcredit_card_fraud_detection.git
cd crcredit_card_fraud_detection
2. Install dependencies
Make sure you are in a virtual environment, then run:

pip install -r requirements.txt
3. Run the app
streamlit run app/app.py
4.Sample Predictions
Step	Type	Amount	OldBalanceOrig	NewBalanceOrig	OldBalanceDest	NewBalanceDest	Flagged	Prediction
1	TRANSFER	9847.91	170136.00	160288.09	0.0	0.0	0	FRAUD
1	PAYMENT	3000.00	3000.00	0.00	50000.00	53000.00	0	LEGITIMATE
5.📁 Project Structure

.
├── app/
│   └── app.py                # Streamlit UI
├── utils/
│   └── predictor.py          # Prediction logic
├── assets/
│   └── model_compressed.pkl  # Compressed ML model
├── logs/
│   └── predictions_log.csv   # Auto-generated log of predictions
├── data/
│   └── (Not pushed due to size)
├── train.py                  # Training + model compression
├── requirements.txt
└── README.md
6.Deployment (Live Demo)
Deployed on Streamlit Cloud
🔗 http://localhost:8501/
7.🧠 Author
Manikanta
3rd Year AI Student @ Mahindra University
Email: [onlinebusiness20228@gmail.com]
GitHub: github.com/money453



