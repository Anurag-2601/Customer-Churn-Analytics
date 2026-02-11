import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Prediction System")

# -------------------------
# Load Saved Artifacts
# -------------------------

@st.cache_resource
@st.cache_resource
def load_model():
    model = joblib.load("pickle files/churn_model.pkl")
    scaler = joblib.load("pickle files/scaler.pkl")
    encoders = joblib.load("pickle files/encoders.pkl")
    feature_columns = joblib.load("pickle_files/feature_columns.pkl")
    return model, scaler, encoders, feature_columns


model, scaler, encoders, feature_columns = load_model()

st.sidebar.header("Enter Customer Details")

# -------------------------
# Dynamic Input Fields
# -------------------------

input_data = {}

for feature in encoders.keys():
    options = encoders[feature].classes_
    input_data[feature] = st.sidebar.selectbox(feature, options)

# Numerical features (modify based on your dataset)
tenure = st.sidebar.slider("tenure", 0, 72, 12)
MonthlyCharges = st.sidebar.slider("MonthlyCharges", 0.0, 200.0, 70.0)
TotalCharges = st.sidebar.slider("TotalCharges", 0.0, 10000.0, 2000.0)

input_data["tenure"] = tenure
input_data["MonthlyCharges"] = MonthlyCharges
input_data["TotalCharges"] = TotalCharges

input_df = pd.DataFrame([input_data])

# -------------------------
# Encoding
# -------------------------

for col in encoders:
    # Reorder input columns correctly
    input_df = input_df[feature_columns]


# -------------------------
# Scaling
# -------------------------

input_scaled = scaler.transform(input_df)


# -------------------------
# Prediction
# -------------------------

if st.button("Predict Churn"):

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")

    st.write(f"Churn Probability: {round(probability * 100, 2)}%")
