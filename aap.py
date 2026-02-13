import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Prediction System")

# -------------------------
# Load Pipeline
# -------------------------
@st.cache_resource
def load_model():
    return joblib.load("pickle files/churn_pipeline.pkl")

model = load_model()

st.sidebar.header("Enter Customer Details")

st.write("Model expects columns:", model.feature_names_in_)

# -------------------------
# User Input (edit fields to match dataset)
# -------------------------

gender = st.sidebar.selectbox("gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("SeniorCitizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

tenure = st.sidebar.slider("tenure", 0, 72, 12)
MonthlyCharges = st.sidebar.slider("MonthlyCharges", 0.0, 200.0, 70.0)
TotalCharges = st.sidebar.slider("TotalCharges", 0.0, 10000.0, 2000.0)

input_data = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}

input_df = pd.DataFrame([input_data])

# -------------------------
# Prediction
# -------------------------

st.write("Model expects these columns:")
st.write(model.feature_names_in_)

st.write("Your input columns:")
st.write(input_df.columns.tolist())

st.stop()

if prediction == 1:
    st.error("⚠️ Customer is likely to CHURN")
else:
    st.success("✅ Customer is likely to STAY")
