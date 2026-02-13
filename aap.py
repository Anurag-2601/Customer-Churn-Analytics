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

# -------------------------
# User Input (CORRECT DATASET INPUTS)
# -------------------------
st.sidebar.header("Enter Customer Details")

Account_length = st.sidebar.number_input("Account length", 0, 300, 100)
Area_code = st.sidebar.selectbox("Area code", [408, 415, 510])
International_plan = st.sidebar.selectbox("International plan", ["Yes", "No"])
Voice_mail_plan = st.sidebar.selectbox("Voice mail plan", ["Yes", "No"])
Number_vmail_messages = st.sidebar.number_input("Number vmail messages", 0, 60, 10)

Total_day_minutes = st.sidebar.number_input("Total day minutes", 0.0, 400.0, 180.0)
Total_day_calls = st.sidebar.number_input("Total day calls", 0, 200, 100)
Total_day_charge = st.sidebar.number_input("Total day charge", 0.0, 100.0, 30.0)

Total_eve_minutes = st.sidebar.number_input("Total eve minutes", 0.0, 400.0, 200.0)
Total_eve_calls = st.sidebar.number_input("Total eve calls", 0, 200, 100)
Total_eve_charge = st.sidebar.number_input("Total eve charge", 0.0, 100.0, 20.0)

Total_night_minutes = st.sidebar.number_input("Total night minutes", 0.0, 400.0, 200.0)
Total_night_calls = st.sidebar.number_input("Total night calls", 0, 200, 100)
Total_night_charge = st.sidebar.number_input("Total night charge", 0.0, 100.0, 10.0)

Total_intl_minutes = st.sidebar.number_input("Total intl minutes", 0.0, 20.0, 10.0)
Total_intl_calls = st.sidebar.number_input("Total intl calls", 0, 20, 5)
Total_intl_charge = st.sidebar.number_input("Total intl charge", 0.0, 10.0, 2.0)

Customer_service_calls = st.sidebar.number_input("Customer service calls", 0, 10, 1)

input_df = pd.DataFrame([{
    "Account length": Account_length,
    "Area code": Area_code,
    "International plan": 1 if International_plan=="Yes" else 0,
    "Voice mail plan": 1 if Voice_mail_plan=="Yes" else 0,
    "Number vmail messages": Number_vmail_messages,
    "Total day minutes": Total_day_minutes,
    "Total day calls": Total_day_calls,
    "Total day charge": Total_day_charge,
    "Total eve minutes": Total_eve_minutes,
    "Total eve calls": Total_eve_calls,
    "Total eve charge": Total_eve_charge,
    "Total night minutes": Total_night_minutes,
    "Total night calls": Total_night_calls,
    "Total night charge": Total_night_charge,
    "Total intl minutes": Total_intl_minutes,
    "Total intl calls": Total_intl_calls,
    "Total intl charge": Total_intl_charge,
    "Customer service calls": Customer_service_calls
}])

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Churn"):

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")
