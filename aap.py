import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Customer Churn Analytics", layout="wide")

st.title("📊 Customer Churn Analytics App")

st.write("Upload your customer churn dataset (CSV format)")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ------------------------------
    # Basic Preprocessing
    # ------------------------------

    df = df.dropna()

    # Encode categorical columns
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = le.fit_transform(df[col])

    # Target column selection
    target_column = st.selectbox("Select Target Column (Churn Column)", df.columns)

    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        st.subheader("📈 Model Performance")
        st.write(f"Accuracy: {acc:.2f}")

        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # ------------------------------
        # Prediction Section
        # ------------------------------
        st.subheader("🔮 Predict New Customer Churn")

        input_data = []
        for col in df.drop(columns=[target_column]).columns:
            value = st.number_input(f"Enter value for {col}")
            input_data.append(value)

        if st.button("Predict"):
            input_array = np.array(input_data).reshape(1, -1)
            input_array = scaler.transform(input_array)
            prediction = model.predict(input_array)

            if prediction[0] == 1:
                st.error("⚠️ Customer is likely to Churn")
            else:
                st.success("✅ Customer is Not Likely to Churn")
