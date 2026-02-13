import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------- Load files ----------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# ---------------- Page config ----------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📉 Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn using machine learning.")

# ---------------- Sidebar ----------------
st.sidebar.header("🧾 Customer Details")

st.sidebar.markdown("### 👤 Personal Info")
dependents = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)

st.sidebar.markdown("### 🌐 Services")
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

st.sidebar.markdown("### 💳 Billing")
monthly_charges = st.sidebar.slider("Monthly Charges", 0.0, 200.0, 70.0)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment = st.sidebar.selectbox("Payment Method", ["Bank transfer", "Credit card", "Electronic check", "Mailed check"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

# ---------------- Input processing ----------------
input_data = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": tenure * monthly_charges,
    "Dependents": 1 if dependents == "Yes" else 0,
    "PaperlessBilling": 1 if paperless == "Yes" else 0
}

input_df = pd.DataFrame(columns=features)
input_df.loc[0] = 0

for col in input_data:
    input_df[col] = input_data[col]

input_scaled = scaler.transform(input_df)

# ---------------- Session state ----------------
if "history" not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = None

# ---------------- Prediction ----------------
if st.button("🔍 Predict Churn"):
    prob = model.predict_proba(input_scaled)[0][1]

    if prob > 0.40:
        risk = "High Risk 🔴"
        status = "Customer has HIGH churn risk"
        st.error(status)
    elif prob > 0.25:
        risk = "Medium Risk 🟠"
        status = "Customer may CHURN"
        st.warning(status)
    else:
        risk = "Low Risk 🟢"
        status = "Customer is likely to STAY"
        st.success(status)

    st.progress(int(prob * 100))
    st.write(f"**Churn Probability:** {prob*100:.2f}%")
    st.write(f"**Risk Level:** {risk}")

    st.session_state.result = {
        "prob": prob,
        "risk": risk
    }

    st.session_state.history.append({
        "Probability (%)": round(prob * 100, 2),
        "Risk": risk
    })

# ---------------- Reports ----------------
if st.session_state.result:
    prob = st.session_state.result["prob"]
    risk = st.session_state.result["risk"]

    # TXT Report
    report_text = f"""
Customer Churn Prediction Report
--------------------------------
Churn Probability : {prob*100:.2f} %
Risk Level        : {risk}

Customer Summary:
Tenure            : {tenure} months
Monthly Charges   : ₹{monthly_charges}
Contract Type     : {contract}
Payment Method    : {payment}
Paperless Billing : {paperless}
"""

    st.download_button(
        "⬇️ Download TXT Report",
        report_text,
        "churn_prediction_report.txt",
        "text/plain"
    )

    # CSV Report
    report_df = pd.DataFrame([{
        "Churn Probability (%)": round(prob * 100, 2),
        "Risk Level": risk,
        "Tenure (months)": tenure,
        "Monthly Charges": monthly_charges,
        "Contract Type": contract,
        "Payment Method": payment,
        "Paperless Billing": paperless
    }])

    st.download_button(
        "⬇️ Download CSV Report",
        report_df.to_csv(index=False),
        "churn_prediction_report.csv",
        "text/csv"
    )

    # Feature importance
    st.subheader("📊 Why this prediction?")
    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(10)

    st.bar_chart(imp_df.set_index("Feature"))

# ---------------- History ----------------
st.subheader("🕒 Recent Predictions")
if st.session_state.history:
    st.table(pd.DataFrame(st.session_state.history[-5:]))

# ---------------- Reset ----------------
if st.button("🔄 Reset App"):
    st.session_state.history = []
    st.session_state.result = None
    st.rerun()
