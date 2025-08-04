
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import joblib

# Load saved pipeline
model = joblib.load("smart_premium_model.pkl")

st.set_page_config(page_title="SmartPremium Predictor", layout="centered")
st.title("💰 SmartPremium: Insurance Premium Predictor")

st.markdown("Provide all customer and policy details below, including feedback.")

# 📥 Input fields
age = st.slider("Age", 18, 100, 30)
income = st.number_input("Annual Income (₹)", min_value=10000, max_value=10000000, value=500000, step=1000)
dependents = st.slider("Number of Dependents", 0, 10, 1)
health_score = st.slider("Health Score", 0, 100, 70)
previous_claims = st.slider("Previous Claims", 0, 10, 0)
vehicle_age = st.slider("Vehicle Age (years)", 0, 30, 5)
credit_score = st.slider("Credit Score", 300, 900, 650)
insurance_duration = st.slider("Insurance Duration (years)", 1, 15, 3)
policy_start_date = st.date_input("Policy Start Date", min_value=date(2000, 1, 1), max_value=date.today())
policy_age = (date.today() - policy_start_date).days // 365

# 🧑‍💼 Categorical fields
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"])
location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
smoking_status = st.radio("Smoking Status", ["Yes", "No"])
exercise_freq = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo"])
customer_feedback = st.selectbox("Customer Feedback", ["Poor","Average","Good"])

# 🧾 Assemble input into a DataFrame
input_data = {
    "Age": age,
    "Annual Income": income,
    "Number of Dependents": dependents,
    "Health Score": health_score,
    "Previous Claims": previous_claims,
    "Vehicle Age": vehicle_age,
    "Credit Score": credit_score,
    "Insurance Duration": insurance_duration,
    "Policy Age": policy_age,
    "Gender": gender,
    "Marital Status": marital_status,
    "Education Level": education,
    "Occupation": occupation,
    "Location": location,
    "Policy Type": policy_type,
    "Smoking Status": smoking_status,
    "Exercise Frequency": exercise_freq,
    "Property Type": property_type,
    "Customer Feedback": customer_feedback
}

input_df = pd.DataFrame([input_data])

# 🔮 Predict
if st.button("📊 Predict Insurance Premium"):
    try:
        prediction = model.predict(input_df)
        st.success(f"💡 Predicted Premium: ₹{np.round(prediction[0], 2):,.2f}")
    except Exception as e:
        st.error("🚫 Error in prediction.")
        st.text(str(e))

