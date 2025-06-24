import streamlit as st
import joblib
import numpy as np

# App title and header
st.set_page_config(page_title="📊 Bank Term Deposit Subscription Prediction", layout="centered")

# Load your trained model
model = joblib.load('xgb_model_final.pkl')  # or 'xgb_model_final.pkl' if you prefer and have XGBoost installed

# User inputs (adapt this part to match your existing input fields)
euribor3m = st.number_input("Euribor 3m rate", value=4.5)
age = st.number_input("Age", value=30)
campaign = st.number_input("Number of contacts during campaign", value=1)
emp_var_rate = st.number_input("Employment variation rate", value=1.10)
nr_employed = st.number_input("Number of employees", value=5191.00)
cons_conf_idx = st.number_input("Consumer confidence index", value=-30.00)
cons_price_idx = st.number_input("Consumer price index", value=93.50)
pdays = st.number_input("Days since last contact (999 = never)", value=999)
contact_telephone = st.selectbox("Was contact via telephone?", [0, 1])
default_unknown = st.selectbox("Default status unknown?", [0, 1])
job_blue_collar = st.selectbox("Job: Blue-collar?", [0, 1])
poutcome_success = st.selectbox("Previous campaign outcome: Success?", [0, 1])
previous = st.number_input("Number of contacts before this campaign", value=0)
marital_single = st.selectbox("Marital status: Single?", [0, 1])
education_uni = st.selectbox("Education: University degree?", [0, 1])

# Collect inputs into array (make sure this matches your training feature order)
input_data = np.array([[nr_employed, pdays, euribor3m, emp_var_rate,
                        contact_telephone, cons_price_idx, default_unknown,
                        job_blue_collar, campaign, poutcome_success, previous,
                        cons_conf_idx, marital_single, education_uni, age]])

# Prediction button
if st.button("Predict"):
    proba = model.predict_proba(input_data)[:, 1]

    # Adjustable threshold
    threshold = st.slider("Adjust decision threshold", 0.0, 1.0, 0.35)

    prediction = 1 if proba >= threshold else 0

    st.subheader("📢 Prediction Result")
    if prediction == 1:
        st.success(f"✅ Subscribed\n\nProbability of Subscription (Yes): {proba[0]*100:.2f}%")
    else:
        st.error(f"❌ Did Not Subscribe\n\nProbability of Subscription (Yes): {proba[0]*100:.2f}%")
