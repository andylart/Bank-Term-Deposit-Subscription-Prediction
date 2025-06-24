import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model.pkl")

st.title("📊 Bank Term Deposit Subscription Prediction")

# Function to collect user input for all required features
def user_input_features():
    euribor3m = st.number_input("Euribor 3m rate", value=4.857)
    age = st.number_input("Age", value=35)
    campaign = st.number_input("Number of contacts during campaign", value=1)
    emp_var_rate = st.number_input("Employment variation rate", value=-1.8)
    nr_employed = st.number_input("Number of employees", value=5099.1)
    cons_conf_idx = st.number_input("Consumer confidence index", value=-36.4)
    cons_price_idx = st.number_input("Consumer price index", value=92.201)
    pdays = st.number_input("Days since last contact (999 means never)", value=999)

    contact_telephone = st.selectbox("Was contact via telephone? (1=Yes, 0=No)", [1, 0])
    default_unknown = st.selectbox("Default status unknown? (1=Yes, 0=No)", [1, 0])
    job_blue_collar = st.selectbox("Job: Blue-collar? (1=Yes, 0=No)", [1, 0])
    poutcome_success = st.selectbox("Previous campaign outcome: Success? (1=Yes, 0=No)", [1, 0])
    previous = st.number_input("Number of contacts before this campaign", value=0)
    marital_single = st.selectbox("Marital status: Single? (1=Yes, 0=No)", [1, 0])
    education_university = st.selectbox("Education: University degree? (1=Yes, 0=No)", [1, 0])

    data = {
        'nr_employed': nr_employed,
        'pdays': pdays,
        'euribor3m': euribor3m,
        'emp_var_rate': emp_var_rate,
        'contact_telephone': contact_telephone,
        'cons_price_idx': cons_price_idx,
        'default_unknown': default_unknown,
        'job_blue-collar': job_blue_collar,
        'campaign': campaign,
        'poutcome_success': poutcome_success,
        'previous': previous,
        'cons_conf_idx': cons_conf_idx,
        'marital_single': marital_single,
        'education_university.degree': education_university,
        'age': age
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Collect input features
input_df = user_input_features()

# Make prediction when user clicks the button
if st.button("Predict Subscription Outcome"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    result = "✅ Subscribed to Term Deposit" if prediction[0] == 1 else "❌ Did Not Subscribe"
    st.subheader(f"Prediction: {result}")
    st.write(f"Probability of Subscription (Yes): {prediction_proba[0][1]:.2%}")
    st.write(f"Probability of No Subscription: {prediction_proba[0][0]:.2%}")
