import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model.pkl')

st.title("📊 Term Deposit Subscription Predictor")

# User input function
def user_input_features():
    euribor3m = st.number_input("Euribor 3m rate", value=4.857)
    age = st.number_input("Age", value=35)
    campaign = st.number_input("Number of contacts during campaign", value=1)
    emp_var_rate = st.number_input("Employment variation rate", value=-1.8)
    nr_employed = st.number_input("Number of employees", value=5099.1)
    cons_conf_idx = st.number_input("Consumer confidence index", value=-36.4)
    cons_price_idx = st.number_input("Consumer price index", value=92.201)
    pdays = st.number_input("Days since last contact (999 means never)", value=999)

    data = {
        'euribor3m': euribor3m,
        'age': age,
        'campaign': campaign,
        'emp_var_rate': emp_var_rate,
        'nr_employed': nr_employed,
        'cons_conf_idx': cons_conf_idx,
        'cons_price_idx': cons_price_idx,
        'pdays': pdays
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Collect user input
input_df = user_input_features()

# Predict when button is clicked
if st.button("Predict Subscription"):
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.success(f"✅ Client is likely to subscribe ({proba[0][1]:.2%} probability).")
    else:
        st.error(f"❌ Client is unlikely to subscribe ({proba[0][0]:.2%} probability).")
