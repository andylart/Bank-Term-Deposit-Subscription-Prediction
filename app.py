import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('model.pkl')

st.title("📊 Term Deposit Subscription Predictor")

# User inputs for the 15 selected features
def user_input_features():
    euribor3m = st.number_input("Euribor 3m rate", value=4.857)
    age = st.number_input("Age", value=35)
    campaign = st.number_input("Number of contacts during campaign", value=1)
    emp_var_rate = st.number_input("Employment variation rate", value=-1.8)
    nr_employed = st.number_input("Number of employees", value=5099.1)
    cons_conf_idx = st.number_input("Consumer confidence index", value=-36.4)
    cons_price_idx = st.number_input("Consumer price index", value=92.201)
    pdays = st.number_input("Days since last contact (999 means never)", value=999)
    education_university_degree = st.selectbox("Has university degree?", ['0', '1'])
    marital_single = st.selectbox("Is single?", ['0', '1'])
    default_unknown = st.selectbox("Default status unknown?", ['0', '1'])
    contact_telephone = st.selectbox("Contacted via telephone?", ['0', '1'])
    previous = st.number_input("Number of contacts before this campaign", value=0)
    poutcome_success = st.selectbox("Previous outcome was success?", ['0', '1'])
    job_blue_collar = st.selectbox("Job is blue-collar?", ['0', '1'])

    data = {
        'euribor3m': euribor3m,
        'age': age,
        'campaign': campaign,
        'emp_var_rate': emp_var_rate,
        'nr_employed': nr_employed,
        'cons_conf_idx': cons_conf_idx,
        'cons_price_idx': cons_price_idx,
        'pdays': pdays,
        'education_university.degree': int(education_university_degree),
        'marital_single': int(marital_single),
        'default_unknown': int(default_unknown),
        'contact_telephone': int(contact_telephone),
        'previous': previous,
        'poutcome_success': int(poutcome_success),
        'job_blue-collar': int(job_blue_collar)
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

if st.button("Predict Subscription"):
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.success(f"✅ Client is likely to subscribe to the term deposit ({proba[0][1]:.2%} probability).")
    else:
        st.error(f"❌ Client is unlikely to subscribe ({proba[0][0]:.2%} probability).")
