import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

# Define function to collect user inputs
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
    return pd.DataFrame(data, index=[0])

# Main Streamlit app layout
st.title("Bank Term Deposit Subscription Prediction")

# Get user inputs
input_df = user_input_features()

# 📌 Debug: Show model expected and input columns for verification
st.write("✅ Model expects these features:", model.feature_names_in_.tolist())
st.write("✅ Input DataFrame columns:", input_df.columns.tolist())

# Align input features with model expectations
input_df = input_df[model.feature_names_in_]

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.write(f"🎯 Prediction: **{prediction[0]}**")
    st.write("📊 Prediction probabilities:")
    st.write(f" - No: {prediction_proba[0][0]:.2f}")
    st.write(f" - Yes: {prediction_proba[0][1]:.2f}")
