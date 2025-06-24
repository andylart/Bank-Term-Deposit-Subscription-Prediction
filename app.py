import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model_final.pkl')

# Streamlit App Title
st.title("📊 Bank Term Deposit Subscription Prediction")

# Function to collect user input
def user_input_features():
    euribor3m = st.number_input("Euribor 3m rate", value=4.50)
    age = st.number_input("Age", value=30)
    campaign = st.number_input("Number of contacts during campaign", value=1)
    emp_var_rate = st.number_input("Employment variation rate", value=1.10)
    nr_employed = st.number_input("Number of employees", value=5191.00)
    cons_conf_idx = st.number_input("Consumer confidence index", value=-30.00)
    cons_price_idx = st.number_input("Consumer price index", value=93.50)
    pdays = st.number_input("Days since last contact (999 means never)", value=999)
    contact_telephone = st.selectbox("Was contact via telephone?", [1, 0])
    default_unknown = st.selectbox("Default status unknown?", [1, 0])
    job_blue_collar = st.selectbox("Job: Blue-collar?", [1, 0])
    poutcome_success = st.selectbox("Previous campaign outcome: Success?", [1, 0])
    previous = st.number_input("Number of contacts before this campaign", value=0)
    marital_single = st.selectbox("Marital status: Single?", [1, 0])
    education_uni_degree = st.selectbox("Education: University degree?", [1, 0])

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
        'education_university.degree': education_uni_degree,
        'age': age
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Collect input
input_df = user_input_features()

# Make prediction when button is clicked
if st.button('Predict'):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📢 Prediction Result")
    if prediction == 1:
        st.success(f"✅ Will Subscribe\n\nProbability of Subscription (Yes): {probability*100:.2f}%")
    else:
        st.error(f"❌ Did Not Subscribe\n\nProbability of Subscription (Yes): {probability*100:.2f}%")

