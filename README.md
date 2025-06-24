# 📊 Bank Term Deposit Subscription Prediction

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-blue?logo=streamlit)](https://version-2-gjjtsujxw8scusxnxk9ybg.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)]
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.5-orange?logo=xgboost)]
[![License](https://img.shields.io/badge/License-MIT-green)]()

---

## 🚀 Project Overview

This project predicts whether a bank client will subscribe to a term deposit using a powerful machine learning model. It leverages economic indicators, client demographics, and campaign details to make accurate predictions.

The model is deployed as a user-friendly **Streamlit** app where users input client details and instantly get subscription probabilities and predictions.

---

## ⚙️ Features Used

| Feature Category      | Examples                                      |
|----------------------|-----------------------------------------------|
| Economic Indicators  | Euribor 3-month rate, Consumer Price Index, Consumer Confidence Index |
| Client Demographics  | Age, Marital status, Job type, Education level |
| Campaign Information | Number of contacts, Previous campaign outcome |
| Contact Details      | Telephone contact, Days since last contact     |
| Financial Metrics    | Employment variation rate, Number of employees |

---

## 🧠 Model Development & Improvements

- **Imbalanced Data Handling:** Applied SMOTE to balance subscription classes and improve recall.
- **Feature Selection:** Selected most impactful features to reduce noise and avoid overfitting.
- **Model Upgrade:** Moved from Random Forest to XGBoost for better speed and accuracy.
- **Hyperparameter Tuning:** Used Optuna to optimize model parameters, boosting accuracy from ~88% to ~91%.
- **Model Compression:** Compressed model size to ~1.1 MB using `joblib` for efficient deployment.
- **User Interaction:** Added threshold adjustment slider for flexible precision-recall tradeoff.
- **UI Enhancements:** Streamlit interface with clear inputs, defaults, and color-coded prediction results.

---

## 📊 Model Performance

| Metric    | Score  |
|-----------|---------|
| Accuracy  | 91.19%  |
| Precision | ~0.91   |
| Recall    | ~0.91   |
| F1 Score  | ~0.91   |

> Confusion matrix indicates balanced performance between identifying subscribers and non-subscribers.

---

## 🎯 How to Use

1. Visit the app: [Streamlit Demo](https://version-2-gjjtsujxw8scusxnxk9ybg.streamlit.app/)
2. Fill in client and campaign features or keep default values.
3. Adjust the subscription threshold slider as needed.
4. Click **Predict** to get a subscription probability and decision.

---

## 📈 Future Enhancements

- Add explainability features (e.g., SHAP values) for transparent model decisions.
- Support batch predictions and downloadable reports.
- Expand dataset with new features and external data sources.
- Build an automated feedback loop to update the model over time.
- Explore ensemble methods or deep learning for improved accuracy.

---

## 📂 Project Structure

├── app.py # Streamlit application code
├── xgb_model_final.pkl # Trained and optimized XGBoost model
├── data_preprocessing.py # Scripts for data cleaning and feature engineering (optional)
├── model_training.ipynb # Jupyter notebook with training and tuning details (optional)
└── README.md # Project documentation

---

## 📬 Contact & Collaboration

Feel free to open issues or pull requests, or contact me for collaborations and questions!

---

*Made with ❤️ using Python, XGBoost & Streamlit.*

---

![Streamlit Logo](https://streamlit.io/images/brand/streamlit-mark-color.svg)