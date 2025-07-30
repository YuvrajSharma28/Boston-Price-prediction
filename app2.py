# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import fetch_california_housing

st.title("Boston Housing Price Predictor")

# Load dataset (to get feature names)
housing = fetch_california_housing(as_frame=True)
feature_names = housing.feature_names

# Load trained model
model = joblib.load("model.pkl")

# Create inputs
st.sidebar.header("Enter feature values")
user_data = {}
for feat in feature_names:
    user_data[feat] = st.sidebar.number_input(feat, value=float(housing.frame[feat].median()))

input_df = pd.DataFrame([user_data])

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    # Target is in units of $100,000
    st.success(f"Predicted median house value: ${pred * 100000:,.0f}")
