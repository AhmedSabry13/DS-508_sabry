import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models & encoders
price_model = joblib.load("../models/price_regression_model.pkl")
city_encoder = joblib.load("../models/city_label_encoder.pkl")
state_encoder = joblib.load("../models/statezip_label_encoder.pkl")
feature_names = joblib.load("../models/regression_features.pkl")

st.set_page_config(page_title="Real Estate Price Predictor", layout="centered")

st.title("🏠 Real Estate Price Prediction")
st.write("Enter house details to predict the price and get an explanation.")

# -------------------------------
# User Inputs
# -------------------------------
city = st.text_input("City", "Seattle")
statezip = st.text_input("State Zip", "WA 98103")
#----------
sale_year = st.number_input("Sale Year", 2000, 2025, 2015)
sale_month = st.slider("Sale Month", 1, 12, 6)

sqft_above = st.number_input("Above Ground Area (sqft)", 300, 10000, 1500)
sqft_basement = st.number_input("Basement Area (sqft)", 0, 5000, 500)
#-------------------

bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1.0, 10.0, 2.0)
sqft_living = st.number_input("Living Area (sqft)", 300, 10000, 2000)
sqft_lot = st.number_input("Lot Size (sqft)", 500, 50000, 5000)
floors = st.number_input("Floors", 1.0, 5.0, 1.0)

waterfront = st.selectbox("Waterfront", [0, 1])
view = st.slider("View Quality (0–4)", 0, 4, 1)
condition = st.slider("Condition (1–5)", 1, 5, 3)

yr_built = st.number_input("Year Built", 1900, 2024, 2000)
yr_renovated = st.number_input("Year Renovated (0 if never)", 0, 2024, 0)

# -------------------------------
# Encode Inputs
# -------------------------------
try:
    city_encoded = city_encoder.transform([city])[0]
    state_encoded = state_encoder.transform([statezip])[0]
except ValueError:
    st.error("City or StateZip not seen during training.")
    st.stop()

input_data = {
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "sqft_living": sqft_living,
    "sqft_lot": sqft_lot,
    "floors": floors,
    "waterfront": waterfront,
    "view": view,
    "condition": condition,
    "yr_built": yr_built,
    "yr_renovated": yr_renovated,
    "sqft_above": sqft_above,
    "sqft_basement": sqft_basement,
    "sale_year": sale_year,
    "sale_month": sale_month,
    "city_encoded": city_encoded,
    "statezip_encoded": state_encoded
}

input_df = pd.DataFrame([input_data])

# 🔥 Align columns exactly as training
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# -------------------------------
# Predict
# -------------------------------
if st.button("Predict Price 💰"):
    predicted_price = price_model.predict(input_df)[0]

    st.success(f"🏷️ Estimated House Price: ${predicted_price:,.0f}")

    # -------------------------------
    # LLM-style Interpretation (Rule-Based)
    # -------------------------------
    explanation = f"""
    The predicted price is influenced mainly by the living area ({sqft_living} sqft),
    number of bathrooms ({bathrooms}), and the house location ({city}).
    
    {'Waterfront access significantly increases value.' if waterfront else 'Lack of waterfront keeps the price closer to the market average.'}
    
    Overall, this price appears realistic given the property characteristics and typical market trends in {city}.
    """

    st.subheader("🧠 Explanation")
    st.write(explanation)
