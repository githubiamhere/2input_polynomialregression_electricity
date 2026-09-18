import streamlit as st
import pandas as pd
import joblib

model = joblib.load("2input_electricity_bill_prediction_model.pkl")
poly = joblib.load("poly.pkl")
st.title("Electrical_Bill AC Price Prediction")

units = st.number_input(
    "Enter AC Units : ",
    min_value=0.0,
    max_value=150.0,
    value=100.0
)


if st.button("Predict"):
        input_data = pd.DataFrame({
            "AC_Units": [units],
        })
        input_data_poly = poly.transform(input_data)
        prediction = model.predict(input_data_poly)
        pred = prediction[0]
        st.success(f"Predicted Price: ₹{pred:.2f} Rs")
