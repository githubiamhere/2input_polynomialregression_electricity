import streamlit as st
import pandas as pd
import joblib

model = joblib.load("2input_electricity_bill_prediction_model.pkl")
poly = joblib.load("poly.pkl")

st.title("Electrical_Bill AC Price Prediction")

ac_units = st.number_input(
    "Enter AC Units : ",
    min_value=1.0,
    value=100.0
)

fan_units = st.number_input(
    "Enter Fan Units : ",
    min_value=1.0,
    value=100.0
)

if st.button("Predict"):

    valid = True

    if ac_units <= 0 or ac_units > 150:
        st.error("AC Units should be between 0 and 150")
        valid = False

    if fan_units <= 0 or fan_units > 150:
        st.error("Fan Units should be between 0 and 150")
        valid = False

    if valid:

        input_data = pd.DataFrame({
            "AC_Units": [ac_units],
            "Fan_Units": [fan_units]
        })

        input_data_poly = poly.transform(input_data)

        prediction = model.predict(input_data_poly)

        pred = prediction[0]

        st.success(f"Predicted Price: ₹{pred:.2f}")



if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
