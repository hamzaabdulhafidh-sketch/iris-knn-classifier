import streamlit as st
import pandas as pd
import joblib

model = joblib.load("iris_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🌸 Iris Flower Species Predictor")

sl = st.number_input("Sepal Length")
sw = st.number_input("Sepal Width")
pl = st.number_input("Petal Length")
pw = st.number_input("Petal Width")

if st.button("Predict"):
    data = pd.DataFrame(
        [[sl, sw, pl, pw]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )

    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)

    st.success(f"Prediction: {prediction[0]}")