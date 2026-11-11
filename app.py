import streamlit as st
import joblib

model = joblib.load("iris_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Iris Flower Classifier")

sl = st.number_input("Sepal Length")
sw = st.number_input("Sepal Width")
pl = st.number_input("Petal Length")
pw = st.number_input("Petal Width")

if st.button("Predict"):
    data = [[sl, sw, pl, pw]]
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)

    st.success(f"Prediction: {prediction[0]}")