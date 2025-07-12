import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the model
with open('./model/pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

# --- App title ---
st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict if they would survive.")

# --- Collect input in correct order ---
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", min_value=0, max_value=80, value=30)
sibsp = st.number_input("Number of siblings/spouses aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of parents/children aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare Paid", min_value=0.0, value=32.0)
pclass = st.selectbox("Passenger Class", ["First", "Second", "Third"])
embark_town = st.selectbox("Embark Town", ["Southampton", "Cherbourg", "Queenstown"])

# --- Prediction trigger ---
if st.button("Predict Survival"):
    # Create DataFrame in correct order and format
    input_df = pd.DataFrame([[
        sex, age, sibsp, parch, fare, pclass, embark_town
    ]], columns=['sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'embark_town'])

    # Predict
    prediction = model.predict(input_df)

    # Result
    if prediction[0] == 1:
        st.success("🎉 The passenger is likely to survive!")
    else:
        st.error("😢 The passenger is unlikely to survive.")


