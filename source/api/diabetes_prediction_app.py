import streamlit as st
import time
import pickle
import numpy as np
import os

# Load the trained model with absolute path
model_path = "C:/Users/Admin/Desktop/MLOps/hi/final/model/final_model.pkl"
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()
try:
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# App Title
st.title("🩺 Diabetes Prediction Platform")
st.subheader("🔍 Enter your health details")

# Field names for diabetes dataset
field_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
               "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# Default values
default_values = {name: 0 for name in field_names}

# Create input fields
user_inputs = {}
cols = st.columns(3)
for idx, name in enumerate(field_names):
    with cols[idx % 3]:
        user_inputs[name] = st.text_input(name, value=str(default_values[name]), help=f"Enter value for {name}")

# Prediction function
def predict_diabetes(inputs):
    input_array = np.array([float(inputs[name]) for name in field_names]).reshape(1, -1)
    prediction = model.predict(input_array)
    return "Diabetic" if prediction[0] == 1 else "Not Diabetic"

# Button Actions
col1, col2 = st.columns([1, 1])
with col1:
    predict_button = st.button("🚀 Predict Diabetes")
with col2:
    reset_button = st.button("🔄 Reset Fields")

# Reset Fields
if reset_button:
    st.experimental_rerun()

# Prediction with Progress Bar
if predict_button:
    try:
        for name in field_names:
            float(user_inputs[name])
    except ValueError:
        st.error("Please enter valid numeric values for all fields.")
    else:
        with st.spinner("Analyzing health data..."):
            progress_bar = st.progress(0)
            for i in range(5):
                time.sleep(1)
                progress_bar.progress((i + 1) * 20)
            result = predict_diabetes(user_inputs)
            st.success(f"📝 Prediction: **{result}**")