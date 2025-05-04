import streamlit as st
import time
import joblib
import numpy as np
import os

# ===== Load model & scaler =====
model_path = "model/final_model.pkl"
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()

try:
    loaded = joblib.load(model_path)
    model = loaded[0]
    scaler = loaded[1]
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ===== App UI =====
st.title("🩺 Diabetes Prediction Platform")
st.subheader("🔍 Enter your health details")

field_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
               "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
default_values = {name: 0 for name in field_names}

# ===== Input fields =====
user_inputs = {}
cols = st.columns(3)
for idx, name in enumerate(field_names):
    with cols[idx % 3]:
        user_inputs[name] = st.text_input(name, value=str(default_values[name]), help=f"Enter value for {name}")

# ===== Prediction function =====
def predict_diabetes(inputs):
    input_array = np.array([float(inputs[name]) for name in field_names]).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    probability = model.predict_proba(input_scaled)[0][1]
    prediction = "Diabetic" if probability >= 0.5 else "Not Diabetic"
    return prediction, probability

# ===== Buttons =====
col1, col2 = st.columns([1, 1])
with col1:
    predict_button = st.button("🚀 Predict Diabetes")
with col2:
    reset_button = st.button("🔄 Reset Fields")

if reset_button:
    st.experimental_rerun()

# ===== Prediction logic =====
if predict_button:
    try:
        for name in field_names:
            float(user_inputs[name])
    except ValueError:
        st.error("❌ Please enter valid numeric values for all fields.")
    else:
        with st.spinner("🔍 Analyzing health data..."):
            progress_bar = st.progress(0)
            for i in range(5):
                time.sleep(0.4)
                progress_bar.progress((i + 1) * 20)

            label, prob = predict_diabetes(user_inputs)
            st.success(f"📝 Prediction: **{label}**")
            st.markdown(f"📊 **Probability of Diabetes:** `{prob * 100:.2f}%`")

# streamlit run source/api/diabetes_prediction_app.py
