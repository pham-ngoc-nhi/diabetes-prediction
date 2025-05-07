import streamlit as st
import time
import joblib
import numpy as np
import pandas as pd
import os
import sys

# Fix import path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ===== Load final_pipeline.pkl (includes preprocessing + model) =====
final_pipeline_path = "model/final_pipeline.pkl"

if not os.path.exists(final_pipeline_path):
    st.error(f"🚫 Pipeline file not found: {final_pipeline_path}")
    st.stop()

try:
    pipeline = joblib.load(final_pipeline_path)
except Exception as e:
    st.error(f"❌ Error loading pipeline: {e}")
    st.stop()

# ===== App UI =====
st.title("🧪 Diabetes Prediction Platform")
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
    df_input = pd.DataFrame([inputs])
    df_input = df_input.astype(float)
    prob = pipeline.predict_proba(df_input)[0][1]
    label = "Diabetic" if prob >= 0.5 else "Not Diabetic"
    return label, prob

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
            for percent in range(0, 101, 5):
                time.sleep(0.05)
                progress_bar.progress(percent)

            label, prob = predict_diabetes(user_inputs)
            st.success(f"📜 Prediction: **{label}** ({prob:.2%} probability)")



# streamlit run source/api/diabetes_prediction_app.py
