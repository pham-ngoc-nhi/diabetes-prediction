import streamlit as st
import time
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Load the trained model with absolute path
model_path = os.path.join("model", "final_model.pkl")

if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()
try:
    with open(model_path, "rb") as model_file:
        model, scaler = joblib.load(model_path)
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
#def predict_diabetes(inputs):
 #   input_array = np.array([float(inputs[name]) for name in field_names]).reshape(1, -1)
  #  prediction = model.predict(input_array)
   # return "Diabetic" if prediction[0] == 1 else "Not Diabetic"

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
#if predict_button:
 #   try:
  #      for name in field_names:
   #         float(user_inputs[name])
#    except ValueError:
 #       st.error("Please enter valid numeric values for all fields.")
  #  else:
   #     with st.spinner("Analyzing health data..."):
    #        progress_bar = st.progress(0)
     #       for i in range(5):
      #          time.sleep(1)
       #         progress_bar.progress((i + 1) * 20)
        #    result = predict_diabetes(user_inputs)
         #   st.success(f"📝 Prediction: **{result}**")
            
if predict_button:
    try:
        input_values = {name: float(user_inputs[name]) for name in field_names}
        
        # Validate: Không có giá trị âm
        for name, val in input_values.items():
            if val < 0:
                st.error(f"❌ Giá trị {name} không được âm.")
                st.stop()

        # Scale input
        input_df = pd.DataFrame([input_values])
        input_scaled = scaler.transform(input_df)

        # Predict
        with st.spinner("🔍 Đang phân tích..."):
            progress = st.progress(0)
            for i in range(5):
                time.sleep(0.3)
                progress.progress((i + 1) * 20)

            prob = model.predict_proba(input_scaled)[0][1]
            label = "🩸 **Diabetic**" if prob >= 0.5 else "✅ **Not Diabetic**"

            st.markdown(f"## 🧪 Result: {label}")
            st.markdown(f"**📊 Probability of Diabetes:** `{prob*100:.2f}%`")

        # Plot input values
        st.markdown("### 🔬 Input Summary")
        fig, ax = plt.subplots()
        ax.bar(input_values.keys(), input_values.values(), color='skyblue')
        ax.set_ylabel("Value")
        ax.set_title("Your Health Indicator Values")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    except ValueError:
        st.error("❌ Vui lòng nhập các giá trị hợp lệ (số thực không âm).")