# 📊 Diabetes Prediction App

A web application that predicts the likelihood of diabetes based on health indicators such as glucose, BMI, insulin, age, etc.

## 🚀 Key features
- Train on the Pima Indian Diabetes dataset
- Complete preprocessing pipeline (handling zero values, feature creation, encoding, feature selection)
- XGBoost model with parameters optimized by Optuna
- Streamlit interface for direct prediction visualization
- Track and store the model with Weights & Biases (W&B)
---

## 🗂️ Folder structure

```
diabetes-prediction/
├── data/
│   └── raw_data/
├── model/
│   ├── final_model.pkl
│   ├── final_pipeline.pkl
│   └── pipeline_diabetes.pkl
├── notebooks/
│   ├── 1_fetch_data.ipynb
│   ├── 2_eda.ipynb
│   ├── 3_preprocessing.ipynb
│   ├── 4_data_check.ipynb
│   ├── 5_data_segregation.ipynb
│   ├── 6_train.ipynb
│   └── 7_test.ipynb
├── source/
│   └── api/
│       ├── __init__.py
│       ├── pipeline.py
│       ├── pipeline_config.py
│       ├── test_diabetes_prediction.py
│       └── diabetes_prediction_app.py
├── .github/workflows/deploy.yml
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Set up the environment

```bash
# Clone repository
git clone https://github.com/pham-ngoc-nhi/diabetes-prediction.git
cd diabetes-prediction

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate

# Install the dependencies
pip install -r requirements.txt
```

---

## ▶️ Run the application

```bash
streamlit run source/api/diabetes_prediction_app.py
```

---

## 🧠 Retrain the model

```bash
python source/api/pipeline.py
```

---

## 📝 Input example

| Indicator                  | Sample value|
|----------------------------|-------------|
| Pregnancies                | 2           |
| Glucose                    | 130         |
| BloodPressure              | 70          |
| SkinThickness              | 25          |
| Insulin                    | 100         |
| BMI                        | 28.5        |
| DiabetesPedigreeFunction   | 0.6         |
| Age                        | 35          |

---

## 🧪 Testing

```bash
pytest
```

---

## 📌 Lưu ý

- Đảm bảo sử dụng đúng phiên bản `scikit-learn==1.1.3` và `numpy==1.23.5` để tương thích với mô hình đã lưu.
- Các artifact đã được lưu trữ trên [W&B project](https://wandb.ai/ngocnhi-p4work-national-economics-university/diabetes).
