### 📊 Diabetes Prediction App
# Introduction

This project aims to develop a simple and user-friendly Streamlit application that enables patients to estimate their risk of diabetes at home based on medical test results. By inputting key health metrics such as age, BMI, HbA1c levels, and blood glucose, users can receive a quick prediction of their diabetes risk. The application leverages a machine learning model trained on a diabetes dataset to provide accurate and reliable predictions.

The project is built using the following technologies:

- Python: Core programming language for data processing and model development.
- Streamlit: Framework for creating an interactive web-based user interface.
- scikit-learn: Library for building and training the machine learning model.
- pandas and NumPy: Libraries for data manipulation and numerical computations.
- joblib: Used for saving and loading the trained machine learning model.

# Model Card

The model was deployed to the web using the Streamlit package, creating an interactive user interface for predictions. The application was integrated into a CI/CD framework using GitHub Actions, as defined in the .github/workflows/deploy.yml file. After building and testing the Streamlit app locally, it was prepared for deployment, with potential live testing supported by the infrastructure. Weights & Biases were utilized to manage and track all artifacts, including the trained model (final_model.pkl) and pipeline (final_pipeline.pkl), stored and monitored through the platform

![Project Architecture](MLOps_diabetes_architecture.png)

So, in general, the notebooks used were divided into 7 parts:

1.The search for data
2.Exploratory analysis
3.Pre-Processing
4.Tests
5.Splitting the data between training and testing.
6.Training
7.Test

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

## ⚙️ Virtual Environment
```bash
# Clone repository
git clone https://github.com/pham-ngoc-nhi/diabetes-prediction.git
cd diabetes-prediction

# Create and activate a virtual environment
python -m venv .venv

# To activate this environment , use the following command:
- For Linux, macOS
source .venv/bin/activate
- For Windows command line
.venv\Scripts\activate
- For Windows PowerShell
.venv\Scripts\Activate.ps1

# Install the dependencies
pip install -r requirements.txt
```

---

## Practice Notebooks
To understand the procedures of designing code locally, including important steps like fetching data, visualization, preprocessing, checking data, segregating data, training, and testing, you should run all notebooks step-by-step in the notebooks/ directory. Note that you should select the correct kernel (.venv) to avoid library conflicts.

You should register with Weights & Biases to store all artifacts, metrics, and models. You can create an account on wandb.ai. After that, create a .env file in the notebooks/ directory with the following content:
```bash
WANDB_API_KEY=Your_key_here
```

---
This WANDB_API_KEY is used to log into your W&B project inside each notebook.
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
