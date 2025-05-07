# 📊 Diabetes Prediction App

Ứng dụng web dự đoán khả năng mắc bệnh tiểu đường dựa trên các chỉ số sức khỏe như glucose, BMI, insulin, tuổi, v.v.

## 🚀 Tính năng nổi bật
- Huấn luyện trên bộ dữ liệu Pima Indian Diabetes
- Pipeline tiền xử lý đầy đủ (xử lý giá trị 0, tạo đặc trưng mới, mã hóa, chọn đặc trưng)
- Mô hình XGBoost với tham số được tối ưu bằng Optuna
- Giao diện Streamlit trực quan cho dự đoán trực tiếp
- Theo dõi và lưu trữ mô hình với Weights & Biases (W&B)

---

## 🗂️ Cấu trúc thư mục

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

## ⚙️ Cài đặt môi trường

```bash
# Clone repository
git clone https://github.com/pham-ngoc-nhi/diabetes-prediction.git
cd diabetes-prediction

# Tạo và kích hoạt môi trường ảo
python -m venv .venv
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

---

## ▶️ Chạy ứng dụng

```bash
streamlit run source/api/diabetes_prediction_app.py
```

---

## 🧠 Huấn luyện lại mô hình

```bash
python source/api/train_full_pipeline_model.py
```

---

## 📝 Ví dụ đầu vào

| Chỉ số                     | Giá trị mẫu |
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

## 🧪 Kiểm thử

```bash
pytest
```

---

## 📌 Lưu ý

- Đảm bảo sử dụng đúng phiên bản `scikit-learn==1.1.3` và `numpy==1.23.5` để tương thích với mô hình đã lưu.
- Các artifact đã được lưu trữ trên [W&B project](https://wandb.ai/ngocnhi-p4work-national-economics-university/diabetes).
