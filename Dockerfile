# Sử dụng image Python chính thức làm base image
FROM python:3.11-slim

# Đặt thư mục làm việc trong container
WORKDIR /app

# Sao chép requirements.txt trước để tận dụng cache của Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn và mô hình
COPY . .

# Chạy ứng dụng Streamlit khi container khởi động
CMD ["streamlit", "run", "source/api/diabetes_prediction_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
