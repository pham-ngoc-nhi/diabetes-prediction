# Use the official Python image as the base image.
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt first to take advantage of Docker's cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire source code and model
COPY . .

# Run the Streamlit application when the container starts
CMD ["streamlit", "run", "source/api/diabetes_prediction_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
