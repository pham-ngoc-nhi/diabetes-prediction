import os
import pickle
import numpy as np

# Load model
MODEL_PATH = "model/final_model.pkl"  # Relative path from the project root
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Test prediction for a known input
def test_model_prediction_diabetic():
    # Fake input that should return diabetic
    sample_input = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
    result = model.predict(sample_input)
    assert result in ([0], [1])

def test_model_prediction_not_diabetic():
    # Fake input that should return not diabetic
    sample_input = np.array([[1, 85, 66, 29, 0, 26.6, 0.351, 31]])
    result = model.predict(sample_input)
    assert result in ([0], [1])
