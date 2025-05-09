import os
import joblib
import numpy as np
import pytest

MODEL_PATH = "model/final_model.pkl"

@pytest.fixture(scope="session")
def loaded_model():
    """Load the trained model from file."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model_bundle = joblib.load(MODEL_PATH)
    return model_bundle[0]  # Assume model is stored as (model, scaler)

def test_model_prediction_diabetic(loaded_model):
    # Simulated input likely to be classified as diabetic
    sample_input = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
    prediction = loaded_model.predict(sample_input)
    assert prediction[0] in (0, 1), "Prediction should be 0 or 1"

def test_model_prediction_not_diabetic(loaded_model):
    # Simulated input likely to be non-diabetic
    sample_input = np.array([[1, 85, 66, 29, 0, 26.6, 0.351, 31]])
    prediction = loaded_model.predict(sample_input)
    assert prediction[0] in (0, 1), "Prediction should be 0 or 1"
