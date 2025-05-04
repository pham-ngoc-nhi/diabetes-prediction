"""
Creator: Morsinaldo Medeiros and Alessandro Neto, updated by Grok
Date: 16 May 2022, updated May 2025
Create API for Diabetes Prediction
"""
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import os
import wandb
import sys
from source.api.pipeline import FeatureSelector, CategoricalTransformer, NumericalTransformer

pipe = None  # Biến toàn cục để lưu model đã load

# global variables
setattr(sys.modules["__main__"], "FeatureSelector", FeatureSelector)
setattr(sys.modules["__main__"], "CategoricalTransformer", CategoricalTransformer)
setattr(sys.modules["__main__"], "NumericalTransformer", NumericalTransformer)

# name of the model artifact
artifact_model_name = "diabetes/model_export:latest"

# initiate the wandb project
run = wandb.init(project="diabetes", job_type="api")

# create the api
app = FastAPI(
    title="Diabetes Prediction API",
    description="API to predict diabetes risk based on patient features using a trained ML model.",
    version="1.0.0"
)
@app.on_event("startup")
def load_model():
    global pipe
    run = wandb.init(project="diabetes", job_type="api", reinit=True)
    artifact = run.use_artifact(artifact_model_name)
    artifact_dir = artifact.download()
    model_export_path = os.path.join(artifact_dir, "final_model_xgboost.pkl")
    pipe = joblib.load(model_export_path)
    run.finish()


# declare request example data using pydantic
class Person(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int 
    SkinThickness: int 
    Insulin: int 
    BMI: float 
    DiabetesPedigreeFunction: float 
    Age: int 
    class Config:
        schema_extra = {
            "example": {
                "Pregnancies": 1,
                "Glucose": 90,
                "BloodPressure": 60,
                "SkinThickness": 30,
                "Insulin": 70,
                "BMI": 25,
                "DiabetesPedigreeFunction": 0.178,
                "Age": 20
            }
        }

# Root endpoint with a welcome message
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <p><span style="font-size:28px"><strong>Diabetes Prediction API</strong></span></p>
    <p><span style="font-size:20px">This API predicts diabetes risk based on patient features using a machine learning model trained on the Pima Indians Diabetes Dataset. Try it out at <a href="/docs">/docs</a>!</span></p>
    """

# Prediction endpoint
@app.post("/predict")
async def get_inference(person: Person):
    try:
        df = pd.DataFrame([person.dict()])
        predict = pipe.predict(df)
        result = "Positive" if predict[0] == 1 else "Negative"
        return {"prediction": result, "details": "Diabetes risk prediction based on input features"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Run the app with: uvicorn main:app --reload