import sys, os
import pandas as pd
import joblib
import wandb

from sklearn.ensemble import RandomForestClassifier

# Fix relative import when running as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.api.pipeline_config import create_preprocessing_pipeline

# === Load raw data ===
data_path = "data/raw_data/raw_data.csv"
df = pd.read_csv(data_path)
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# === Fit preprocessing pipeline ===
pipeline = create_preprocessing_pipeline()
X_processed = pipeline.fit_transform(X, y)
print("✅ Preprocessing pipeline fitted.")

# === Save pipeline only ===
os.makedirs("model", exist_ok=True)
pipeline_path = os.path.join("model", "pipeline_diabetes.pkl")
joblib.dump(pipeline, pipeline_path)
print(f"✅ Saved preprocessing pipeline to {pipeline_path}")

# === Train model on preprocessed full data ===
import joblib
loaded = joblib.load("model/final_model.pkl")
model = loaded[0]
scaler = loaded[1]
model.fit(X_processed, y)
print("✅ Model trained on preprocessed data.")

# === Save model only ===
model_path = os.path.join("model", "final_model.pkl")
joblib.dump((model, scaler), model_path)

# === Log both to W&B ===
run = wandb.init(project="diabetes", entity="ngocnhi-p4work-national-economics-university", job_type="train_model")
artifact_model = wandb.Artifact("final_model", type="model")
artifact_model.add_file(model_path)
run.log_artifact(artifact_model)

artifact_pipeline = wandb.Artifact("pipeline_diabetes", type="pipeline")
artifact_pipeline.add_file(pipeline_path)
run.log_artifact(artifact_pipeline)

run.finish()

from sklearn.pipeline import Pipeline

final_pipeline = Pipeline([
    ("preprocessing", pipeline),
    ("model", model)
])
joblib.dump(final_pipeline, "model/final_pipeline.pkl")

