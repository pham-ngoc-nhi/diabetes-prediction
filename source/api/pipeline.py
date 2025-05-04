from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

# Custom feature engineering transformer
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df.columns = df.columns.str.upper()

        # Replace 0s in selected columns with NaN
        zero_replace = ["GLUCOSE", "BLOODPRESSURE", "SKINTHICKNESS", "INSULIN", "BMI"]
        for col in zero_replace:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())

        # Feature engineering
        df["NEW_GLUCOSE*INSULIN"] = df["GLUCOSE"] * df["INSULIN"]
        df["NEW_GLUCOSE_PREGNANCIES"] = df["GLUCOSE"] * df["PREGNANCIES"]
        df["NEW_GLUCOSE_PREDIABETES"] = df["GLUCOSE"].apply(lambda x: 1 if 100 <= x <= 125 else 0)
        df["NEW_BMI_OBESE"] = df["BMI"].apply(lambda x: 1 if x >= 30 else 0)
        df["NEW_AGE_CAT"] = pd.cut(df["AGE"], bins=[20, 40, 60, 80], labels=["young", "mature", "highmature"]).astype(str)
        df["NEW_AGE_GLUCOSE_NOM_highmature"] = (df["AGE"] > 50).astype(int)

        # Drop object columns except NEW_AGE_CAT
        drop_cols = [col for col in df.columns if df[col].dtype == 'O' and col != 'NEW_AGE_CAT']
        df.drop(columns=drop_cols, inplace=True, errors='ignore')

        return df

# Create the preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ("feature_engineering", FeatureEngineer())
])
