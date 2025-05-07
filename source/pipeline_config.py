import wandb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class CustomPreprocessorFull(BaseEstimator, TransformerMixin):
    def __init__(self, target_col='Outcome', wandb_project=None, wandb_entity=None, threshold=0.1, top_k_features=8):
        self.target_col = target_col.upper()
        self.threshold = threshold
        self.top_k_features = top_k_features
        self.zero_columns = []
        self.columns_to_flag = []
        self.columns_to_fill = []
        self.medians = {}
        self.selected_features = []
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity

    def get_zero_columns(self, df):
        return [col for col in df.columns if df[col].min() == 0 and col not in ['PREGNANCIES', self.target_col]]

    def analyze_missing_impact(self, df):
        columns_to_flag, columns_to_fill = [], []
        for col in self.zero_columns:
            temp = df.copy()
            temp[col + '_NA_FLAG'] = temp[col].isnull().astype(int)
            summary = temp.groupby(col + '_NA_FLAG')[self.target_col].mean()
            if len(summary) == 2 and abs(summary[0] - summary[1]) >= self.threshold:
                columns_to_flag.append(col)
            else:
                columns_to_fill.append(col)
        return columns_to_flag, columns_to_fill

    def process_missing_values(self, df):
        for col in self.columns_to_flag + self.columns_to_fill:
            if col in self.columns_to_flag:
                df[col + '_NA_FLAG'] = df[col].isnull().astype(int)
            df[col] = df[col].fillna(self.medians[col])
        return df

    def create_features(self, df):
        df['NEW_AGE_CAT'] = pd.cut(df['AGE'], bins=[0, 21, 50, np.inf], labels=['young', 'mature', 'senior'])
        df['NEW_BMI'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, np.inf], labels=['Underweight', 'Healthy', 'Overweight', 'Obese'])
        df['NEW_GLUCOSE'] = pd.cut(df['GLUCOSE'], bins=[0, 140, 200, np.inf], labels=['Normal', 'Prediabetes', 'Diabetes'])
        df['NEW_INSULIN_SCORE'] = df['INSULIN'].apply(lambda x: 'Normal' if 16 <= x <= 166 else 'Abnormal')
        df['NEW_GLUCOSE*INSULIN'] = df['GLUCOSE'] * df['INSULIN']
        df['NEW_GLUCOSE*PREGNANCIES'] = df['GLUCOSE'] * df['PREGNANCIES']
        return df

    def encode_features(self, df):
        cat_cols = [col for col in df.columns if df[col].dtype == 'O']
        binary_cols = [col for col in cat_cols if df[col].nunique() == 2]
        for col in binary_cols:
            df[col] = LabelEncoder().fit_transform(df[col])
        cat_cols = [col for col in cat_cols if col not in binary_cols]
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        return df

    def feature_selection(self, df):
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        X_numeric = X.select_dtypes(include=[np.number])
        mi_scores = mutual_info_classif(X_numeric, y, random_state=42)
        mi_df = pd.DataFrame({'Feature': X_numeric.columns, 'MI': mi_scores}).sort_values(by='MI', ascending=False)
        self.selected_features = mi_df['Feature'].head(self.top_k_features).tolist()
        return df[self.selected_features + [self.target_col]]

    def fit(self, X, y):
        df = pd.concat([X.copy(), y.reset_index(drop=True)], axis=1)
        df.columns = [col.upper() for col in df.columns]
        self.target_col = self.target_col.upper()

        self.zero_columns = self.get_zero_columns(df)
        df[self.zero_columns] = df[self.zero_columns].replace(0, np.nan)

        self.columns_to_flag, self.columns_to_fill = self.analyze_missing_impact(df)
        for col in self.columns_to_flag + self.columns_to_fill:
            self.medians[col] = df[col].median()

        df = self.process_missing_values(df)
        df = self.create_features(df)
        df = self.encode_features(df)
        df = self.feature_selection(df)
        return self

    def transform(self, X):
        df = X.copy()
        df.columns = [col.upper() for col in df.columns]
        df[self.zero_columns] = df[self.zero_columns].replace(0, np.nan)
        df = self.process_missing_values(df)
        df = self.create_features(df)
        df = self.encode_features(df)

        # Thêm cột bị thiếu do feature selection
        for col in self.selected_features:
            if col not in df.columns:
                df[col] = 0  # hoặc np.nan nếu bạn muốn kiểm tra thiếu

        df = df[self.selected_features]

        # Ghi lên W&B nếu được cấu hình
        if self.wandb_project and self.wandb_entity:
            wandb.login()
            run = wandb.init(project=self.wandb_project, entity=self.wandb_entity, job_type="preprocessing")
            df.to_csv("processed_data.csv", index=False)
            artifact = wandb.Artifact("processed_data", type="dataset")
            artifact.add_file("processed_data.csv")
            run.log_artifact(artifact)
            run.finish()

        print("NaNs after preprocessing:", df.isnull().sum().sum())
        return df


from sklearn.preprocessing import StandardScaler

def create_preprocessing_pipeline():
    return Pipeline([
        ('custom_preprocessing_full', CustomPreprocessorFull(
            wandb_project="diabetes",
            wandb_entity="ngocnhi-p4work-national-economics-university"
        )),
        ('scaler', StandardScaler())
    ])

