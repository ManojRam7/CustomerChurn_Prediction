"""
Preprocessing module for Customer Churn Prediction.

Provides the Preprocessor class used for fitting and transforming
customer data before model training and inference.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


class Preprocessor:
    """Fit/transform pipeline for categorical encoding and numerical scaling."""

    def __init__(self):
        self.cat_cols = None
        self.num_cols = None
        self.label_encoders: dict = {}
        self.scaler: StandardScaler | None = None

    def fit(self, df: pd.DataFrame) -> None:
        """Fit encoders and scaler on the training DataFrame."""
        self.cat_cols = df.select_dtypes(include="object").columns.tolist()
        self.num_cols = (
            df.select_dtypes(include=np.number)
            .columns.drop("Churn", errors="ignore")
            .tolist()
        )

        for col in self.cat_cols:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            le.fit(df[col])
            self.label_encoders[col] = le

        self.scaler = StandardScaler()
        self.scaler.fit(df[self.num_cols])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a DataFrame using the fitted encoders and scaler."""
        df = df.copy()

        for col in self.cat_cols:
            le = self.label_encoders.get(col)
            if le:
                df[col] = df[col].astype(str)
                df[col] = df[col].map(lambda s: s if s in le.classes_ else "unknown")
                if "unknown" not in le.classes_:
                    le.classes_ = np.append(le.classes_, "unknown")
                df[col] = le.transform(df[col])
            else:
                df[col] = 0

        df[self.num_cols] = self.scaler.transform(df[self.num_cols])
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
