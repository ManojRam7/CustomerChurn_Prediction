"""
Model training script for Customer Churn Prediction.

Loads preprocessed data, trains a Random Forest classifier,
and saves the model and feature column list to disk.

Usage:
    python -m src.model
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_and_save(
    data_path: str = "data/processed/preprocessed_from_script.csv",
    model_path: str = "models/random_forest_churn_from_script.pkl",
    columns_path: str = "models/model_columns.pkl",
) -> None:
    """Train a Random Forest model and save it along with feature column metadata."""
    df = pd.read_csv(data_path)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, model_path)
    joblib.dump(list(X.columns), columns_path)

    print(f"✅ Model saved to {model_path}")
    print(f"✅ Columns saved to {columns_path}")


if __name__ == "__main__":
    train_and_save()
