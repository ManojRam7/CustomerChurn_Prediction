"""
Unit tests for the Preprocessor class.
"""

import pandas as pd

from src.preprocessing import Preprocessor


def test_preprocessor_output_shape():
    """Transformed DataFrame should have the same shape as input."""
    df = pd.DataFrame(
        {
            "Gender": ["Male", "Female"],
            "Subscription_Type": ["Basic", "Premium"],
            "Contract_Length": ["Monthly", "Annual"],
            "CustomerID": [1, 2],
            "Age": [25, 30],
            "Tenure": [12, 24],
            "Usage_Frequency": [5, 10],
            "Support_Calls": [1, 2],
            "Payment_Delay": [0, 1],
            "Total_Spend": [100.0, 200.0],
            "Last_Interaction": [10, 20],
        }
    )
    pre = Preprocessor()
    pre.fit(df)
    transformed = pre.transform(df)
    assert transformed.shape == df.shape


def test_preprocessor_no_nulls():
    """Transformed DataFrame should contain no null values."""
    df = pd.DataFrame(
        {
            "Gender": ["Male", "Female"],
            "Subscription_Type": ["Basic", "Premium"],
            "Contract_Length": ["Monthly", "Annual"],
            "CustomerID": [1, 2],
            "Age": [25, 30],
            "Tenure": [12, 24],
            "Usage_Frequency": [5, 10],
            "Support_Calls": [1, 2],
            "Payment_Delay": [0, 1],
            "Total_Spend": [100.0, 200.0],
            "Last_Interaction": [10, 20],
        }
    )
    pre = Preprocessor()
    transformed = pre.fit_transform(df)
    assert not transformed.isnull().any().any()


def test_preprocessor_unseen_label():
    """Preprocessor should handle unseen categorical values without crashing."""
    train_df = pd.DataFrame(
        {
            "Gender": ["Male", "Female"],
            "CustomerID": [1, 2],
            "Age": [25, 30],
        }
    )
    test_df = pd.DataFrame(
        {
            "Gender": ["Other"],  # unseen label
            "CustomerID": [3],
            "Age": [28],
        }
    )
    pre = Preprocessor()
    pre.fit(train_df)
    transformed = pre.transform(test_df)
    assert transformed.shape[0] == 1
