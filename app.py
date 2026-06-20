"""
Customer Churn Prediction API
FastAPI application serving both a web UI and REST API for telecom customer churn prediction.
"""

from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator

# Load preprocessor, model, and model columns at startup
preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/random_forest_churn_from_script.pkl")
model_columns = joblib.load("models/model_columns.pkl")

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict telecom customer churn using a trained Random Forest model.",
    version="1.0.0",
)
templates = Jinja2Templates(directory="templates")


# ── HTML Frontend ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    """Render the churn prediction web form."""
    return templates.TemplateResponse(
        request, "index.html", {"prediction": None, "form_data": {}}
    )


@app.post("/", response_class=HTMLResponse)
async def form_post(
    request: Request,
    CustomerID: int = Form(...),
    Age: int = Form(...),
    Gender: str = Form(...),
    Tenure: int = Form(...),
    Usage_Frequency: int = Form(...),
    Support_Calls: int = Form(...),
    Payment_Delay: int = Form(...),
    Subscription_Type: str = Form(...),
    Contract_Length: str = Form(...),
    Total_Spend: float = Form(...),
    Last_Interaction: int = Form(...),
):
    """Handle form submission and return churn prediction."""
    form_data = {
        "CustomerID": CustomerID,
        "Age": Age,
        "Gender": Gender,
        "Tenure": Tenure,
        "Usage_Frequency": Usage_Frequency,
        "Support_Calls": Support_Calls,
        "Payment_Delay": Payment_Delay,
        "Subscription_Type": Subscription_Type,
        "Contract_Length": Contract_Length,
        "Total_Spend": Total_Spend,
        "Last_Interaction": Last_Interaction,
    }
    try:
        data = [
            [
                CustomerID,
                Age,
                Gender,
                Tenure,
                Usage_Frequency,
                Support_Calls,
                Payment_Delay,
                Subscription_Type,
                Contract_Length,
                Total_Spend,
                Last_Interaction,
            ]
        ]
        df = pd.DataFrame(data, columns=model_columns)
        processed = preprocessor.transform(df)
        processed = pd.DataFrame(processed, columns=model_columns)
        pred = model.predict(processed)[0]
        prediction = "Churn" if pred == 1 else "No Churn"
    except Exception as e:
        prediction = f"Error: {str(e)}"
    return templates.TemplateResponse(
        request,
        "index.html",
        {"prediction": prediction, "form_data": form_data},
    )


# ── REST API ───────────────────────────────────────────────────────────────────

class CustomerRow(BaseModel):
    CustomerID: int
    Age: int
    Gender: str
    Tenure: int
    Usage_Frequency: int
    Support_Calls: int
    Payment_Delay: int
    Subscription_Type: str
    Contract_Length: str
    Total_Spend: float
    Last_Interaction: int

    @field_validator("Gender")
    @classmethod
    def gender_allowed(cls, v: str) -> str:
        if v not in ("Male", "Female"):
            raise ValueError("Gender must be 'Male' or 'Female'")
        return v

    @field_validator("Subscription_Type")
    @classmethod
    def subscription_allowed(cls, v: str) -> str:
        if v not in ("Basic", "Standard", "Premium"):
            raise ValueError("Subscription_Type must be 'Basic', 'Standard', or 'Premium'")
        return v

    @field_validator("Contract_Length")
    @classmethod
    def contract_allowed(cls, v: str) -> str:
        if v not in ("Monthly", "Quarterly", "Annual"):
            raise ValueError("Contract_Length must be 'Monthly', 'Quarterly', or 'Annual'")
        return v


class CustomerData(BaseModel):
    data: List[CustomerRow]


@app.post("/predict", summary="Predict churn for one or more customers")
def predict(customer_data: CustomerData):
    """
    Accept a list of customer records and return churn predictions.
    Returns 0 (No Churn) or 1 (Churn) for each record.
    """
    columns = model_columns
    data_rows = [
        [
            row.CustomerID,
            row.Age,
            row.Gender,
            row.Tenure,
            row.Usage_Frequency,
            row.Support_Calls,
            row.Payment_Delay,
            row.Subscription_Type,
            row.Contract_Length,
            row.Total_Spend,
            row.Last_Interaction,
        ]
        for row in customer_data.data
    ]
    for row in data_rows:
        if len(row) != len(columns):
            raise HTTPException(
                status_code=400,
                detail=f"Each row must have {len(columns)} values — invalid input shape.",
            )
    try:
        df = pd.DataFrame(data_rows, columns=columns)
        processed = preprocessor.transform(df)
        processed = pd.DataFrame(processed, columns=model_columns)
        preds = model.predict(processed)
        return {"predictions": preds.tolist()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error.")


# ── Utility Endpoints ──────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
def health():
    """Return service health status."""
    return {"status": "ok"}


@app.get("/model_info", summary="Model metadata")
def model_info():
    """Return information about the loaded model."""
    return {"model": "RandomForest", "version": "1.0", "features": model_columns}


@app.post("/reload_model", summary="Reload model from disk")
def reload_model():
    """Hot-reload the ML model from disk without restarting the server."""
    global model
    model = joblib.load("models/random_forest_churn_from_script.pkl")
    return {"status": "model reloaded"}
