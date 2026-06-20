"""
API integration tests for the Customer Churn Prediction FastAPI application.
"""

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

VALID_CUSTOMER = {
    "CustomerID": 2,
    "Age": 30,
    "Gender": "Female",
    "Tenure": 39,
    "Usage_Frequency": 14,
    "Support_Calls": 5,
    "Payment_Delay": 18,
    "Subscription_Type": "Standard",
    "Contract_Length": "Annual",
    "Total_Spend": 932.0,
    "Last_Interaction": 17,
}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"



def test_root_form_page():
    response = client.get("/")
    assert response.status_code == 200

def test_model_info():
    response = client.get("/model_info")
    assert response.status_code == 200
    assert "model" in response.json()


def test_predict_valid():
    response = client.post("/predict", json={"data": [VALID_CUSTOMER]})
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert isinstance(response.json()["predictions"], list)


def test_predict_invalid_gender():
    invalid = {**VALID_CUSTOMER, "Gender": "Alien"}
    response = client.post("/predict", json={"data": [invalid]})
    assert response.status_code == 422


def test_predict_invalid_subscription():
    invalid = {**VALID_CUSTOMER, "Subscription_Type": "Gold"}
    response = client.post("/predict", json={"data": [invalid]})
    assert response.status_code == 422


def test_predict_invalid_contract():
    invalid = {**VALID_CUSTOMER, "Contract_Length": "Weekly"}
    response = client.post("/predict", json={"data": [invalid]})
    assert response.status_code == 422


def test_predict_batch():
    second_customer = {
        "CustomerID": 1001,
        "Age": 35,
        "Gender": "Male",
        "Tenure": 12,
        "Usage_Frequency": 5,
        "Support_Calls": 1,
        "Payment_Delay": 0,
        "Subscription_Type": "Basic",
        "Contract_Length": "Quarterly",
        "Total_Spend": 500.0,
        "Last_Interaction": 30,
    }
    response = client.post("/predict", json={"data": [VALID_CUSTOMER, second_customer]})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


def test_predict_missing_fields():
    response = client.post("/predict", json={"data": [{"CustomerID": 2, "Age": 30}]})
    assert response.status_code == 422
