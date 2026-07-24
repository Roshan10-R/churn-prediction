"""
Basic tests for the churn API.
Run with: pytest tests/ -v
(Requires models/best_model.pkl to exist -- run `python -m src.train` first
for the /predict tests to return real predictions rather than a 503.)
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    # Using TestClient as a context manager triggers  the app's lifespan
    # startup/shutdown events (that's what actually loads the model).
    with TestClient(app) as c:
        yield c

VALID_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.5,
    "TotalCharges": 840.0,
}


def test_health(client):
    response = client.get("/health")
    print(response.json())
    assert response.status_code == 200


def test_predict_valid_input(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "churn_probability" in body
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_prediction"], bool)


def test_predict_missing_field(client):
    payload = VALID_PAYLOAD.copy()
    del payload["tenure"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error


def test_predict_invalid_type(client):
    payload = VALID_PAYLOAD.copy()
    payload["tenure"] = "not_a_number"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_category(client):
    payload = VALID_PAYLOAD.copy()
    payload["Contract"] = "Lifetime"  # not a valid Literal option
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_tenure_rejected(client):
    payload = VALID_PAYLOAD.copy()
    payload["tenure"] = -5
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
