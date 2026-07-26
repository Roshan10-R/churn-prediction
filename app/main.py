"""
FastAPI service for churn prediction.

Loads the single saved pipeline (preprocessing + model, trained in
src/train.py) and exposes /health and /predict. Because the saved object
is a full sklearn Pipeline, this endpoint never has to re-implement any
encoding logic—it simply passes a one-row DataFrame built from the
validated request body to the pipeline.
"""

from contextlib import asynccontextmanager
import time
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import CustomerData, PredictionResponse
from monitoring.logger import initialize_log, log_prediction
from src.config import MODEL_PATH
from src.config import MLFLOW_TRACKING_URI

_model = None
_model_error = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_error

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        _model = mlflow.sklearn.load_model("models:/telco-churn-model@production")
        _model_error = None
        initialize_log()
    except Exception as e:
        _model = None
        _model_error = repr(e)

    yield


app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts the probability that a customer will churn.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_error": _model_error,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python -m src.train` first to generate models/best_model.pkl.",
        )

    start_time = time.perf_counter()

    # Convert validated request to DataFrame
    df = pd.DataFrame([data.model_dump()])

    # Make prediction
    prediction = int(_model.predict(df)[0])
    probability = float(_model.predict_proba(df)[0][1])

    # Log prediction
    log_prediction(
        features=data.model_dump(),
        prediction=prediction,
        probability=probability,
        start_time=start_time,
    )

    return PredictionResponse(
        churn_probability=round(probability, 4),
        churn_prediction=bool(prediction),
    )