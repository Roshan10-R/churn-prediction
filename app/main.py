"""
FastAPI service for churn prediction.

Loads the single saved pipeline (preprocessing + model, trained in
src/train.py) and exposes /health and /predict. Because the saved object
is a full sklearn Pipeline, this endpoint never has to re-implement any
encoding logic -- it just hands the pipeline a one-row DataFrame built
straight from the validated request body.
"""

from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import CustomerData, PredictionResponse
from src.config import MODEL_PATH

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model

    try:
        print(f"Loading model from: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")

    except Exception:
        import traceback

        print("❌ Failed to load model")
        traceback.print_exc()
        _model = None

    yield


app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts the probability a customer will churn.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python -m src.train` first to generate models/best_model.pkl.",
        )

    # Pipeline expects a DataFrame with the same column names used in training
    df = pd.DataFrame([data.model_dump()])
    proba = _model.predict_proba(df)[0][1]

    return PredictionResponse(
        churn_probability=round(float(proba), 4),
        churn_prediction=bool(proba > 0.5),
    )
