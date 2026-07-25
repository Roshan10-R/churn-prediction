import csv
import os
import uuid
import time
from datetime import datetime

# Folder and file locations
LOG_DIR = "monitoring/logs"
LOG_FILE = os.path.join(LOG_DIR, "prediction_log.csv")

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)


def initialize_log():
    """
    Creates prediction_log.csv with headers if it doesn't exist.
    """

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow([
                "request_id",
                "timestamp",
                "latency_ms",
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "tenure",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod",
                "MonthlyCharges",
                "TotalCharges",
                "prediction",
                "probability"
            ])


def log_prediction(features: dict,
                   prediction: int,
                   probability: float,
                   start_time: float):
    """
    Logs every prediction request.

    Parameters
    ----------
    features : dict
        Input JSON received by FastAPI

    prediction : int
        Model prediction (0/1)

    probability : float
        Churn probability

    start_time : float
        time.perf_counter() recorded before prediction
    """

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    row = [
        str(uuid.uuid4()),
        datetime.utcnow().isoformat(),
        latency_ms,

        features.get("gender"),
        features.get("SeniorCitizen"),
        features.get("Partner"),
        features.get("Dependents"),
        features.get("tenure"),
        features.get("PhoneService"),
        features.get("MultipleLines"),
        features.get("InternetService"),
        features.get("OnlineSecurity"),
        features.get("OnlineBackup"),
        features.get("DeviceProtection"),
        features.get("TechSupport"),
        features.get("StreamingTV"),
        features.get("StreamingMovies"),
        features.get("Contract"),
        features.get("PaperlessBilling"),
        features.get("PaymentMethod"),
        features.get("MonthlyCharges"),
        features.get("TotalCharges"),

        prediction,
        round(probability, 4)
    ]

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)