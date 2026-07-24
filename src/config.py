"""
Central configuration for the churn prediction pipeline.
Keeping paths, feature lists, and constants here means train.py,
preprocess.py, and the FastAPI app all stay in sync.
"""

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")

# --- MLflow ---
MLFLOW_EXPERIMENT_NAME = "Telco Churn"
MLFLOW_TRACKING_URI = "sqlite:///" + os.path.join(BASE_DIR, "mlflow.db")

# --- Target ---
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

# --- Feature groups (raw, pre-encoding) ---
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
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
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.2
