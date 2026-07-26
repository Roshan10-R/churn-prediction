import mlflow
from mlflow import MlflowClient
import json
from pathlib import Path

from src.config import (
    MODEL_NAME,
    MLFLOW_TRACKING_URI,
)

from src.train import train_best_model

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()

production = client.get_model_version_by_alias(
    MODEL_NAME,
    "production",
)

production_run = client.get_run(
    production.run_id
)

production_f1 = production_run.data.metrics["f1"]

print(f"Production F1: {production_f1:.4f}")
new_model = train_best_model()

new_f1 = new_model["best_metrics"]["f1"]

print(f"New F1: {new_f1:.4f}")

if new_f1 > production_f1:

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="production",
        version=new_model["registered_version"],
    )

    print(
        f"Promoted Version "
        f"{new_model['registered_version']} "
        "to production."
    )

else:

    print(
        "Current production model remains unchanged."
    )

SUMMARY_FILE = Path("monitoring/reports/drift_summary.json")


def drift_detected():
    if not SUMMARY_FILE.exists():
        return False

    with open(SUMMARY_FILE) as f:
        summary = json.load(f)

    return summary["drift_detected"]

summary = get_drift_summary()

if summary:
    print(summary["drift_share"])

summary = get_drift_summary()

if summary is None:
    print("No drift report found.")
    exit()

if summary["drift_share"] < 0.30:
    print("Drift below threshold. Skipping retraining.")
    exit()

print("Drift threshold exceeded.")