"""
Train, compare, and save the churn prediction model.

Fixes vs the original Week 1 notebook:
  1. train_with_mlflow() is fully defined (was cut off before).
  2. All three models are actually evaluated and compared (LR-only before).
  3. Every model is wrapped in a full Pipeline(preprocessor + classifier),
     so the object saved to disk does its own encoding/imputation.
     This is what makes the FastAPI /predict endpoint safe in Week 2 --
     no more relying on pd.get_dummies() matching training columns.
  4. The "best" model is chosen by F1 score, not just saved arbitrarily.

Run with:  python -m src.train   (from the project root)
"""

import json

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    METRICS_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.preprocess import build_preprocessor, get_features_and_target, load_and_clean_data
from src.utils import evaluate, print_comparison_table


def train_with_mlflow(run_name: str, pipeline: Pipeline, params: dict,
                       X_train, y_train, X_test, y_test) -> dict:
    """Fit a pipeline, log params/metrics/model to MLflow, return metrics."""
    with mlflow.start_run(run_name=run_name):
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        # Newer MLflow versions default to "skops" serialization, which
        # refuses to save numpy.dtype objects (present in our fitted
        # ColumnTransformer) as "untrusted". Pickle is the classic,
        # widely-compatible format and is fine for a local project like this.
        mlflow.sklearn.log_model(
            pipeline, artifact_path="model", serialization_format="pickle"
        )

    return metrics


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # --- Load & split data ---
    df = load_and_clean_data()
    X, y = get_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor()

    # --- Define candidate models, each wrapped with its own preprocessor copy ---
    candidates = {
        "Logistic Regression": (
            Pipeline([
                ("preprocessor", build_preprocessor()),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]),
            {"model_type": "LogisticRegression", "max_iter": 1000},
        ),
        "Random Forest": (
            Pipeline([
                ("preprocessor", build_preprocessor()),
                ("classifier", RandomForestClassifier(
                    n_estimators=200, random_state=RANDOM_STATE
                )),
            ]),
            {"model_type": "RandomForest", "n_estimators": 200, "max_depth": "None"},
        ),
        "XGBoost": (
            Pipeline([
                ("preprocessor", build_preprocessor()),
                ("classifier", XGBClassifier(
                    random_state=RANDOM_STATE, eval_metric="logloss"
                )),
            ]),
            {"model_type": "XGBoost", "eval_metric": "logloss"},
        ),
    }

    # --- Train + evaluate every candidate ---
    results = {}
    fitted_pipelines = {}
    for name, (pipeline, params) in candidates.items():
        print(f"Training {name}...")
        metrics = train_with_mlflow(name, pipeline, params, X_train, y_train, X_test, y_test)
        results[name] = metrics
        fitted_pipelines[name] = pipeline

    print_comparison_table(results)

    # --- Pick the best model by F1 (most balanced metric for imbalanced churn data) ---
    best_name = max(results, key=lambda name: results[name]["f1"])
    best_pipeline = fitted_pipelines[best_name]
    best_metrics = results[best_name]

    print(f"Best model: {best_name} (F1={best_metrics['f1']:.4f}, "
          f"ROC-AUC={best_metrics['roc_auc']:.4f})")

    # --- Save the FULL pipeline (preprocessing + model together) ---
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Saved best pipeline to {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump({"best_model": best_name, **best_metrics}, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
