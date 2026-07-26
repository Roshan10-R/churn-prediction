"""
Generate data drift reports using Evidently.

Compares:

Reference Data (training)

vs

Current Production Data (prediction logs)

Outputs

monitoring/reports/drift_report.html

monitoring/reports/drift_metrics.json
"""

from pathlib import Path
import json

import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset


REFERENCE_FILE = Path("monitoring/reference_data.csv")
CURRENT_FILE = Path("monitoring/logs/prediction_log.csv")

REPORT_DIR = Path("monitoring/reports")
REPORT_DIR.mkdir(exist_ok=True)

HTML_REPORT = REPORT_DIR / "drift_report.html"
JSON_REPORT = REPORT_DIR / "drift_metrics.json"


def load_reference():
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            "Reference dataset not found. Run train.py first."
        )

    return pd.read_csv(REFERENCE_FILE)


def load_current():
    if not CURRENT_FILE.exists():
        raise FileNotFoundError(
            "Prediction log not found."
        )

    df = pd.read_csv(CURRENT_FILE)

    if df.empty:
        raise ValueError(
            "Prediction log is empty."
        )

    return df


def prepare_current(df):
    """
    Remove columns that are not model features.
    """

    drop_columns = [
        "request_id",
        "timestamp",
        "latency_ms",
        "prediction",
        "probability",
    ]

    existing = [c for c in drop_columns if c in df.columns]

    return df.drop(columns=existing)


def generate_report(reference_df, current_df):
    report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    snapshot = report.run(
        reference_data=reference_df,
        current_data=current_df,
    )

    snapshot.save_html(str(HTML_REPORT))
    snapshot.save_json(str(JSON_REPORT))

    report_dict = snapshot.dict()

    metrics = report_dict.get("metrics", [])

    drift_info = metrics[0]["value"]

    summary = {
        "reference_rows": len(reference_df),
        "current_rows": len(current_df),
        "drifted_columns": int(drift_info["count"]),
        "total_columns": len(reference_df.columns),
        "drift_share": round(drift_info["share"], 4),
        "drift_detected": drift_info["share"] >= 0.5,
        "model_health": (
            "Healthy"
            if drift_info["share"] < 0.5
            else "Retraining Recommended"
        ),
    }

    with open(
        REPORT_DIR / "drift_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=4)

    return summary
    report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    snapshot = report.run(
        reference_data=reference_df,
        current_data=current_df,
    )

    # Save HTML report
    snapshot.save_html(str(HTML_REPORT))

    # Save JSON report
    snapshot.save_json(str(JSON_REPORT))

    return snapshot.dict()
    report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    result = report.run(
        reference_data=reference_df,
        current_data=current_df,
    )

    # Save HTML report
    result.save_html(str(HTML_REPORT))

    # Save JSON report
    report_dict = result.dict()

    with open(JSON_REPORT, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)

    return report_dict

    report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    report.run(
        reference_data=reference_df,
        current_data=current_df,
    )

    report.save_html(str(HTML_REPORT))

    report_dict = report.as_dict()

    with open(JSON_REPORT, "w") as f:
        json.dump(report_dict, f, indent=4)

    return report_dict


def main():

    reference_df = load_reference()

    current_df = load_current()

    current_df = prepare_current(current_df)

    report = generate_report(
        reference_df,
        current_df,
    )

    print("✓ Drift report generated")
    print(f"HTML : {HTML_REPORT}")
    print(f"JSON : {JSON_REPORT}")

    return report


if __name__ == "__main__":
    main()