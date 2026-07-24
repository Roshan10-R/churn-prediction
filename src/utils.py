"""Small shared helpers used by train.py."""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(model, X_test, y_test) -> dict:
    """
    Return a dict of standard classification metrics.
    We care most about F1 and ROC-AUC here, not accuracy: churn is
    imbalanced (~27% positive class), so a model that always predicts
    "no churn" would still score ~73% accuracy while being useless.
    """
    preds = model.predict(X_test)

    # predict_proba gives us the probability needed for ROC-AUC
    proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def print_comparison_table(results: dict):
    """Pretty-print a metrics comparison across models."""
    print("\n" + "=" * 70)
    print(f"{'Model':<20}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1':<10}{'ROC-AUC':<10}")
    print("-" * 70)
    for name, metrics in results.items():
        print(
            f"{name:<20}"
            f"{metrics['accuracy']:<12.4f}"
            f"{metrics['precision']:<12.4f}"
            f"{metrics['recall']:<12.4f}"
            f"{metrics['f1']:<10.4f}"
            f"{metrics['roc_auc']:<10.4f}"
        )
    print("=" * 70 + "\n")
