# Telco Customer Churn — MLOps Pipeline

Predicts the probability that a telecom customer will churn, served via a
FastAPI REST endpoint, with experiment tracking through MLflow and a
Dockerized deployment. Built as a  MLOps project.

## Architecture

```
Raw CSV data
      │
      ▼
src/preprocess.py   (cleaning + sklearn ColumnTransformer:
                      median-impute + scale numerics,
                      most-frequent-impute + one-hot-encode categoricals)
      │
      ▼
src/train.py         Trains Logistic Regression, Random Forest, and XGBoost,
                      each wrapped in a full Pipeline(preprocessor + model).
                      Logs params/metrics/model to MLflow for every run.
                      Picks the best model by F1 score and saves the
                      FULL pipeline to models/best_model.pkl.
      │
      ▼
app/main.py          FastAPI service. Loads models/best_model.pkl once at
                      startup. /predict takes raw customer fields (validated
                      by Pydantic), builds a one-row DataFrame, and hands it
                      straight to the pipeline — no manual encoding needed,
                      because the pipeline carries its own preprocessing.
      │
      ▼
Dockerfile            Packages the API + model into a container image.

## Current model performance



| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** (best) | 0.806 | 0.657 | 0.559 | **0.604** | 0.842 |
| Random Forest | 0.776 | 0.600 | 0.465 | 0.524 | 0.819 |
| XGBoost | 0.784 | 0.605 | 0.537 | 0.569 | 0.820 |

F1 and ROC-AUC are used to pick the winner rather than accuracy, since
churn is imbalanced (~27% positive class) — a model that always predicts
"no churn" would still score ~73% accuracy while being useless.

## Project structure

```
telco-churn-mlops/
├── app/
│   ├── main.py          FastAPI app (/health, /predict)
│   └── schemas.py        Pydantic request/response models
├── src/
│   ├── config.py          Paths, feature lists, constants
│   ├── preprocess.py      Data cleaning + ColumnTransformer builder
│   ├── train.py           Trains all 3 models, logs to MLflow, saves best
│   └── utils.py           Evaluation metrics + comparison table
├── tests/
│   └── test_api.py        API tests (health, valid/invalid predictions)
├── notebooks/
│   └── EDA.ipynb           Original exploratory analysis
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   ├── best_model.pkl      Saved pipeline (included, pre-trained)
│   └── metrics.json        Metrics for the saved best model
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── .gitignore
```




