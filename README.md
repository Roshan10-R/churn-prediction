# Telco Customer Churn — MLOps Pipeline

Predicts the probability that a telecom customer will churn, served via a
FastAPI REST endpoint, with experiment tracking through MLflow and a
Dockerized deployment. Built as a resume/portfolio MLOps project.

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
```

### Why a full sklearn `Pipeline` instead of `pd.get_dummies`

An earlier version of this project used `pd.get_dummies()` on the whole
training dataframe and saved only the raw model. That breaks in production:
one-hot-encoding a single incoming API request produces different columns
than encoding the full training set, so the model receives mismatched
input. Wrapping `ColumnTransformer` (impute + scale + one-hot-encode) and
the classifier together in one `sklearn.Pipeline`, and saving *that whole
object*, guarantees preprocessing at inference time is identical to
training — even for a single customer record.

## Current model performance

(from the last training run, see `models/metrics.json`)

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

## Running locally (without Docker)

**1. Set up environment**
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Train the model** (this recreates `models/best_model.pkl` and logs
runs to a local `mlflow.db`; a pre-trained model is already included, so
this step is optional unless you want to retrain)
```bash
python -m src.train
```
Expected output ends with something like:
```
Best model: Logistic Regression (F1=0.6040, ROC-AUC=0.8419)
Saved best pipeline to .../models/best_model.pkl
```

To inspect experiment runs in the MLflow UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open `http://localhost:5000` in a browser.

**3. Run the API**
```bash
uvicorn app.main:app --reload
```
Open `http://localhost:8000/docs` for the interactive Swagger UI, or test
directly:
```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 70.5, "TotalCharges": 840.0
  }'
```
Expected response:
```json
{"churn_probability": 0.6959, "churn_prediction": true}
```

**4. Run tests**
```bash
pytest tests/ -v
```
All 6 tests should pass (health check, valid prediction, and four
input-validation cases for missing fields, wrong types, invalid category
values, and out-of-range numbers).

## Running with Docker

> Note: this was written and tested outside of a container (via `uvicorn`
> directly, above) since Docker wasn't available in the build sandbox.
> The Dockerfile follows standard practice, but build it once yourself to
> confirm the image works in your environment before relying on it for
> a demo.

**1. Build the image**
```bash
docker build -t churn-api .
```

**2. Run the container**
```bash
docker run -p 8000:8000 churn-api
```

**3. Test it** — same as above, hit `http://localhost:8000/docs` or
`http://localhost:8000/health` / `/predict` with curl.

**4. Stop it**
```bash
docker ps                 # find the container ID
docker stop <container_id>
```

## What's next (Week 3+)

- GitHub Actions CI/CD: run tests → build image → push to a registry on
  every push
- Deploy to Render/Railway for a live public endpoint
- Add drift detection (Evidently AI) and a monitoring dashboard
- Add a retraining trigger when drift crosses a threshold
