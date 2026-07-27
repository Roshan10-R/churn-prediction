# 🚀 Telco Customer Churn Prediction MLOps Pipeline

An end-to-end MLOps project that trains, tracks, serves, monitors, and deploys a machine learning model for predicting customer churn using the Telco Customer Churn dataset.

This project demonstrates the complete machine learning lifecycle, including experiment tracking, model versioning, API deployment, monitoring, drift detection, CI/CD automation, and cloud deployment.

---

## 📌 Features

- 📊 Exploratory Data Analysis
- 🤖 Multiple ML Models
  - Logistic Regression
  - Random Forest
  - XGBoost
- 🏆 Automatic Best Model Selection
- 📈 MLflow Experiment Tracking
- 📦 MLflow Model Registry
- 🌐 FastAPI Prediction API
- 📄 Interactive Swagger Documentation
- 🐳 Docker Containerization
- ⚙️ GitHub Actions CI/CD
- ☁️ Render Deployment
- 📉 Prediction Logging
- 📊 Streamlit Monitoring Dashboard
- 🔍 Data Drift Detection (Evidently AI)
- 🔄 Automatic Retraining Workflow

---

# 🏗️ Project Architecture


<img width="942" height="777" alt="image" src="https://github.com/user-attachments/assets/ea826b31-6663-4041-8f8e-f8c6de93679a" />


---

# 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Machine Learning | Scikit-learn, XGBoost |
| Experiment Tracking | MLflow |
| API | FastAPI |
| Validation | Pydantic |
| Dashboard | Streamlit |
| Monitoring | Evidently AI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deployment | Render |
| Version Control | Git & GitHub |

---

# 📂 Project Structure

```text
telco-churn-mlops/
│
├── app/
│   ├── main.py
│   └── schemas.py
│
├── src/
│   ├── train.py
│   ├── retrain.py
│   ├── config.py
│   └── preprocessing.py
│
├── monitoring/
│   ├── dashboard.py
│   ├── logger.py
│   ├── drift.py
│   ├── logs/
│   └── reports/
│
├── models/
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# 🚀 Local Setup

Clone the repository

```bash
git clone https://github.com/Roshan10-R/telco-churn-mlops.git

cd telco-churn-mlops
```

Install dependencies

```bash
pip install -r requirements.txt
```

Train the model

```bash
python -m src.train
```

Run the API

```bash
uvicorn app.main:app --reload
```

API Documentation

```
http://127.0.0.1:8000/docs
```

Run Streamlit Dashboard

```bash
streamlit run monitoring/dashboard.py
```

Dashboard

```
http://localhost:8501
```

---

# 🐳 Docker

Build Docker Image

```bash
docker build -t telco-churn-api .
```

Run Container

```bash
docker run -p 8000:8000 telco-churn-api
```

---

# 📈 MLflow

Start MLflow

```bash
mlflow ui
```

Open

```
http://127.0.0.1:5000
```

Features

- Experiment Tracking
- Metrics
- Parameters
- Model Artifacts
- Model Registry
- Production Alias

---

# 📊 Monitoring Dashboard

The Streamlit dashboard displays

- Total Predictions
- Prediction Distribution
- Average Latency
- Churn Probability Trend
- Recent Predictions
- Drift Status
- Drift Percentage
- Downloadable Drift Report

---

# 🔍 Drift Detection

Uses Evidently AI to compare

- Reference Dataset
- Production Predictions

Displays

- Drifted Features
- Drift Percentage
- Model Health
- HTML Drift Report

---

# 🔄 CI/CD Pipeline

GitHub Actions automatically

- Install Dependencies
- Train Model
- Run Tests
- Build Docker Image
- Push Image to GitHub Container Registry
- Deploy to Render

---

# 🌐 Deployment

## Live API

```
https://churn-api-svki.onrender.com/docs

```

Swagger Documentation

```
https://churn-api-svki.onrender.com/docs
```

Health Endpoint

```
https://churn-api-svki.onrender.com/health
```

---

# 📸 Screenshots



- MLflow Dashboard
- <img width="1881" height="912" alt="image" src="https://github.com/user-attachments/assets/55180091-6470-4a59-b3b0-c73f60bbf475" />

- Swagger UI
- <img width="1887" height="892" alt="image" src="https://github.com/user-attachments/assets/d4f79981-2d70-4c5a-829f-496ccc00d231" />

- Streamlit Dashboard
- <img width="1912" height="906" alt="image" src="https://github.com/user-attachments/assets/842b5749-98f5-43e3-a536-edcc71d03067" />
- <img width="1855" height="880" alt="image" src="https://github.com/user-attachments/assets/3c144c25-3506-478d-88bf-2024df87038f" />

- GitHub Actions
- 
- Render Deployment
- <img width="1477" height="786" alt="image" src="https://github.com/user-attachments/assets/8fa4ea05-cd26-460f-b2cb-3ffc7b58c691" />
-<img width="1877" height="876" alt="image" src="https://github.com/user-attachments/assets/d3bc5aba-bb5d-4f9f-a07b-7b6757dc39e0" />


---

# 📚 Future Improvements

- Kubernetes Deployment
- PostgreSQL Prediction Logging
- Prometheus Metrics
- Grafana Dashboard
- Authentication
- Model Explainability (SHAP)

---

# 👨‍💻 Author

**Roshan R**

Computer Science Engineering Student, Dayanand College of Engineering Student

Interested in

- Deep Learning
- MLOps
- Backend Development
- Cloud Computing

---

# ⭐ If you found this project useful

Please consider giving it a ⭐ on GitHub.
