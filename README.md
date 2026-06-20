# 📊 Customer Churn Prediction Platform

[![CI/CD](https://github.com/ManojRam7/CustomerChurn_Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/ManojRam7/CustomerChurn_Prediction/actions)
[![Docker Pulls](https://img.shields.io/docker/pulls/manojram7/customer-churn-api)](https://hub.docker.com/r/manojram7/customer-churn-api)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready machine learning platform for predicting telecom customer churn.  
Built with **FastAPI**, containerised with **Docker**, and shipped via **GitHub Actions CI/CD**.

---

## 🚀 Features

| Feature | Details |
|---|---|
| **ML Model** | Random Forest Classifier trained on 400K+ telecom records |
| **Web UI** | Responsive single-page form with live predictions |
| **REST API** | Batch & single-record predictions via `/predict` |
| **CI/CD** | Automated test → build → push on every push to `main` |
| **Containerised** | Docker image built for `linux/amd64`, pushed to Docker Hub |
| **Tests** | Unit tests (preprocessing) + integration tests (API) with coverage |

---

## 🏗️ Architecture

```
Browser / Postman / API Client
          │
          ▼
    FastAPI Application
    (uvicorn, port 8080)
          │
          ▼
  Random Forest Model (models/)
          ▲
          │
  Preprocessor (LabelEncoder + StandardScaler)
```

**CI/CD Pipeline:**
```
Push to main → pytest → docker buildx → Docker Hub
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| API Framework | FastAPI + Uvicorn |
| ML | scikit-learn (Random Forest), pandas, NumPy |
| Serialisation | joblib (`.pkl` model artifacts) |
| Frontend | Jinja2 HTML templates |
| Testing | pytest + httpx (FastAPI TestClient) |
| Containerisation | Docker |
| CI/CD | GitHub Actions + Docker Hub |

---

## 📁 Project Structure

```
CustomerChurn_Prediction/
├── app.py                        # FastAPI application (UI + API)
├── Dockerfile                    # Container build instructions
├── requirements.txt              # Production dependencies
│
├── src/                          # ML pipeline scripts
│   ├── preprocessing.py          # Preprocessor class (fit/transform)
│   └── model.py                  # Model training script
│
├── models/                       # Serialised model artefacts
│   ├── preprocessor.pkl
│   ├── random_forest_churn_from_script.pkl
│   └── model_columns.pkl
│
├── templates/
│   └── index.html                # Jinja2 web form template
│
├── notebooks/                    # Exploratory analysis
│   ├── EDA Analysis.ipynb
│   ├── Pre-Processing.ipynb
│   └── Model_training.ipynb
│
├── data/
│   ├── raw/                      # Original CSVs (train.csv, test.csv)
│   └── processed/                # Cleaned & preprocessed CSVs
│
├── reports/
│   └── eda_profile.html          # Automated EDA report
│
├── tests/
│   ├── conftest.py               # pytest hooks
│   ├── test_api.py               # API integration tests
│   └── test_preprocessing.py    # Preprocessor unit tests
│
└── .github/workflows/
    └── ci.yml                    # CI/CD pipeline
```

---

## ⚡ Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/ManojRam7/CustomerChurn_Prediction.git
cd CustomerChurn_Prediction
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Locally

```bash
uvicorn app:app --reload --port 8080
```

- Web UI → [http://localhost:8080](http://localhost:8080)
- API docs → [http://localhost:8080/docs](http://localhost:8080/docs)

### 3. Run Tests

```bash
pytest --cov=src --cov=app --cov-report=term -v
```

---

## 🐳 Docker

```bash
# Build
docker build --platform linux/amd64 -t manojram7/customer-churn-api:latest .

# Run
docker run -p 8080:8080 manojram7/customer-churn-api:latest
```

- Web UI → [http://localhost:8080](http://localhost:8080)
- API docs → [http://localhost:8080/docs](http://localhost:8080/docs)

---

## 🔄 CI/CD Pipeline

```
Push to main
    │
    ├─ 1. Checkout code
    ├─ 2. Set up Python 3.12
    ├─ 3. Install dependencies (with pip cache)
    ├─ 4. Run pytest with coverage report
    ├─ 5. Build Docker image (linux/amd64)
    └─ 6. Push to Docker Hub (:latest + :<git-sha>)
```

Secrets required: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`  
→ Add at **Settings → Secrets and variables → Actions**

---

## 🧪 API Reference

### `POST /predict`

**Request:**
```json
{
  "data": [
    {
      "CustomerID": 1,
      "Age": 35,
      "Gender": "Male",
      "Tenure": 12,
      "Usage_Frequency": 5,
      "Support_Calls": 2,
      "Payment_Delay": 0,
      "Subscription_Type": "Standard",
      "Contract_Length": "Quarterly",
      "Total_Spend": 1200.50,
      "Last_Interaction": 5
    }
  ]
}
```

**Response:** `{ "predictions": [0] }` — `0` = No Churn · `1` = Churn

### All Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI prediction form |
| `POST` | `/` | Submit form and get prediction |
| `POST` | `/predict` | REST API batch prediction |
| `GET` | `/health` | Health check |
| `GET` | `/model_info` | Model name, version, and feature list |
| `POST` | `/reload_model` | Hot-reload model from disk |
| `GET` | `/docs` | Interactive Swagger UI |

---

## 📊 Dataset

- **Domain:** Telecom customer behaviour
- **Size:** ~400K training records
- **Features:** CustomerID, Age, Gender, Tenure, Usage Frequency, Support Calls, Payment Delay, Subscription Type, Contract Length, Total Spend, Last Interaction
- **Target:** `Churn` (binary — 0 = stays, 1 = churns)

---

## 📈 Roadmap

- [ ] Add SHAP explainability endpoint (`/explain`)
- [ ] Add Prometheus metrics endpoint (`/metrics`)
- [ ] Add authentication for `/predict`
- [ ] Expand test coverage to 90%+

---

## 📄 License

MIT © [ManojRam7](https://github.com/ManojRam7)
