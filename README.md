# 📊 Customer Churn Prediction Platform

[![CI/CD](https://github.com/ManojRam7/CustomerChurn_Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/ManojRam7/CustomerChurn_Prediction/actions)
[![Docker Pulls](https://img.shields.io/docker/pulls/manojram7/customer-churn-api)](https://hub.docker.com/r/manojram7/customer-churn-api)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready machine learning platform for predicting telecom customer churn.  
Built with **FastAPI**, containerised with **Docker**, and deployed to **Azure Container Apps** via **GitHub Actions CI/CD**.

---

## 🚀 Features

| Feature | Details |
|---|---|
| **ML Model** | Random Forest Classifier trained on 400K+ telecom records |
| **Web UI** | Responsive single-page form with live predictions |
| **REST API** | Batch & single-record predictions via `/predict` |
| **CI/CD** | Automated test → build → push → deploy on every push to `main` |
| **Containerised** | Docker image built for `linux/amd64`, pushed to Docker Hub |
| **Cloud Deployed** | Azure Container Apps (auto-scaling, zero-downtime updates) |
| **Tests** | Unit tests (preprocessing) + integration tests (API) with coverage |

---

## 🏗️ Architecture

```
Browser / Postman
      │
      ▼
Azure Container App  ←──── GitHub Actions CI/CD
  (FastAPI + ML)                │
      │                         ├── pytest (unit + integration)
      │                         ├── docker buildx → Docker Hub
      │                         └── az containerapp update
      ▼
Random Forest Model (models/)
      ▲
      │
Preprocessor (LabelEncoder + StandardScaler)
```

---

## 🛠️ Tech Stack

- **Language**: Python 3.12
- **API Framework**: FastAPI + Uvicorn
- **ML**: scikit-learn (Random Forest), pandas, NumPy
- **Serialisation**: joblib (`.pkl` model artifacts)
- **Frontend**: Jinja2 HTML templates
- **Testing**: pytest + httpx (FastAPI TestClient)
- **Containerisation**: Docker
- **CI/CD**: GitHub Actions
- **Cloud**: Azure Container Apps

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
│   └── eda_profile.html          # Automated EDA report (sweetviz)
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
pytest --cov=src --cov=app --cov-report=term
```

---

## 🐳 Docker

```bash
# Build
docker build --platform linux/amd64 -t manojram7/customer-churn-api:latest .

# Run
docker run -p 8080:8080 manojram7/customer-churn-api:latest
```

---

## ☁️ Azure Deployment

```bash
# Push image
docker push manojram7/customer-churn-api:latest

# Deploy (Azure CLI)
az containerapp update \
  --name
```

---

> 📸 Screenshot note: add a current UI screenshot (for example `docs/ui-screenshot.png`) and reference it here once available.

## Suggested Repository Topics

`fastapi`, `machine-learning`, `customer-churn`, `scikit-learn`, `docker`, `azure-container-apps`, `mlops`, `python`
