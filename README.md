# Sales Forecast ML Pipeline

### End-to-End Machine Learning System for Retail Revenue Prediction

---

## Overview

This project is a **production-grade ML pipeline** that predicts daily sales for retail stores. It encompasses the complete lifecycle: from raw data and feature engineering to model training, API integration, Docker packaging, and deployment.

The project is fully reproducible, fast to deploy, and tested for real-time inference.


##  Tech Stack

| Layer             | Tools                         |
| ----------------- | ----------------------------- |
| Data Processing   | pandas, NumPy                 |
| Modeling          | XGBoost, scikit-learn, Optuna |
| API Interface     | FastAPI, Pydantic             |
| Deployment        | Docker, Uvicorn               |
| Logging / Testing | logging, curl                 |

---

## Model Performance

| Metric | Value  |
| ------ | ------ |
| RMSPE  | \~16.8% |

The model is trained on log-transformed sales data for variance stabilization, evaluated on a holdout test set, and validated with RMSPE (Root Mean Square Percentage Error).

---

##  Project Architecture

```
sales-forecast-ml-pipeline/
├── data/                  # Raw CSV data
├── notebooks/             # EDA + modeling notebooks
├── src/
│   ├── predict.py         # Submission for competition
│   └── features.py        # Feature generation
├── fastapi_app/
│   ├── main.py            # FastAPI app with endpoints
│   └── input_example.json # Example request payload
├── models/                # Trained XGBoost model
├── requirements.txt       # All Python dependencies
├── train.py               # Model training
├── Dockerfile             # Docker build instructions
└── README.md              # Project documentation
```

---

## Live API Demo

This project is deployed and running on **Hugging Face Spaces**.

🔗 **Live Endpoint:** [https://huggingface.co/spaces/Alisa0040/sales-forecast-api](https://huggingface.co/spaces/Alisa0040/sales-forecast-api)

🔍 **Interactive Swagger UI:** `https://huggingface.co/spaces/Alisa0040/sales-forecast-api/docs`

Supports `POST /predict` endpoint for batch sales predictions.

---

## 🛠️ How to Run Locally

```bash
# 1. Train model
python train.py

# 2. Launch API
uvicorn fastapi_app.main:app --reload
```

---

## 🐳 Docker Instructions

```bash
# Build image
docker build -t sales-api .

# Run container
docker run -p 8000:8000 sales-api
```

API will be available at `http://localhost:8000`

---

## Key Features

* 📊 Time-based feature extraction (Month, Year, Holidays)
* 📅 Promo and competition logic
* ✨ Log-target regression for skew correction
* ⚡ Fast, batched predictions via FastAPI
* 🚀 Docker-ready deployment in seconds
* 🌐 Hugging Face Space online deployment

---

