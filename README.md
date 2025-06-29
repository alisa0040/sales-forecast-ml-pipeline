# Sales Forecast ML Pipeline

### End-to-End ML System for Real-Time Retail Revenue Prediction

A full-stack machine learning pipeline that predicts daily store sales using historical data and business signals.  
Built for production use: clean architecture, fast inference, Docker deployment, and Hugging Face integration.

---

## 🛠️ Tech Stack

| Layer             | Tools                           |
|------------------|----------------------------------|
| Data Processing   | `pandas`, `numpy`               |
| Modeling          | `xgboost`, `scikit-learn`, `optuna` |
| Serving API       | `fastapi`, `pydantic`, `uvicorn` |
| Deployment        | `Docker`, `Hugging Face Spaces` |
| Experimentation   | `matplotlib`, `seaborn`, `jupyter` |
| Logging           | `logging`, `curl`               |

---

## 📈 Model Performance

The model is trained on log-transformed daily sales and optimized using RMSPE — a metric well-suited for retail time series.

| Metric | Value     |
|--------|-----------|
| RMSPE  | ≈ 16.8%   | *(on validation set)*

---

## 🗂️ Project Structure

```bash
sales-forecast-ml-pipeline/
├── data/                  # Raw input data (CSV)
├── notebooks/             # Jupyter notebooks for EDA & prototyping
├── src/
│   └── features.py        # Feature engineering logic
├── fastapi_app/
│   └── main.py            # FastAPI API code
├── models/                # Trained model artifacts (e.g., .joblib)
├── train.py               # Model training pipeline
├── submission.csv         # Example submission format
├── test_input.json        # Sample API input for testing
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration for deployment
└── README.md              # Project documentation
```
##  Deployment (Hugging Face)
This API is hosted on Hugging Face Spaces:

- 🔧 API endpoint: [https://alisa0040-sales-forecast-api.hf.space](https://alisa0040-sales-forecast-api.hf.space)
- 📁 Deployment repository: [View on Hugging Face](https://huggingface.co/spaces/Alisa0040/sales-forecast-api/tree/main)

You can interact with the API using this command:

```bash
curl -X POST https://alisa0040-sales-forecast-api.hf.space/predict \
     -H "Content-Type: application/json" \
     -d @test_input.json
```
Example response:
{"predictions": [5504.970703125]}
---

## 🛠️ How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Alisa0040/sales-forecast-api.git
cd sales-forecast-api

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the API
uvicorn fastapi_app.main:app --reload
```
The API will be available at: http://localhost:8000

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

