from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import uvicorn
import os
import traceback

from src.features import create_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Pydantic input schema
class InputData(BaseModel):
    Store: int
    DayOfWeek: int
    Date: str
    Promo: int
    StateHoliday: str
    SchoolHoliday: int
    StoreType: str
    Assortment: str
    CompetitionDistance: float = Field(default=0)
    CompetitionOpenSinceMonth: int = Field(default=0)
    CompetitionOpenSinceYear: int = Field(default=0)
    Promo2: int
    Promo2SinceWeek: int = Field(default=0)
    Promo2SinceYear: int = Field(default=0)
    PromoInterval: str = Field(default="")
    Open: int = Field(default=1)

class InputBatch(BaseModel):
    inputs: List[InputData]

app = FastAPI(title="Sales Forecast API", version="1.0")

# load model
MODEL_PATH = "models/xgb_model.joblib"
model = joblib.load(MODEL_PATH)
logger.info("Model loaded successfully")

@app.post("/predict")
def predict(batch: InputBatch):
    try:
        df = pd.DataFrame([item.dict() for item in batch.inputs])
        logger.info(f"Received batch of size {len(df)}")

        # convert Date to datetime
        df["Date"] = pd.to_datetime(df["Date"])

        # Only open 
        df = df[df["Open"] == 1]
        if df.empty:
            return {"predictions": []}

        df = create_features(df)

        features = [
            "Store", "DayOfWeek", "Month", "Year", "Promo", "SchoolHoliday",
            "CompetitionDistance", "CompetitionOpenMonths", "Promo2", "Promo2Active"
        ]
        one_hot_cols = [col for col in df.columns if col.startswith("StoreType_") or col.startswith("Assortment_") or col.startswith("StateHoliday_")]
        full_features = features + one_hot_cols

        # prediction 
        y_pred_log = model.predict(df[full_features])
        y_pred = np.expm1(y_pred_log)

        return {"predictions": y_pred.tolist()}

    except Exception as e:
        logger.error("Prediction failed: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Prediction failed")

# Healthcheck
@app.get("/ping")
def ping():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.post("/reload_model")
def reload_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model reloaded")
        return {"status": "Model reloaded"}
    except Exception as e:
        logger.error("Model reload failed: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Reload failed")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("fastapi_app.main:app", host="0.0.0.0", port=port, reload=True)