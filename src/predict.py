import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from features import create_features

#Load test and store data
df_test = pd.read_csv("data/test.csv", parse_dates=["Date"])
df_store = pd.read_csv("data/store.csv")
df = pd.merge(df_test, df_store, on="Store", how="left")

#  Clean missing values (как в train.py)
df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0)
df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(0)
df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0)
df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0)
df["PromoInterval"] = df["PromoInterval"].fillna("")
df["StateHoliday"] = df["StateHoliday"].astype(str).replace("0", "0")

#  Predict only for open stores
df["Sales"] = 0  # заполняем нулями заранее
df_open = df[df["Open"] == 1].copy()

# Feature engineering
df_open = create_features(df_open)

# Features used in training
features = [
    "Store", "DayOfWeek", "Month", "Year", "Promo", "SchoolHoliday",
    "CompetitionDistance", "CompetitionOpenMonths", "Promo2", "Promo2Active"
]
one_hot_cols = [
    col for col in df_open.columns 
    if col.startswith("StoreType_") or col.startswith("Assortment_") or col.startswith("StateHoliday_")
]
features += one_hot_cols

# Load model
model = joblib.load("models/xgb_model.joblib")

#  Make predictions
X_open = df_open[features]
y_pred_log = model.predict(X_open)
y_pred = np.expm1(y_pred_log)

# Fill predictions into the original df
df.loc[df["Open"] == 1, "Sales"] = y_pred

# Create submission file
submission = df[["Id", "Sales"]]
submission.to_csv("submission.csv", index=False)

print(" Submission saved as submission.csv")