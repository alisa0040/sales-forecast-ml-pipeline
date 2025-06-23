import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features import create_features

# Load data
df_train = pd.read_csv("data/train.csv", parse_dates=["Date"], dtype={"StateHoliday": "str"}, low_memory=False)
df_store = pd.read_csv("data/store.csv")
df = pd.merge(df_train, df_store, on="Store", how="left")

# Clean NaNs
df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0)
df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(0)
df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0)
df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0)
df["PromoInterval"] = df["PromoInterval"].fillna("")
df["StateHoliday"] = df["StateHoliday"].astype(str).replace("0", "0")

# Use only open stores
df = df[df["Open"] == 1]

# Feature engineering
df = create_features(df)

# Split data
train_df = df[df["Date"] < "2015-06-01"]

# Features
features = [
    "Store", "DayOfWeek", "Month", "Year", "Promo", "SchoolHoliday",
    "CompetitionDistance", "CompetitionOpenMonths", "Promo2", "Promo2Active"
]
one_hot_cols = [col for col in df.columns if col.startswith("StoreType_") or col.startswith("Assortment_") or col.startswith("StateHoliday_")]
features += one_hot_cols

X_train = train_df[features]
y_train = np.log1p(train_df["Sales"])

# Train model
model = xgb.XGBRegressor(
    n_estimators=250,
    max_depth=12,
    learning_rate=0.15,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Save model
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/xgb_model.joblib")

print("Model trained and saved as models/xgb_model.joblib")