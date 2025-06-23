import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based, promo and competition-related features.
    Assumes input has 'Date' column in datetime format.
    """
    df = df.copy()

    # --- Time-based features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    # --- Promo2 active (binary flag)
    df["Promo2Active"] = 0
    df.loc[
        (df["Promo2"] == 1) & (df["Promo2SinceYear"] > 0),
        "Promo2Active"
    ] = df.apply(
        lambda row: int(
            (row["Year"] > row["Promo2SinceYear"]) or
            (row["Year"] == row["Promo2SinceYear"] and row["WeekOfYear"] >= row["Promo2SinceWeek"])
        ),
        axis=1
    )

    # --- Months since competitor opened
    df["CompetitionOpenMonths"] = (
        (df["Year"] - df["CompetitionOpenSinceYear"]) * 12 +
        (df["Month"] - df["CompetitionOpenSinceMonth"])
    )
    df["CompetitionOpenMonths"] = df["CompetitionOpenMonths"].apply(lambda x: x if x > 0 else 0)

    # --- Category conversion
    df["StoreType"] = df["StoreType"].astype("category")
    df["Assortment"] = df["Assortment"].astype("category")
    df["StateHoliday"] = df["StateHoliday"].astype("category")

    return df