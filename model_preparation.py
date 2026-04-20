import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def load_data(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    df_list = [pd.read_csv(file_path) for file_path in files]
    df = pd.concat(df_list, ignore_index=True)
    return df


def create_features(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday

    numeric_defaults = {
        "discount_percent": 0.0,
        "is_sale_day": 0,
        "monthly_days_on_sale": 0,
        "monthly_price_changes_count": 0,
        "estimated_daily_sales": 0,
    }
    for column, default_value in numeric_defaults.items():
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(default_value)

    df["is_sale_day"] = df["is_sale_day"].astype(int)

    for lag in [1, 3, 7]:
        df[f"price_lag_{lag}"] = df["price_usd"].shift(lag)

    df["rolling_mean_7"] = df["price_usd"].rolling(7).mean()
    df["price_source"] = df["price_source"].fillna("unknown").astype("category")

    df = df.dropna().reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "Dataset is empty after feature engineering. "
            "Check source data and feature creation steps."
        )

    return df


def evaluate(model, df, features):
    X = df[features]
    y = df["price_usd"]

    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))

    print(f"RMSE: {rmse:.4f}")


def main():
    print("Done!")


if __name__ == "__main__":
    main()
