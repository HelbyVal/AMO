import os
import glob
import pandas as pd
import numpy as np

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import joblib
from general_methods import load_data, create_features


# =====================
# Загрузка данных
# =====================



# =====================
# Обучение модели
# =====================
def train_model(train_df):
    target = 'price_usd'

    features = [
        'discount_percent',
        'is_sale_day',
        'monthly_days_on_sale',
        'monthly_price_changes_count',
        'estimated_daily_sales',

        'day',
        'month',
        'weekday',

        'price_lag_1',
        'price_lag_3',
        'price_lag_7',
        'rolling_mean_7',

        'price_source'
    ]

    X = train_df[features]
    y = train_df[target]

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6
    )

    model.fit(X, y)

    return model, features


# =====================
# Валидация
# =====================
def evaluate(model, df, features):
    X = df[features]
    y = df['price_usd']

    preds = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, preds))
    print(f"RMSE: {rmse:.4f}")


# =====================
# Основной пайплайн
# =====================
def main():
    print("Loading data...")

    train_df = load_data("train")


    print("Feature engineering...")
    train_df = create_features(train_df)

    print("Train shape:", train_df.shape)
    print(train_df.head())

    print("Training model...")
    model, features = train_model(train_df)

    print("Evaluating on train:")
    evaluate(model, train_df, features)


    print("Saving model...")
    joblib.dump({
        "model": model,
        "features": features
    }, "model.pkl")

    print("Done!")

if __name__ == "__main__":
    main()