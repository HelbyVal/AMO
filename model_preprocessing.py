import numpy as np
import os

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import joblib
from model_preparation import load_data, create_features, evaluate


DEFAULT_MAX_ITER = 500


def read_max_iter():
    raw_value = os.getenv("MAX_ITER")
    if raw_value is None or raw_value.strip() == "":
        return DEFAULT_MAX_ITER

    try:
        max_iter = int(raw_value)
    except ValueError as exc:
        raise ValueError("MAX_ITER must be an integer.") from exc

    if max_iter <= 0:
        raise ValueError("MAX_ITER must be greater than 0.")

    return max_iter

# =====================
# Обучение модели
# =====================
def train_model(train_df, max_iter):
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
        n_estimators=max_iter,
        learning_rate=0.05,
        max_depth=6
    )

    model.fit(X, y)

    return model, features

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
    max_iter = read_max_iter()
    print(f"MAX_ITER: {max_iter}")
    model, features = train_model(train_df, max_iter)

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
