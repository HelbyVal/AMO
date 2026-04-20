import pandas as pd
import glob
import os
import numpy as np
from sklearn.metrics import mean_squared_error

def load_data(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    return df

# =====================
# Feature Engineering
# =====================
def create_features(df):
    df = df.copy()

    # дата → datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # базовые временные признаки
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.weekday

    df['estimated_daily_sales'] = df['estimated_daily_sales'].fillna(0)

    # лаги
    for lag in [1, 3, 7]:
        df[f'price_lag_{lag}'] = df['price_usd'].shift(lag)

    # скользящее среднее
    df['rolling_mean_7'] = df['price_usd'].rolling(7).mean()

    # категориальный признак
    df['price_source'] = df['price_source'].astype('category')

    df = df.dropna()

    return df

def evaluate(model, df, features):
    X = df[features]
    y = df['price_usd']

    preds = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, preds))

    print(f"RMSE: {rmse:.4f}")

def main():
    print("Done!")
    
if __name__ == "__main__":
    main()