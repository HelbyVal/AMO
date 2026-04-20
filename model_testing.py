import joblib
from general_methods import load_data, create_features, evaluate


COLUMNS = [
    'date',
    'price_usd',
    'discount_percent',
    'is_sale_day',
    'price_source',
    'monthly_days_on_sale',
    'monthly_price_changes_count',
    'estimated_daily_sales'
]

def main():
    print("Loading model...")
    saved = joblib.load("model.pkl")

    model = saved["model"]
    features = saved["features"]

    print("Loading test data...")
    df = load_data("test")

    print("Creating features...")
    df = create_features(df)

    assert len(df) > 0, "Test dataset is empty!"

    print("Evaluating...")
    evaluate(model, df, features)

    print("Done!")


if __name__ == "__main__":
    main()