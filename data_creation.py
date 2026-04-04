from datetime import date
from html import unescape
from io import StringIO
from pathlib import Path
import calendar
import json
import time
import re

import pandas as pd
import requests


STEAMBASE_URL = "https://steambase.io/games/stardew-valley/price"
STEAM_PRICE_HISTORY_URL = "https://steampricehistory.com/app/413150"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Connection": "close",
}

TRAIN_DIR = Path("train")
TEST_DIR = Path("test")
TRAIN_RATIO = 0.8
RELEASE_DATE = pd.Timestamp("2016-02-26")


def request_page(url: str) -> str:
    last_error = None

    for attempt in range(3):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as error:
            last_error = error
            if attempt < 2:
                time.sleep(2)

    raise last_error


def unwrap_astro_value(value):
    if isinstance(value, list) and len(value) == 2 and value[0] == 0:
        return value[1]
    return value


def split_dataframe(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(dataframe) < 2:
        raise ValueError("Not enough rows to split data into train and test.")

    split_index = int(len(dataframe) * TRAIN_RATIO)
    split_index = max(1, min(len(dataframe) - 1, split_index))

    train_df = dataframe.iloc[:split_index].copy()
    test_df = dataframe.iloc[split_index:].copy()
    return train_df, test_df


def prepare_directories() -> None:
    TRAIN_DIR.mkdir(exist_ok=True)
    TEST_DIR.mkdir(exist_ok=True)


def fetch_steam_price_history_events() -> pd.DataFrame:
    html = request_page(STEAM_PRICE_HISTORY_URL)
    tables = pd.read_html(StringIO(html))

    if len(tables) < 2:
        raise ValueError("Could not find event price history table.")

    dataframe = tables[1].copy()
    dataframe.columns = ["date", "price_usd", "gain_usd", "discount_text"]

    dataframe["date"] = pd.to_datetime(dataframe["date"])
    dataframe["price_usd"] = dataframe["price_usd"].str.replace("$", "", regex=False).astype(float)
    dataframe["discount_percent"] = (
        dataframe["discount_text"]
        .replace("-", "0%")
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace("-", "", regex=False)
        .astype(float)
    )

    dataframe = dataframe.drop(columns=["gain_usd", "discount_text"])
    dataframe = dataframe.sort_values("date").reset_index(drop=True)
    return dataframe


def fetch_steambase_monthly_data() -> pd.DataFrame:
    html = request_page(STEAMBASE_URL)

    match = re.search(
        r'component-url="/_astro/SteamPricesMonthlyBreakdownTable[^"]*"[^>]*props="([^"]+)"',
        html,
    )
    if match is None:
        raise ValueError("Could not find monthly price data on Steambase.")

    props = json.loads(unescape(match.group(1)))
    raw_items = props["items"][1]

    rows = []
    for item in raw_items:
        row = item[1]
        month = pd.to_datetime(unwrap_astro_value(row["month"])).tz_localize(None)
        rows.append(
            {
                "month": month,
                "base_price_usd": unwrap_astro_value(row["base_price"]) / 100,
                "lowest_price_usd": unwrap_astro_value(row["lowest_price"]) / 100,
                "days_on_sale": int(unwrap_astro_value(row["days_on_sale"])),
                "price_changes_count": int(unwrap_astro_value(row["price_changes_count"])),
            }
        )

    dataframe = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    return dataframe


def build_daily_from_events(events_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for index, row in events_df.iterrows():
        start_date = row["date"]
        if index + 1 < len(events_df):
            end_date = events_df.loc[index + 1, "date"] - pd.Timedelta(days=1)
        else:
            end_date = row["date"]

        daily_dates = pd.date_range(start=start_date, end=end_date, freq="D")
        for current_date in daily_dates:
            rows.append(
                {
                    "date": current_date,
                    "price_usd": row["price_usd"],
                    "discount_percent": row["discount_percent"],
                    "is_sale_day": int(row["discount_percent"] > 0),
                    "price_source": "exact_event_history",
                    "monthly_days_on_sale": pd.NA,
                    "monthly_price_changes_count": pd.NA,
                    "estimated_daily_sales": pd.NA,
                }
            )

    return pd.DataFrame(rows)


def build_gap_period(start_date: pd.Timestamp, end_date: pd.Timestamp, price_usd: float) -> pd.DataFrame:
    if start_date > end_date:
        return pd.DataFrame()

    rows = []
    for current_date in pd.date_range(start=start_date, end=end_date, freq="D"):
        rows.append(
            {
                "date": current_date,
                "price_usd": price_usd,
                "discount_percent": 0.0,
                "is_sale_day": 0,
                "price_source": "inferred_constant_price",
                "monthly_days_on_sale": pd.NA,
                "monthly_price_changes_count": pd.NA,
                "estimated_daily_sales": pd.NA,
            }
        )

    return pd.DataFrame(rows)


def build_daily_from_monthly(monthly_df: pd.DataFrame) -> pd.DataFrame:
    today = pd.Timestamp(date.today())
    rows = []

    for _, row in monthly_df.iterrows():
        month_start = row["month"]
        month_end_day = calendar.monthrange(month_start.year, month_start.month)[1]
        month_end = pd.Timestamp(month_start.year, month_start.month, month_end_day)

        if month_start.year == today.year and month_start.month == today.month:
            month_end = min(month_end, today)

        if month_end < month_start:
            continue

        sale_days = min(row["days_on_sale"], (month_end - month_start).days + 1)
        sale_discount = 0.0
        if row["base_price_usd"] > 0:
            sale_discount = round((1 - row["lowest_price_usd"] / row["base_price_usd"]) * 100, 2)

        if sale_days > 0:
            sale_start = month_end - pd.Timedelta(days=sale_days - 1)
        else:
            sale_start = month_end + pd.Timedelta(days=1)

        for current_date in pd.date_range(start=month_start, end=month_end, freq="D"):
            is_sale_day = int(current_date >= sale_start and sale_days > 0)
            if is_sale_day:
                price_usd = row["lowest_price_usd"]
                discount_percent = sale_discount
            else:
                price_usd = row["base_price_usd"]
                discount_percent = 0.0

            rows.append(
                {
                    "date": current_date,
                    "price_usd": price_usd,
                    "discount_percent": discount_percent,
                    "is_sale_day": is_sale_day,
                    "price_source": "estimated_from_monthly_summary",
                    "monthly_days_on_sale": row["days_on_sale"],
                    "monthly_price_changes_count": row["price_changes_count"],
                    "estimated_daily_sales": pd.NA,
                }
            )

    return pd.DataFrame(rows)


def build_full_daily_dataset() -> pd.DataFrame:
    events_df = fetch_steam_price_history_events()
    exact_daily_df = build_daily_from_events(events_df)
    frames = [exact_daily_df]

    try:
        monthly_df = fetch_steambase_monthly_data()
    except requests.RequestException as error:
        print(f"Warning: could not load Steambase monthly data: {error}")
        monthly_df = pd.DataFrame()

    gap_price = exact_daily_df.iloc[-1]["price_usd"]

    if monthly_df.empty:
        fallback_gap_df = build_gap_period(
            exact_daily_df["date"].max() + pd.Timedelta(days=1),
            pd.Timestamp(date.today()),
            gap_price,
        )
        frames.append(fallback_gap_df)
    else:
        gap_df = build_gap_period(
            exact_daily_df["date"].max() + pd.Timedelta(days=1),
            monthly_df["month"].min() - pd.Timedelta(days=1),
            gap_price,
        )
        monthly_daily_df = build_daily_from_monthly(monthly_df)
        frames.extend([gap_df, monthly_daily_df])

    dataframe = pd.concat(frames, ignore_index=True)
    dataframe = dataframe.sort_values("date").drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
    dataframe["date"] = pd.to_datetime(dataframe["date"]).dt.strftime("%Y-%m-%d")
    return dataframe


def save_split(dataframe: pd.DataFrame) -> None:
    train_df, test_df = split_dataframe(dataframe)

    train_path = TRAIN_DIR / "stardew_valley_price_history_train.csv"
    test_path = TEST_DIR / "stardew_valley_price_history_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved {train_path.name}: {len(train_df)} rows")
    print(f"Saved {test_path.name}: {len(test_df)} rows")


def main() -> None:
    prepare_directories()
    daily_df = build_full_daily_dataset()
    save_split(daily_df)
    print("Data creation completed successfully.")


if __name__ == "__main__":
    main()
