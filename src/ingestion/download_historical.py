import pandas as pd
import yaml
from pathlib import Path
from pandas_datareader import data as web
from datetime import datetime, timedelta

# Load asset configuration
with open("config/assets.yaml", "r") as f:
    config = yaml.safe_load(f)

assets = config["assets"]
references = config["references"]

end_date = datetime.today()
start_date = end_date - timedelta(days=365 * 10)

output_dir = Path("data/raw/raw_clean")
output_dir.mkdir(parents=True, exist_ok=True)

ref_dir = Path("data/raw/references")
ref_dir.mkdir(parents=True, exist_ok=True)

for asset in references:
    ticker = asset["ticker"]
    ticker_stooq = ticker.lower()
    print(f"Downloading reference {ticker} from Stooq...")
    try:
        df = web.DataReader(ticker_stooq, "stooq", start_date, end_date)
    except Exception as e:
        print(f"Failed {ticker}: {e}")
        continue
    if df.empty:
        print(f"No data for {ticker}")
        continue
    df = df.sort_index()
    df.reset_index(inplace=True)
    df.to_parquet(ref_dir / f"{ticker}.parquet")
    print(f"Saved {ticker}")

for asset in assets:
    ticker = asset["ticker"]

    # Stooq uses lowercase tickers
    ticker_stooq = ticker.lower()

    print(f"Downloading {ticker} from Stooq...")

    try:
        df = web.DataReader(ticker_stooq, "stooq", start_date, end_date)
    except Exception as e:
        print(f"Failed {ticker}: {e}")
        continue

    if df.empty:
        print(f"No data for {ticker}")
        continue

    df = df.sort_index()
    df.reset_index(inplace=True)

    file_path = output_dir / f"{ticker}.parquet"
    df.to_parquet(file_path)

    print(f"Saved {ticker}")

print("Download finished.")