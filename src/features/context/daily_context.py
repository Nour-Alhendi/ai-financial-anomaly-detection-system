# Compares each stock's return against SPX and its sector ETF to identify market-wide vs stock-specific moves.
# Columns: spx_return, etf_return, is_market_wide, is_sector_wide, excess_return, relative_return, sector_relative

import pandas as pd
import yaml
from pathlib import Path

with open ("config/assets.yaml", "r") as f:
    config = yaml.safe_load(f)
assets = config["assets"]
sector_etfs = config["sector_etfs"]

INPUT_DIR = Path("data/detection")
OUTPUT_DIR = Path("data/features")
REF_DIR = Path("data/raw/references")
TICKERS = [a["ticker"] for a in assets if a["sector"] != "Index"]

# Load S&P 500 daily returns from detection data (already processed in Layer 4)
def load_spx():
    df = pd.read_parquet(INPUT_DIR / "^SPX.parquet")
    df = df.set_index("Date")
    return df[["returns"]].rename(columns={"returns": "spx_return"})

# Load all sector ETFs from reference data (downloaded in Layer 1)
def load_etf_data():
    etf_data = {}
    for etf in set(sector_etfs.values()):
        df = pd.read_parquet(REF_DIR / f"{etf}.parquet")
        df = df.set_index("Date")
        df = df[["Close"]].rename(columns={"Close": f"{etf}_close"})
        etf_data[etf] = df
    return etf_data

# Add market context to each stock: is the anomaly market-wide or stock-specific?
def compare(df, spx, etf_data, ticker):
    sector = next(a["sector"] for a in assets if a["ticker"] == ticker)
    etf_name = sector_etfs[sector]
    etf_df = etf_data[etf_name]
    df = df.set_index("Date")
    if "spx_return" in df.columns:
        df = df.drop(columns=["spx_return"])
    etf_col = f"{etf_name}_close"
    if etf_col in df.columns:
        df = df.drop(columns=[etf_col])
    df = df.join(spx, how="left")
    df = df.join(etf_df, how="left")

    # Calculate ETF return for sector context
    etf_col = f"{etf_name}_close"
    df["etf_return"] = df[etf_col].pct_change()

    # True if market also moved strongly that day (systemic, not stock-specific)
    df["is_market_wide"] = (df["spx_return"].abs() > 0.02) & df["combined_anomaly"]

    # True if sector also moved strongly that day (sector-wide, not stock-specific)
    df["is_sector_wide"] = (df["etf_return"].abs() > 0.025) & df["combined_anomaly"]

    # How much did the stock move beyond the market?
    df["excess_return"] = df["returns"] - df["spx_return"]

    # Relative performance vs market and sector
    df["relative_return"] = df["returns"] - df["spx_return"]
    df["sector_relative"] = df["returns"] - df["etf_return"]

    return df.reset_index()

# Main function: run daily context for all stocks and save results
def run_daily_context():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spx = load_spx()
    etf_data = load_etf_data()
    for ticker in TICKERS:
        file = INPUT_DIR / f"{ticker}.parquet"
        if not file.exists():
            print(f"Skipping {ticker}: file not found")
            continue
        df = pd.read_parquet(file)
        df = compare(df, spx, etf_data, ticker)
        df.to_parquet(OUTPUT_DIR / f"{ticker}.parquet")
        print(f"Saved: {ticker}.parquet")

if __name__ == "__main__":
    run_daily_context()
