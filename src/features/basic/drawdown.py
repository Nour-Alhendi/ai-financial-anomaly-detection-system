# Calculates 30-day rolling maximum drawdown per ticker.
# max_drawdown_30d = worst peak-to-trough loss within a 30-day rolling window.
# Negative float, e.g. -0.04 = -4% max loss from peak over last 30 days.
#
# Used in:
#   - Layer 5: XGBoost feature (tail risk signal)
#   - Layer 6: Hard override threshold in Decision Engine

import pandas as pd
from pathlib import Path

INPUT_DIR  = Path("data/features")
OUTPUT_DIR = Path("data/features")
WINDOW     = 30


def _rolling_max_drawdown(close: pd.Series, window: int) -> pd.Series:
    values = []
    arr = close.values
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        window_data = arr[start:i + 1]
        peak = window_data.max()
        trough = window_data.min()
        drawdown = (trough - peak) / peak if peak != 0 else 0.0
        values.append(round(drawdown, 6))
    return pd.Series(values, index=close.index)


def run_drawdown():
    for file in INPUT_DIR.glob("*.parquet"):
        df = pd.read_parquet(file)
        if "Close" not in df.columns:
            print(f"Skipping {file.name} — no Close column")
            continue
        df["max_drawdown_30d"] = _rolling_max_drawdown(df["Close"], WINDOW)
        df.to_parquet(OUTPUT_DIR / file.name)
        print(f"Saved {file.name}")


if __name__ == "__main__":
    run_drawdown()
