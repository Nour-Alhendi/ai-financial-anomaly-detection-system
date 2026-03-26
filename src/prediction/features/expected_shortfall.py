# Computes Value at Risk (VaR 95%) and Expected Shortfall (ES 95%) per ticker.
#
# Two outputs:
#   1) Rolling ES per row → written back into data/detection/*.parquet (XGBoost features)
#   2) Snapshot (latest value per ticker) → data/risk/expected_shortfall.parquet (Layer 6)
#
# New columns added to detection parquets:
#   var_95    → rolling 252-day VaR (5th percentile of returns)
#   es_95     → rolling 252-day ES  (mean of returns below VaR)
#   es_ratio  → ES / VaR            (>1 = tail is worse than threshold)

import numpy as np
import pandas as pd
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[3]
DATA_DIR   = ROOT / "data/detection"
OUTPUT_DIR = ROOT / "data/risk"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW     = 252   # 1 trading year
CONFIDENCE = 0.95
MIN_OBS    = 30    # minimum observations needed to compute ES


def _rolling_es(returns: pd.Series, window: int, confidence: float):
    """Computes rolling VaR, ES, and ES-ratio for a returns Series."""
    var_vals = []
    es_vals  = []

    arr = returns.values
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        window_data = arr[start:i + 1]
        valid = window_data[~np.isnan(window_data)]

        if len(valid) < MIN_OBS:
            var_vals.append(np.nan)
            es_vals.append(np.nan)
            continue

        var = np.percentile(valid, (1 - confidence) * 100)
        tail = valid[valid <= var]
        es   = tail.mean() if len(tail) > 0 else var

        var_vals.append(var)
        es_vals.append(es)

    var_series = pd.Series(var_vals, index=returns.index)
    es_series  = pd.Series(es_vals,  index=returns.index)

    # es_ratio: both negative — >1 means tail is more extreme than threshold
    es_ratio = es_series / var_series.replace(0, np.nan)

    return var_series, es_series, es_ratio


def run():
    snapshots = []

    for f in sorted(DATA_DIR.glob("*.parquet")):
        if f.stem.startswith("^"):
            continue

        df = pd.read_parquet(f)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        if "returns" not in df.columns or len(df) < MIN_OBS:
            print(f"Skipping {f.stem} — insufficient data")
            continue

        # --- 1) Rolling ES per row ---
        var_s, es_s, ratio_s = _rolling_es(df["returns"], WINDOW, CONFIDENCE)
        df["var_95"]   = var_s.round(6)
        df["es_95"]    = es_s.round(6)
        df["es_ratio"] = ratio_s.round(4)

        df.to_parquet(f)
        print(f"Updated: {f.name}")

        # --- 2) Snapshot (latest row) for Layer 6 ---
        last = df.dropna(subset=["es_95"]).iloc[-1]
        snapshots.append({
            "ticker":   f.stem,
            "var_95":   last["var_95"],
            "es_95":    last["es_95"],
            "es_ratio": last["es_ratio"],
        })

    result = (
        pd.DataFrame(snapshots)
        .sort_values("es_95")
        .reset_index(drop=True)
    )

    out_path = OUTPUT_DIR / "expected_shortfall.parquet"
    result.to_parquet(out_path, index=False)
    print(f"\nSnapshot saved: {out_path}")
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    run()
