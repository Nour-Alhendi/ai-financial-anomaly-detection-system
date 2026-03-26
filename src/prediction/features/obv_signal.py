# Computes OBV Signal per ticker.
#
# obv_signal = volume_zscore * returns
#
# Interpretation:
#   positive → high volume + positive return  = institutional buying
#   negative → high volume + negative return  = panic selling / distribution
#   near 0   → low volume or no clear direction
#
# Output: obv_signal column written into data/detection/*.parquet

import pandas as pd
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data/detection"


def run():
    for f in sorted(DATA_DIR.glob("*.parquet")):
        if f.stem.startswith("^"):
            continue

        df = pd.read_parquet(f)

        if "volume_zscore" not in df.columns or "returns" not in df.columns:
            print(f"Skipping {f.stem} — missing volume_zscore or returns")
            continue

        df["obv_signal"] = df["volume_zscore"] * df["returns"]
        df.to_parquet(f)
        print(f"Updated: {f.name}")

    print("OBV Signal complete.")


if __name__ == "__main__":
    run()
