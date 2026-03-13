import pandas as pd
from pathlib import Path

INPUT_DIR  = Path("data/features")
OUTPUT_DIR = Path("data/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# calculate beta

def beta(file_path):
    df = pd.read_parquet(file_path)
    market = pd.read_parquet(INPUT_DIR / "^SPX.parquet")
    #beta = Kovarianz(Aktie, Markt) / Varianz(Markt)
    cov = df["returns"].rolling(60).cov(market["returns"])
    var = market["returns"].rolling(60).var()
    df["beta"] = cov / var
    return df

# loop over all files and saves results
def run_beta():
    for file in INPUT_DIR.glob("*.parquet"):
        if file.name == "^SPX.parquet":
            continue
        df = beta(file)
        df.to_parquet(OUTPUT_DIR / file.name)
        print(f"Saved: {file.name}")

# Entry Point
if __name__ == "__main__":
    run_beta()