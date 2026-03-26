import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from returns import run_returns
from volatility import run_volatility
from rolling_stats import run_rolling_stats
from beta import run_beta
from correlation import run_correlation
from rsi import run_rsi
from drawdown import run_drawdown

def run():
    print("--- Layer 3: Feature Engineering ---")
    run_returns()
    run_volatility()
    run_rolling_stats()
    run_beta()
    run_correlation()
    run_rsi()
    run_drawdown()


if __name__ == "__main__":
    run()
