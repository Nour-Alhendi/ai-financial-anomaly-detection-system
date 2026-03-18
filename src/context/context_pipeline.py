import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from context.daily_context import run_daily_context
from context.trend_context import run_trend_context
from context.state_context import run_state_context

def run_context_pipeline():
    run_daily_context()
    run_trend_context()
    run_state_context()


if __name__ == "__main__":
    run_context_pipeline()