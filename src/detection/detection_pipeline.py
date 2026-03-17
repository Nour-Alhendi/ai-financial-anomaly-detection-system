import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.statistical import run_zscore
from detection.isolation_forest import run_isolation_forest
from detection.autoencoder import run_autoencoder
from detection.combine import run_combine
from detection.severity import run_severity

def run():
    print("––– Layer 4: Anomaly Detection –––")
    run_zscore()
    run_isolation_forest()
    run_autoencoder()
    run_combine()
    print("––– Layer 5: Severity Scoring –––")
    run_severity()

if __name__ == "__main__":
    run()
