import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from quality.quality_pipeline              import run_quality_pipeline
from features.feature_pipeline             import run_feature_pipeline
from detection.detection_pipeline          import run as run_detection
from prediction.prediction_pipeline        import run_prediction_pipeline
from decision.decision_pipeline            import run as run_decision
from explainability.explainability_pipeline import run as run_explainability
from reporting.anomaly_log                 import log as log_anomalies
from reporting.daily_report                import run as run_daily_report
import pandas as pd
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    run_quality_pipeline()
    run_feature_pipeline()
    run_detection()
    run_prediction_pipeline()
    decisions_df = run_decision()
    run_explainability()
    log_anomalies(decisions_df)
    run_daily_report()


