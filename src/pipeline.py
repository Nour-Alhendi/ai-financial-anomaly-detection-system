import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from quality.quality_pipeline import run_quality_pipeline
from features.feature_pipeline import run as run_features
from detection.detection_pipeline import run as run_detection

if __name__ == "__main__":
    run_quality_pipeline()
    run_features()
    run_detection()


