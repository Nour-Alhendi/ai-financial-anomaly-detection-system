#==================================================
# Data Quality Pipeline
#Run all data validation checks
#==================================================
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from missing_check import run_missing_check
from duplicate_check import run_duplicate_check
from gap_check import run_gap_check
from schema_validation import run_schema_validation

# -----------------------------------------------------
# Function: run_quality_pipeline
# Purpose : Execute all quality checks sequentially
# -----------------------------------------------------

def run_quality_pipeline():
    print("\nStarting Data Quality Pipleline\n")
    run_schema_validation()
    run_duplicate_check()
    run_missing_check()
    run_gap_check()
    print("\nData Quality Pipeline Completed\n")

# -----------------------------------------------------
# Entry Point
# -----------------------------------------------------
if __name__ == "__main__":
    run_quality_pipeline()
