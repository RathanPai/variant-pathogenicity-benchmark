#!/usr/bin/env python3
"""
Master Pipeline Runner
Executes the entire end-to-end Bioinformatics and QML pipeline.
"""

import subprocess
import sys
from pathlib import Path

def run_pipeline(script_path):
    print(f"\\n{'#'*80}\\n# EXECUTING PIPELINE: {script_path.name}\\n{'#'*80}")
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    if result.returncode != 0:
        print(f"\\n[FATAL] Pipeline failed at {script_path.name}. Exiting.")
        sys.exit(result.returncode)

def main():
    pipelines_dir = Path("pipelines")
    
    stages = [
        "01_data_collection.py",
        "02_feature_engineering.py",
        "03_preprocessing.py",
        "04_train_models.py",
        "05_evaluate_benchmark.py"
    ]
    
    for stage in stages:
        script_path = pipelines_dir / stage
        if script_path.exists():
            run_pipeline(script_path)
        else:
            print(f"[WARNING] Pipeline stage not found: {script_path}")
            
    print(f"\\n{'#'*80}\\n# END-TO-END PIPELINE COMPLETE\\n{'#'*80}")

if __name__ == "__main__":
    main()
