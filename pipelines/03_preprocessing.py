#!/usr/bin/env python3
"""
Pipeline Step 3: Preprocessing & Feature Selection
Runs train/test splitting, statistical and biological feature selection, 
and sklearn pipeline transformations (scaling, imputation).
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_path):
    print(f"\\n{'='*70}\\nRunning {script_path}\\n{'='*70}")
    result = subprocess.run([sys.executable, script_path], check=False)
    if result.returncode != 0:
        print(f"\\n[ERROR] Pipeline failed at {script_path}")
        sys.exit(result.returncode)

def main():
    src_dir = Path("src/preprocessing")
    
    scripts = [
        "split.py",
        "pipeline.py",
        "biological_selector.py",
        "statistical_selector.py"
    ]
    
    for script in scripts:
        script_path = src_dir / script
        if script_path.exists():
            run_script(str(script_path))
        else:
            print(f"[WARNING] Script not found: {script_path}")
            
if __name__ == "__main__":
    main()
