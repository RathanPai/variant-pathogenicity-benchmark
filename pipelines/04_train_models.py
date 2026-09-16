#!/usr/bin/env python3
"""
Pipeline Step 4: Model Training
Trains Classical (RF, XGBoost, LR), Deep Tabular (MLP, NODE, KAN, FT-Transformer),
and Quantum models on the processed dataset.
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
    src_dir = Path("src/models")
    
    scripts = [
        "classical.py",
        "mlp.py",
        "ft_transformer.py",
        "kan.py",
        "node.py",
        "qnn.py"
    ]
    
    for script in scripts:
        script_path = src_dir / script
        if script_path.exists():
            run_script(str(script_path))
        else:
            print(f"[WARNING] Script not found: {script_path}")
            
if __name__ == "__main__":
    main()
