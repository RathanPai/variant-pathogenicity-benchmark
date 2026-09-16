#!/usr/bin/env python3
"""
Pipeline Step 1: Data Collection & External Annotations
Runs the entire data ingestion process: ClinVar cleaning, gnomAD annotation,
VEP parsing, and UniProt functional extraction, before building the unified dataset.
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
    src_dir = Path("src/data")
    
    scripts = [
        "clinvar_cleaner.py",
        "gnomad_annotator.py",
        # "spdi_to_vcf.py", # Requires VEP CLI, may skip if VEP output already exists
        "vep_parser.py",
        "ensembl_to_uniprot.py",
        "uniprot_extractor.py",
        "dataset_builder.py"
    ]
    
    for script in scripts:
        script_path = src_dir / script
        if script_path.exists():
            run_script(str(script_path))
        else:
            print(f"[WARNING] Script not found: {script_path}")
            
if __name__ == "__main__":
    main()
