#!/usr/bin/env python3
"""
Run All Analysis Scripts for the Forex Trading Platform

This script runs all the analysis scripts for the forex trading platform in sequence,
generating a comprehensive analysis of the platform's architecture.

Usage:
    python run_all_analysis.py [--output-dir OUTPUT_DIR]

Options:
    --output-dir OUTPUT_DIR    Output directory for the analysis results (default: tools/output)
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

# Root directory of the forex trading platform
ROOT_DIR = "D:/MD/forex_trading_platform"

# Analysis scripts to run
ANALYSIS_SCRIPTS = [
    "analyze_dependencies_fixed.py",
    "analyze_service_structure.py",
    "analyze_database_schema.py",
    "analyze_api_endpoints.py",
    "visualize_architecture.py",
    "generate_comprehensive_diagram.py",
    "generate_simple_report.py"
]

def run_script(script_path: str) -> None:
    """Run a Python script."""
    print(f"Running {script_path}...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ["python", script_path],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True
        )
        
        print(f"Output: {result.stdout}")
        
        if result.stderr:
            print(f"Errors: {result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        print(f"Output: {e.stdout}")
        print(f"Errors: {e.stderr}")
    
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f} seconds")
    print()

def run_all_analysis() -> None:
    """Run all analysis scripts."""
    print("Starting analysis of the forex trading platform...")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(ROOT_DIR, 'tools', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Run each script
    for script in ANALYSIS_SCRIPTS:
        script_path = os.path.join(ROOT_DIR, 'tools', 'script', 'fix platform scripts', script)
        run_script(script_path)
    
    print("Analysis complete!")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Run all analysis scripts for the forex trading platform')
    parser.add_argument('--output-dir', default='tools/output', help='Output directory for the analysis results')
    args = parser.parse_args()
    
    run_all_analysis()

if __name__ == "__main__":
    main()