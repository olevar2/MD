#!/usr/bin/env python3
"""
PyDeps Analyzer for the Forex Trading Platform

This script uses PyDeps to analyze module dependencies in the forex trading platform.
It generates visualizations of the module dependencies for each service.

Usage:
    python pydeps_analyzer.py [--service SERVICE] [--output-dir OUTPUT_DIR]

Options:
    --service SERVICE       Service to analyze (default: all services)
    --output-dir OUTPUT_DIR Output directory for visualization files (default: tools/output/architecture_diagrams)
"""

import os
import sys
import subprocess
import argparse
from typing import List, Optional

# Root directory of the forex trading platform
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Service directories to analyze
SERVICE_DIRS = [
    'analysis-engine-service',
    'api-gateway',
    'data-management-service',
    'data-pipeline-service',
    'feature-store-service',
    'ml-integration-service',
    'ml-workbench-service',
    'model-registry-service',
    'monitoring-alerting-service',
    'portfolio-management-service',
    'risk-management-service',
    'strategy-execution-engine',
    'trading-gateway-service',
    'ui-service',
    'common-lib',
    'common-js-lib'
]

def check_pydeps_installed() -> bool:
    """Check if PyDeps is installed."""
    try:
        subprocess.run(['pydeps', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def run_pydeps(module_path: str, output_file: str) -> bool:
    """Run PyDeps on the specified module."""
    try:
        cmd = [
            'pydeps',
            '--noshow',
            '--max-bacon=10',
            '--cluster',
            '--rankdir=LR',
            '--exclude=tests,__pycache__,site-packages,dist-packages',
            f'--output={output_file}',
            module_path
        ]
        
        subprocess.run(cmd, check=True)
        return True
    except subprocess.SubprocessError as e:
        print(f"Error running PyDeps: {e}")
        return False

def analyze_service(service: str, output_dir: str) -> bool:
    """Analyze a specific service using PyDeps."""
    service_dir = os.path.join(ROOT_DIR, service)
    if not os.path.isdir(service_dir):
        print(f"Error: Service directory {service_dir} not found")
        return False
    
    print(f"Analyzing {service}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate module dependency diagram
    output_file = os.path.join(output_dir, f'{service}_module_dependencies.png')
    if run_pydeps(service_dir, output_file):
        print(f"Module dependency diagram saved to {output_file}")
        return True
    
    return False

def main():
    parser = argparse.ArgumentParser(description='Analyze the forex trading platform using PyDeps')
    parser.add_argument('--service', help='Service to analyze (default: all services)')
    parser.add_argument('--output-dir', default='tools/output/architecture_diagrams', help='Output directory for visualization files')
    args = parser.parse_args()
    
    # Check if PyDeps is installed
    if not check_pydeps_installed():
        print("Error: PyDeps is not installed. Please install it with:")
        print("pip install pydeps")
        sys.exit(1)
    
    # Create output directory
    output_dir = os.path.join(ROOT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    if args.service:
        if args.service == 'all':
            # Analyze all services
            for service in SERVICE_DIRS:
                analyze_service(service, output_dir)
        else:
            # Analyze specific service
            analyze_service(args.service, output_dir)
    else:
        # Analyze all services
        for service in SERVICE_DIRS:
            analyze_service(service, output_dir)
    
    print("\nAnalysis complete!")
    print(f"All visualization files saved to {output_dir}")

if __name__ == "__main__":
    main()
