#!/usr/bin/env python3
"""
Pyan Analyzer for the Forex Trading Platform

This script uses Pyan to analyze function/method dependencies in the forex trading platform.
It generates visualizations of the dependencies at different levels of granularity.

Usage:
    python pyan_analyzer.py [--service SERVICE] [--output-dir OUTPUT_DIR]

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
    'common-js-lib',
    '.venv'  # Include virtual environment for completeness
]

def check_pyan_installed() -> bool:
    """Check if Pyan is installed."""
    try:
        subprocess.run(['pyan3', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def find_python_files(directory: str) -> List[str]:
    """Find all Python files in a directory recursively."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def run_pyan(python_files: List[str], output_file: str, graph_type: str = 'classes') -> bool:
    """Run Pyan on the specified Python files."""
    try:
        cmd = [
            'pyan3',
            '--dot',
            f'--{graph_type}',
            '--no-defines',
            '--colored',
            '--grouped',
            '--annotated',
            *python_files,
            '-o', output_file
        ]
        
        subprocess.run(cmd, check=True)
        return True
    except subprocess.SubprocessError as e:
        print(f"Error running Pyan: {e}")
        return False

def generate_png_from_dot(dot_file: str) -> bool:
    """Generate a PNG file from a DOT file using Graphviz."""
    try:
        output_file = dot_file.replace('.dot', '.png')
        cmd = ['dot', '-Tpng', dot_file, '-o', output_file]
        subprocess.run(cmd, check=True)
        return True
    except subprocess.SubprocessError as e:
        print(f"Error generating PNG: {e}")
        return False

def analyze_service(service: str, output_dir: str) -> bool:
    """Analyze a specific service using Pyan."""
    service_dir = os.path.join(ROOT_DIR, service)
    if not os.path.isdir(service_dir):
        print(f"Error: Service directory {service_dir} not found")
        return False
    
    print(f"Analyzing {service}...")
    python_files = find_python_files(service_dir)
    if not python_files:
        print(f"No Python files found in {service_dir}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate class diagram
    class_dot_file = os.path.join(output_dir, f'classes_{service}_architecture.dot')
    if run_pyan(python_files, class_dot_file, 'classes'):
        if generate_png_from_dot(class_dot_file):
            print(f"Class diagram saved to {class_dot_file.replace('.dot', '.png')}")
    
    # Generate package diagram
    package_dot_file = os.path.join(output_dir, f'packages_{service}_architecture.dot')
    if run_pyan(python_files, package_dot_file, 'modules'):
        if generate_png_from_dot(package_dot_file):
            print(f"Package diagram saved to {package_dot_file.replace('.dot', '.png')}")
    
    return True

def analyze_full_project(output_dir: str) -> bool:
    """Analyze the full project using Pyan."""
    print("Analyzing full project...")
    all_python_files = []
    for service in SERVICE_DIRS:
        service_dir = os.path.join(ROOT_DIR, service)
        if os.path.isdir(service_dir):
            all_python_files.extend(find_python_files(service_dir))
    
    if not all_python_files:
        print("No Python files found in the project")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate class diagram
    class_dot_file = os.path.join(output_dir, 'classes_full_project.dot')
    if run_pyan(all_python_files, class_dot_file, 'classes'):
        if generate_png_from_dot(class_dot_file):
            print(f"Full project class diagram saved to {class_dot_file.replace('.dot', '.png')}")
    
    # Generate package diagram
    package_dot_file = os.path.join(output_dir, 'packages_full_project.dot')
    if run_pyan(all_python_files, package_dot_file, 'modules'):
        if generate_png_from_dot(package_dot_file):
            print(f"Full project package diagram saved to {package_dot_file.replace('.dot', '.png')}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Analyze the forex trading platform using Pyan')
    parser.add_argument('--service', help='Service to analyze (default: all services)')
    parser.add_argument('--output-dir', default='tools/output/architecture_diagrams', help='Output directory for visualization files')
    args = parser.parse_args()
    
    # Check if Pyan is installed
    if not check_pyan_installed():
        print("Error: Pyan is not installed. Please install it with:")
        print("pip install pyan3")
        sys.exit(1)
    
    # Create output directory
    output_dir = os.path.join(ROOT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    if args.service:
        if args.service == 'all':
            # Analyze all services individually
            for service in SERVICE_DIRS:
                analyze_service(service, output_dir)
            
            # Analyze full project
            analyze_full_project(output_dir)
        else:
            # Analyze specific service
            analyze_service(args.service, output_dir)
    else:
        # Analyze all services individually
        for service in SERVICE_DIRS:
            analyze_service(service, output_dir)
        
        # Analyze full project
        analyze_full_project(output_dir)
    
    print("\nAnalysis complete!")
    print(f"All visualization files saved to {output_dir}")

if __name__ == "__main__":
    main()
