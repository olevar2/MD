#!/usr/bin/env python3
"""
Forex Trading Platform Code Flow Analyzer

This script uses Code2flow to generate flowcharts from source code in the forex trading platform.
It helps visualize the flow of code execution and understand the relationships between functions.

Requirements:
- Code2flow: pip install code2flow
- Graphviz: https://graphviz.org/download/

Usage:
python code2flow_analyzer.py --service <service_name> [--output-dir <output_dir>] [--module <module_name>]
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PROJECT_ROOT = r"D:\MD\forex_trading_platform"
DEFAULT_OUTPUT_DIR = r"D:\MD\forex_trading_platform\tools\output\code2flow"

def run_code2flow(source_path: str, output_file: str) -> bool:
    """
    Run Code2flow on the specified source path.
    
    Args:
        source_path: Path to the source directory or file
        output_file: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Construct the Code2flow command
        cmd = ["code2flow", source_path, "--output", output_file]
        
        # Run Code2flow
        logger.info(f"Running Code2flow on {source_path}...")
        subprocess.run(cmd, check=True)
        
        logger.info(f"Code2flow output saved to {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running Code2flow: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

def find_python_modules(service_path: str) -> List[str]:
    """
    Find Python modules in a service directory.
    
    Args:
        service_path: Path to the service directory
        
    Returns:
        List of module paths
    """
    modules = []
    
    for root, dirs, files in os.walk(service_path):
        # Skip __pycache__ directories
        if "__pycache__" in root:
            continue
        
        # Check if this is a Python module
        if "__init__.py" in files:
            modules.append(root)
    
    return modules

def analyze_service(service_name: str, project_root: str, output_dir: str, module_name: Optional[str] = None) -> bool:
    """
    Analyze a specific service.
    
    Args:
        service_name: Name of the service to analyze
        project_root: Root directory of the project
        output_dir: Directory to save output files
        module_name: Name of the module to analyze (None for entire service)
        
    Returns:
        True if successful, False otherwise
    """
    # Construct service path
    service_path = os.path.join(project_root, service_name)
    
    # Check if service exists
    if not os.path.isdir(service_path):
        logger.error(f"Service directory not found: {service_path}")
        return False
    
    # Create output directory
    service_output_dir = os.path.join(output_dir, service_name)
    os.makedirs(service_output_dir, exist_ok=True)
    
    # If module_name is specified, analyze only that module
    if module_name:
        module_path = os.path.join(service_path, module_name.replace('.', os.path.sep))
        
        # Check if module exists
        if not os.path.exists(module_path):
            logger.error(f"Module not found: {module_path}")
            return False
        
        # Define output file
        output_file = os.path.join(service_output_dir, f"{module_name.replace('.', '_')}_flow.png")
        
        # Run Code2flow
        return run_code2flow(module_path, output_file)
    
    # Otherwise, analyze the entire service
    else:
        # Find Python modules
        modules = find_python_modules(service_path)
        
        if not modules:
            logger.error(f"No Python modules found in {service_path}")
            return False
        
        logger.info(f"Found {len(modules)} Python modules in {service_name}")
        
        # Analyze each module
        success = True
        for module_path in modules:
            # Get module name relative to service
            rel_path = os.path.relpath(module_path, service_path)
            module_name = rel_path.replace(os.path.sep, '_')
            
            # Define output file
            output_file = os.path.join(service_output_dir, f"{module_name}_flow.png")
            
            # Run Code2flow
            if not run_code2flow(module_path, output_file):
                success = False
        
        return success

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate flowcharts using Code2flow")
    parser.add_argument(
        "--service",
        required=True,
        help="Name of the service to analyze"
    )
    parser.add_argument(
        "--project-root",
        default=DEFAULT_PROJECT_ROOT,
        help="Root directory of the project"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output files"
    )
    parser.add_argument(
        "--module",
        help="Name of the module to analyze (default: entire service)"
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze service
    success = analyze_service(
        args.service,
        args.project_root,
        args.output_dir,
        args.module
    )
    
    if success:
        logger.info(f"Successfully analyzed {args.service}")
    else:
        logger.error(f"Failed to analyze {args.service}")
        sys.exit(1)

if __name__ == "__main__":
    main()
