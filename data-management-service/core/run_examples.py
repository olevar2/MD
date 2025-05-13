#!/usr/bin/env python
"""
Run all examples.

This script runs all the example scripts to demonstrate the functionality of the
Historical Data Management service.
"""

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import time
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run all examples")
    
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize the database (default: False)"
    )
    
    parser.add_argument(
        "--init-data",
        action="store_true",
        help="Initialize sample data (default: False)"
    )
    
    parser.add_argument(
        "--examples",
        type=str,
        nargs="+",
        choices=["backtest", "ml", "correction", "quality"],
        default=["backtest", "ml", "correction", "quality"],
        help="Examples to run (default: all)"
    )
    
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="API URL (default: http://localhost:8000)"
    )
    
    return parser.parse_args()


def run_script(script_path: str, env: dict = None) -> int:
    """
    Run a script.
    
    Args:
        script_path: Path to the script
        env: Environment variables
        
    Returns:
        Exit code
    """
    logger.info(f"Running {script_path}")
    
    # Create environment
    script_env = os.environ.copy()
    if env:
        script_env.update(env)
    
    # Run script
    process = subprocess.Popen(
        [sys.executable, script_path],
        env=script_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Print output
    for line in process.stdout:
        print(line, end="")
    
    # Wait for process to complete
    process.wait()
    
    # Print errors
    if process.returncode != 0:
        logger.error(f"Script {script_path} failed with exit code {process.returncode}")
        for line in process.stderr:
            print(line, end="")
    
    return process.returncode


def main():
    """Main entry point."""
    args = parse_args()
    
    # Initialize database
    if args.init_db:
        logger.info("Initializing database")
        run_script("scripts/init_db.py")
    
    # Initialize sample data
    if args.init_data:
        logger.info("Initializing sample data")
        run_script("scripts/init_sample_data.py", {"API_URL": args.api_url})
    
    # Run examples
    for example in args.examples:
        if example == "backtest":
            logger.info("Running backtest example")
            run_script("scripts/backtest_example.py", {"API_URL": args.api_url})
        
        elif example == "ml":
            logger.info("Running ML example")
            run_script("scripts/ml_example.py", {"API_URL": args.api_url})
        
        elif example == "correction":
            logger.info("Running correction example")
            run_script("scripts/correction_example.py", {"API_URL": args.api_url})
        
        elif example == "quality":
            logger.info("Running quality report example")
            run_script("scripts/quality_report_example.py", {"API_URL": args.api_url})
    
    logger.info("All examples completed")


if __name__ == "__main__":
    main()
