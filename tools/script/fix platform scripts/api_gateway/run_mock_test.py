#!/usr/bin/env python3
"""
Run mock API Gateway test.

This script runs the mock API Gateway and the test client in separate processes.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_mock_test")

def run_mock_api_gateway():
    """
    Run the mock API Gateway.

    Returns:
        Process object
    """
    logger.info("Starting mock API Gateway...")
    
    # Get script path
    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "mock_api_gateway_test.py"
    )
    
    # Start process
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Wait for API Gateway to start
    logger.info("Waiting for mock API Gateway to start...")
    time.sleep(5)
    
    return process

def run_test_client():
    """
    Run the test client.

    Returns:
        Process exit code
    """
    logger.info("Starting test client...")
    
    # Get script path
    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_mock_api_gateway.py"
    )
    
    # Start process
    process = subprocess.run(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Print output
    logger.info("Test client output:")
    for line in process.stdout.splitlines():
        logger.info(f"  {line}")
    
    # Print errors
    if process.stderr:
        logger.error("Test client errors:")
        for line in process.stderr.splitlines():
            logger.error(f"  {line}")
    
    return process.returncode

def main():
    """
    Main function.
    """
    logger.info("Running mock API Gateway test...")
    
    # Run mock API Gateway
    api_gateway_process = run_mock_api_gateway()
    
    try:
        # Run test client
        exit_code = run_test_client()
        
        # Exit with appropriate status code
        sys.exit(exit_code)
    finally:
        # Terminate API Gateway process
        logger.info("Terminating mock API Gateway...")
        api_gateway_process.terminate()
        api_gateway_process.wait()

if __name__ == "__main__":
    main()