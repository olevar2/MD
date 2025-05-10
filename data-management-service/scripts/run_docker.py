#!/usr/bin/env python
"""
Run the service with Docker.

This script runs the service using Docker Compose.
"""

import argparse
import logging
import os
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the service with Docker")
    
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build the Docker image (default: False)"
    )
    
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Run in detached mode (default: False)"
    )
    
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the service (default: False)"
    )
    
    parser.add_argument(
        "--logs",
        action="store_true",
        help="Show logs (default: False)"
    )
    
    return parser.parse_args()


def run_command(cmd: list) -> int:
    """
    Run a command.
    
    Args:
        cmd: Command to run
        
    Returns:
        Exit code
    """
    logger.info(f"Running command: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
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
        logger.error(f"Command failed with exit code {process.returncode}")
        for line in process.stderr:
            print(line, end="")
    
    return process.returncode


def main():
    """Main entry point."""
    args = parse_args()
    
    # Get the base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Change to the base directory
    os.chdir(base_dir)
    
    # Stop the service
    if args.stop:
        logger.info("Stopping the service")
        run_command(["docker-compose", "down"])
        return 0
    
    # Show logs
    if args.logs:
        logger.info("Showing logs")
        run_command(["docker-compose", "logs", "-f"])
        return 0
    
    # Build the Docker image
    if args.build:
        logger.info("Building the Docker image")
        run_command(["docker-compose", "build"])
    
    # Run the service
    logger.info("Running the service")
    cmd = ["docker-compose", "up"]
    
    if args.detach:
        cmd.append("-d")
    
    return run_command(cmd)


if __name__ == "__main__":
    sys.exit(main())
