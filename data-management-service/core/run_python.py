#!/usr/bin/env python
"""
Run the service with Python.

This script runs the service using Python.
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
    parser = argparse.ArgumentParser(description="Run the service with Python")
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 8000)),
        help="Port to bind to (default: 8000)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (default: False)"
    )
    
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.environ.get(
            "DATABASE_URL",
            "postgresql+asyncpg://postgres:postgres@localhost:5432/forex_platform"
        ),
        help="Database connection URL"
    )
    
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize the database (default: False)"
    )
    
    return parser.parse_args()


def run_command(cmd: list, env: dict = None) -> int:
    """
    Run a command.
    
    Args:
        cmd: Command to run
        env: Environment variables
        
    Returns:
        Exit code
    """
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Create environment
    cmd_env = os.environ.copy()
    if env:
        cmd_env.update(env)
    
    process = subprocess.Popen(
        cmd,
        env=cmd_env,
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
    
    # Set environment variables
    env = {
        "DATABASE_URL": args.db_url,
        "PORT": str(args.port)
    }
    
    # Initialize the database
    if args.init_db:
        logger.info("Initializing the database")
        run_command([sys.executable, "scripts/init_db.py"], env)
    
    # Run the service
    logger.info(f"Running the service on {args.host}:{args.port}")
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "data_management_service.main:app",
        "--host", args.host,
        "--port", str(args.port)
    ]
    
    if args.reload:
        cmd.append("--reload")
    
    return run_command(cmd, env)


if __name__ == "__main__":
    sys.exit(main())
