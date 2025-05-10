#!/usr/bin/env python
"""
Create a new database migration.

This script creates a new database migration using Alembic.
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
    parser = argparse.ArgumentParser(description="Create a new database migration")
    
    parser.add_argument(
        "message",
        type=str,
        help="Migration message"
    )
    
    parser.add_argument(
        "--alembic-dir",
        type=str,
        default="alembic",
        help="Alembic directory (default: alembic)"
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


def create_migration(alembic_dir, message):
    """
    Create a new database migration.
    
    Args:
        alembic_dir: Alembic directory
        message: Migration message
    """
    logger.info(f"Creating migration: {message}")
    
    # Create migration
    cmd = [
        sys.executable, "-m", "alembic",
        "-c", os.path.join(alembic_dir, "alembic.ini"),
        "revision", "--autogenerate",
        "-m", message
    ]
    
    run_command(cmd)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Get the base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Change to the base directory
    os.chdir(base_dir)
    
    # Create migration
    create_migration(args.alembic_dir, args.message)
    
    logger.info("Migration created")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
